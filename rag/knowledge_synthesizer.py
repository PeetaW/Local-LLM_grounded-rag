import time
import logging
import requests
import config as cfg

logger = logging.getLogger(__name__)

_BASE_SYNTHESIS_SYSTEM_PROMPT = """
你是一個學術文獻整理助手。
你的唯一任務是將論文段落整理成結構化的已知事實清單。

嚴格規則：
1. 只陳述文獻中明確出現的資訊，禁止推論或補充背景知識
2. 每條事實必須標注來源，格式：（來源：[論文名稱或chunk ID]）
3. 輸出為編號清單，每條一行，格式：[事實N] 內容（來源：XXX）
4. 若多個 chunk 描述同一事實，合併為一條並列出所有來源
5. 使用繁體中文輸出（無論輸入語言）
6. 禁止輸出任何形式的推論、假設或背景補充
"""

_TERM_FIDELITY_RULES = """
7. 專有名詞保真：酵素、試劑、化合物、方法名、模型名必須保留原文英文拼法；可加中文說明，但英文原詞不可省略或替換
8. 禁止近義替換：chymotrypsin 與 trypsin 是不同酵素；原文寫哪一個就逐字保留哪一個
"""

_COMPARISON_QUERY_HINTS = (
    "compare", "comparison", "different", "difference", "route", "routes",
    "approach", "approaches", "scalability", "cost-effectiveness", "cost",
    "isotopic", "enrichment", "比較", "差異", "不同", "路線", "策略",
    "可擴展", "放大", "成本", "同位素", "富集",
)


def _system_prompt() -> str:
    if getattr(cfg, "TERM_FIDELITY_GUARD_ENABLED", False):
        return _BASE_SYNTHESIS_SYSTEM_PROMPT + _TERM_FIDELITY_RULES
    return _BASE_SYNTHESIS_SYSTEM_PROMPT


def _is_comparison_query(query: str) -> bool:
    text = (query or "").lower()
    return any(hint in text for hint in _COMPARISON_QUERY_HINTS)


def _comparison_schema_instruction(query: str) -> str:
    if not getattr(cfg, "STAGE3_COMPARISON_SCHEMA_ENABLED", False):
        return ""
    if not _is_comparison_query(query):
        return ""
    return """
【比較題蒸餾格式】
若問題要求比較，請用下列小節整理事實；沒有資料的小節可省略：

[source_roles]
- source: ...; role: route / review/comparison source / background; evidence: ...（來源：...）

[direct_route_evidence]
- 只放直接合成目標化合物的路線事實。

[review_comparison_evidence]
- 若 Source metadata 出現 role_hint=review/comparison source，保留它是綜述/比較來源，不要改寫成該文提供單一路線。

[dimension_evidence]
- scalability: ...
- cost-effectiveness: ...
- isotopic enrichment: ...

額外規則：
- Source metadata 與 guidance lines 只用來判斷來源角色；它們 are not paper evidence，不可當作可引用論文事實。
- review/comparison source 若描述某路線，只能寫「該綜述/比較來源報導或比較某路線」，不要寫成「該論文提供/實作該合成路線」。
- 保留問題指名的比較面向；與面向無關的實驗條件、產率、試劑細節可省略。
""".strip()


def _build_user_prompt(formatted: str, query: str) -> str:
    schema = _comparison_schema_instruction(query)
    schema_block = f"\n\n{schema}" if schema else ""
    return (
        f"參考問題方向（僅供整理聚焦，不影響事實陳述）：{query}{schema_block}\n\n"
        f"請將以下論文段落整理為結構化已知事實清單：\n\n"
        f"--- 論文段落開始 ---\n{formatted}\n--- 論文段落結束 ---"
    )


class KnowledgeSynthesizer:
    def __init__(
        self,
        model: str = None,
        ollama_base_url: str = None,
        timeout: int = 21600
    ):
        self.model = model or cfg.SYNTHESIS_MODEL
        self.base_url = ollama_base_url or cfg.OLLAMA_BASE_URL
        self.timeout = timeout

    def _format_chunks(self, chunks: list[dict]) -> str:
        """
        將 chunks 格式化為可讀字串。
        chunks 格式：list of dict，每個 dict 含 "text" 和 "source" 欄位。
        若 dict 沒有 source 欄位，嘗試從 metadata 取，否則標注為「來源不明」。
        """
        lines = []
        for i, chunk in enumerate(chunks):
            # 相容不同的欄位名稱
            text = chunk.get("text") or chunk.get("content") or str(chunk)
            source = (
                chunk.get("source")
                or chunk.get("paper_name")
                or chunk.get("file_name")
                or (chunk.get("metadata") or {}).get("file_name")
                or f"chunk_{i}"
            )
            lines.append(f"[Chunk {i+1}] 來源：{source}\n{text}\n---")
        return "\n".join(lines)

    def synthesize(
        self,
        chunks: list[dict],
        query: str = "",
        on_status=None,
    ) -> str:
        """
        將 chunks 轉化為結構化已知事實清單。
        失敗時 fallback 到直接串接 chunk text，不中斷 pipeline。
        """
        if not chunks:
            return "（無檢索結果）"

        formatted = self._format_chunks(chunks)
        total_chars = sum(len(c.get("text","")) for c in chunks)

        user_prompt = _build_user_prompt(formatted, query)

        def _status(msg):
            if on_status:
                on_status(msg)
            else:
                print(msg)

        logger.info(
            "[Synthesizer] Starting: %d chunks (%d chars), query=\"%s\"",
            len(chunks), total_chars, query[:50]
        )
        t0 = time.time()

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model":  self.model,
                    "system": _system_prompt(),
                    "prompt": user_prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 8192,
                        "num_ctx": cfg.STAGE3_NUM_CTX,
                    }
                },
                timeout=self.timeout,
                stream=True,
            )
            resp.raise_for_status()

            import json as _json
            chunks_out = []
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = _json.loads(raw_line)
                except _json.JSONDecodeError:
                    continue
                token = chunk.get("response", "")
                if token:
                    print(token, end="", flush=True)
                    chunks_out.append(token)
                if chunk.get("done"):
                    break
            print()  # 換行
            result = "".join(chunks_out).strip()

            elapsed = time.time() - t0
            logger.info(
                "[Synthesizer] Done: input %d chars → output %d chars (%.1fs)",
                total_chars, len(result), elapsed
            )
            _status(
                f"  📋 [Synthesizer] {len(chunks)} chunks → "
                f"{len(result)} chars ({elapsed:.1f}s)"
            )
            return result

        except Exception as e:
            elapsed = time.time() - t0
            logger.warning("[Synthesizer] FALLBACK: %s (%.1fs)", e, elapsed)
            _status(f"  ⚠️  [Synthesizer] 失敗，使用原始 chunks ({e})")
            # Fallback：直接串接原始 chunk text
            return "\n\n".join(
                f"[Chunk {i+1}] {c.get('text','')}"
                for i, c in enumerate(chunks)
            )
