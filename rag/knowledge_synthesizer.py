import time
import logging
import requests
import config as cfg

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """
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

        user_prompt = (
            f"參考問題方向（僅供整理聚焦，不影響事實陳述）：{query}\n\n"
            f"請將以下論文段落整理為結構化已知事實清單：\n\n"
            f"--- 論文段落開始 ---\n{formatted}\n--- 論文段落結束 ---"
        )

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
                    "system": SYNTHESIS_SYSTEM_PROMPT,
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
