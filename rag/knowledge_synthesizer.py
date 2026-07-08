import json
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
    "safety", "isotopic", "enrichment", "比較", "差異", "不同", "路線", "策略",
    "可擴展", "放大", "成本", "安全", "同位素", "富集",
)
_ABSENCE_MARKERS = (
    "does not explicitly", "not explicitly", "does not contain", "not contain",
    "did not provide", "does not provide", "not provide", "not reported",
    "unaddressed", "lacks", "missing",
)


def _system_prompt() -> str:
    prompt = _BASE_SYNTHESIS_SYSTEM_PROMPT
    if getattr(cfg, "STAGE3_ENGLISH_DISTILLATION_ENABLED", False):
        prompt = prompt.replace(
            "5. 使用繁體中文輸出（無論輸入語言）",
            "5. Output in English. Preserve exact source wording for technical terms, route-defining phrases, values, and comparison dimensions.",
        )
    if getattr(cfg, "TERM_FIDELITY_GUARD_ENABLED", False):
        return prompt + _TERM_FIDELITY_RULES
    return prompt


def _is_comparison_query(query: str) -> bool:
    text = (query or "").lower()
    return any(hint in text for hint in _COMPARISON_QUERY_HINTS)


def _query_dimension_keys(query: str) -> set[str]:
    text = (query or "").lower()
    keys = set()
    if "isotopic" in text or "10b" in text or "同位素" in text:
        keys.add("isotopic_enrichment")
    if "scalability" in text or "可擴展" in text or "放大" in text:
        keys.add("scalability")
    if "cost" in text or "成本" in text:
        keys.add("cost_effectiveness")
    if "safety" in text or "安全" in text:
        keys.add("safety")
    return keys


def _has_absence_claim(text: str) -> bool:
    lower = str(text or "").lower()
    return any(marker in lower for marker in _ABSENCE_MARKERS)


def _has_review_comparison_source(comparison: dict) -> bool:
    source_roles = comparison.get("source_roles") if isinstance(comparison.get("source_roles"), list) else []
    review_sources = comparison.get("review_comparison_sources")
    return (
        any(
            isinstance(item, dict) and "review/comparison" in str(item.get("role", ""))
            for item in source_roles
        )
        or bool(review_sources)
    )


def _comparison_json_payload(text: str):
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start > 0 and end > start:
        stripped = stripped[start:end + 1]
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _normalize_comparison_json(text: str, query: str = "") -> str:
    data = _comparison_json_payload(text)
    if not isinstance(data, dict):
        return text
    comparison = data.get("comparison_json") if isinstance(data, dict) else None
    if not isinstance(comparison, dict):
        return text
    comparison.setdefault("source_roles", [])
    comparison.setdefault("direct_routes", [])
    comparison.setdefault("review_comparison_sources", [])
    dimensions = comparison.setdefault("dimensions", {})
    query_dims = _query_dimension_keys(query)
    for key in ("isotopic_enrichment", "scalability", "cost_effectiveness", "safety"):
        item = dimensions.setdefault(key, {})
        item["requested"] = key in query_dims or bool(item.get("requested", False))
        item["evidence_found"] = bool(item.get("evidence_found", item.get("present", False)))
        item["present"] = item["evidence_found"]
        item.setdefault("text", "")
        item.setdefault("sources", [])
    review_sources = [
        item.get("source") for item in comparison.get("source_roles", [])
        if isinstance(item, dict) and "review/comparison" in str(item.get("role", ""))
    ]
    if review_sources:
        comparison["direct_routes"] = [
            route for route in comparison.get("direct_routes", [])
            if not isinstance(route, dict) or route.get("source") not in review_sources
        ]
    comparison.setdefault("central_tradeoff", "")
    return json.dumps(data, ensure_ascii=False, indent=2)


def _comparison_json_validation_errors(text: str, query: str = "") -> list[str]:
    data = _comparison_json_payload(text)
    if not isinstance(data, dict):
        return ["Output is not valid JSON."]
    comparison = data.get("comparison_json")
    if not isinstance(comparison, dict):
        return ["Missing root object: comparison_json."]

    errors = []
    for field in ("source_roles", "direct_routes", "review_comparison_sources"):
        if not isinstance(comparison.get(field), list):
            errors.append(f"`{field}` must be a list.")

    dimensions = comparison.get("dimensions")
    if not isinstance(dimensions, dict):
        errors.append("`dimensions` must be an object.")
        dimensions = {}

    query_dims = _query_dimension_keys(query)
    has_review_source = _has_review_comparison_source(comparison)
    for key in query_dims:
        item = dimensions.get(key)
        if not isinstance(item, dict):
            errors.append(f"`dimensions.{key}` is missing.")
            continue
        if not item.get("requested"):
            errors.append(f"`dimensions.{key}.requested` must be true because the question asks for it.")
        if item.get("evidence_found") and (not item.get("text") or not item.get("sources")):
            errors.append(f"`dimensions.{key}` says evidence_found=true but lacks text or sources.")
        if (
            key in ("scalability", "cost_effectiveness")
            and has_review_source
            and not item.get("evidence_found")
            and _has_absence_claim(item.get("text", ""))
        ):
            errors.append(
                f"`dimensions.{key}` is requested but marked missing with absence wording; "
                "re-check qualitative review/comparison evidence before setting evidence_found=false."
            )

    source_roles = comparison.get("source_roles") if isinstance(comparison.get("source_roles"), list) else []
    review_sources = {
        item.get("source") for item in source_roles
        if isinstance(item, dict) and "review/comparison" in str(item.get("role", ""))
    }
    for route in comparison.get("direct_routes", []) if isinstance(comparison.get("direct_routes"), list) else []:
        if not isinstance(route, dict):
            continue
        if route.get("source") in review_sources:
            errors.append(f"Review/comparison source `{route.get('source')}` must not appear in direct_routes.")
        route_blob = " ".join(str(route.get(k, "")) for k in ("source", "route_phrase", "evidence")).lower()
        if any(term in route_blob for term in ("derivative", "formulation", "solubility", "biological propert")):
            errors.append("Derivative/formulation/solubility/biological-property source must not be a direct target-compound route.")

    tradeoff = str(comparison.get("central_tradeoff", "")).lower()
    if _has_absence_claim(tradeoff) and query_dims:
        errors.append("central_tradeoff contains an absence claim for requested comparison dimensions.")
    return errors


def _comparison_json_repair_prompt(original_prompt: str, current_json: str, errors: list[str]) -> str:
    return f"""
Repair the comparison_json below. Return exactly one valid JSON object and no Markdown fence.

Validation errors:
{chr(10).join(f"- {err}" for err in errors)}

Rules:
- Keep the same schema.
- Use only evidence from the original prompt below.
- Set `requested=true` for dimensions asked by the question.
- Set `evidence_found=true` only when the retrieved evidence supports that dimension; otherwise set it false and explain the missing evidence in `text`.
- For scalability/cost-effectiveness, qualitative review evidence about route efficiency, shorter routes, fewer protecting groups, practical synthesis, or lower process burden counts as evidence. Do not require quantitative scale-up metrics or reagent prices.
- Do not put review/comparison sources in `direct_routes`.
- Do not use absence claims in `central_tradeoff` for dimensions the question asks to compare; use a qualitative trade-off when evidence exists.

Current JSON:
{current_json}

Original extraction prompt and retrieved evidence:
{original_prompt}
""".strip()


def _comparison_schema_instruction(query: str) -> str:
    if not getattr(cfg, "STAGE3_COMPARISON_SCHEMA_ENABLED", False):
        return ""
    if not _is_comparison_query(query):
        return ""
    if getattr(cfg, "COMPARISON_JSON_ENABLED", False):
        return """
COMPARISON_JSON MODE:
Ignore the normal numbered-fact output format. Return exactly one valid JSON object and no Markdown fences.
Use this schema:
{
  "comparison_json": {
    "target_compound": "",
    "source_roles": [
      {"source": "", "role": "route | review/comparison source | background", "claim": "", "evidence": ""}
    ],
    "direct_routes": [
      {"source": "", "route_phrase": "", "produces_target": true, "evidence": ""}
    ],
    "review_comparison_sources": [
      {"source": "", "claim": "", "dimensions": [], "evidence": ""}
    ],
    "dimensions": {
      "isotopic_enrichment": {"requested": false, "evidence_found": false, "text": "", "sources": []},
      "scalability": {"requested": false, "evidence_found": false, "text": "", "sources": []},
      "cost_effectiveness": {"requested": false, "evidence_found": false, "text": "", "sources": []},
      "safety": {"requested": false, "evidence_found": false, "text": "", "sources": []}
    },
    "central_tradeoff": ""
  }
}
Rules:
- Source metadata and guidance lines may only decide role; they are not paper evidence.
- A review/comparison source stays role="review/comparison source"; do not rewrite it as an experimental route paper.
- direct_routes must include only non-review route papers that directly synthesize the target compound.
- If a review/comparison source describes example routes, summarize them in review_comparison_sources, not in direct_routes.
- Derivative/formulation/solubility/biological-property papers are background unless they directly synthesize the exact target compound.
- Preserve exact route-defining phrases from the source. For the L-BPA hybrid route, use "enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis".
- Fill every dimension that appears in the question or source evidence.
- Set requested=true for every dimension named in the question.
- Set evidence_found=true when retrieved paper evidence supports that dimension, even if the evidence is qualitative rather than numeric.
- For scalability/cost-effectiveness, qualitative review evidence about route efficiency, shorter routes, fewer protecting groups, practical synthesis, or lower process burden counts as evidence. Do not require quantitative scale-up metrics or reagent prices.
- If requested=true but evidence_found=false, explain the missing retrieved evidence in text; do not silently omit the dimension.
- If scalability, cost-effectiveness, and safety appear together in the source, keep all three.
- If isotopic enrichment or 10B appears, set dimensions.isotopic_enrichment.evidence_found=true.
- Do not write absence claims such as "does not explicitly provide scalability/cost-effectiveness" when a review/comparison source is selected for those dimensions.
- central_tradeoff must mention every evidence_found=true dimension and stay qualitative unless the source explicitly provides values.
""".strip()
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
- safety: ...
- isotopic enrichment: ...

額外規則：
- Source metadata 與 guidance lines 只用來判斷來源角色；它們 are not paper evidence，不可當作可引用論文事實。
- review/comparison source 若描述某路線，只能寫「該綜述/比較來源報導或比較某路線」，不要寫成「該論文提供/實作該合成路線」。
- 保留問題或原文指名的比較面向；若原文同一句或相鄰句同時提到 scalability、cost-effectiveness、safety，必須一起保留在 [dimension_evidence]。
- 與面向無關的實驗條件、產率、試劑細節可省略。
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

    def _generate(self, prompt: str, system_prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model":  self.model,
                "system": system_prompt,
                "prompt": prompt,
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

        chunks_out = []
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                chunk = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            token = chunk.get("response", "")
            if token:
                print(token, end="", flush=True)
                chunks_out.append(token)
            if chunk.get("done"):
                break
        print()
        return "".join(chunks_out).strip()

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
            result = self._generate(user_prompt, _system_prompt())
            comparison_json_mode = getattr(cfg, "COMPARISON_JSON_ENABLED", False) and _is_comparison_query(query)
            if comparison_json_mode:
                result = _normalize_comparison_json(result, query)
                errors = _comparison_json_validation_errors(result, query)
                retries = getattr(cfg, "COMPARISON_JSON_REPAIR_RETRIES", 1)
                if getattr(cfg, "COMPARISON_JSON_VALIDATION_ENABLED", False):
                    for attempt in range(retries):
                        if not errors:
                            break
                        _status(
                            "  🔧 [comparison-json] validator failed; "
                            f"repair {attempt + 1}/{retries}: {'; '.join(errors[:3])}"
                        )
                        repair_prompt = _comparison_json_repair_prompt(user_prompt, result, errors)
                        candidate = self._generate(repair_prompt, _system_prompt())
                        candidate = _normalize_comparison_json(candidate, query)
                        if not isinstance(_comparison_json_payload(candidate), dict):
                            _status("  ⚠️  [comparison-json] repair output invalid; keeping previous JSON")
                            break
                        candidate_errors = _comparison_json_validation_errors(candidate, query)
                        if len(candidate_errors) <= len(errors):
                            result, errors = candidate, candidate_errors
                        else:
                            _status("  ⚠️  [comparison-json] repair introduced more errors; keeping previous JSON")
                            break
                if errors:
                    _status(f"  ⚠️  [comparison-json] validation still failed: {'; '.join(errors[:3])}")
                else:
                    _status("  ✅ [comparison-json] validation passed")

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
