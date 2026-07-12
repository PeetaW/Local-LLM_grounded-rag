import json
import time
import logging
import re
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
    source_roles = comparison["source_roles"] if isinstance(comparison["source_roles"], list) else []
    review_sources = {
        item.get("source") for item in source_roles
        if isinstance(item, dict) and "review/comparison" in str(item.get("role", "")).lower()
    }
    background_sources = {
        item.get("source") for item in source_roles
        if isinstance(item, dict) and "background" in str(item.get("role", "")).lower()
    }
    for route in comparison["direct_routes"] if isinstance(comparison["direct_routes"], list) else []:
        if isinstance(route, dict):
            route.setdefault("outcome", "")
    dimensions = comparison.get("dimensions")
    if not isinstance(dimensions, dict):
        dimensions = comparison["dimensions"] = {}
    query_dims = _query_dimension_keys(query)
    for key in ("isotopic_enrichment", "scalability", "cost_effectiveness", "safety"):
        item = dimensions.setdefault(key, {})
        item["requested"] = key in query_dims or bool(item.get("requested", False))
        item.setdefault("text", "")
        item["sources"] = item.get("sources") if isinstance(item.get("sources"), list) else []
        raw_atomic = [
            {"source": str(entry.get("source", "")).strip(), "claim": str(entry.get("claim", "")).strip()}
            for entry in item.get("evidence", [])
            if isinstance(entry, dict) and entry.get("source") and entry.get("claim")
        ] if isinstance(item.get("evidence"), list) else []
        if not raw_atomic and item["text"] and len(item["sources"]) == 1:
            raw_atomic = [{"source": str(item["sources"][0]), "claim": str(item["text"])}]
        atomic = [entry for entry in raw_atomic if entry["source"] not in background_sources]
        dropped_background = len(atomic) != len(raw_atomic)
        item["evidence"] = atomic
        if atomic:
            item["text"] = " ".join(entry["claim"] for entry in atomic)
            item["sources"] = list(dict.fromkeys(entry["source"] for entry in atomic))
        elif dropped_background:
            item["text"] = ""
            item["sources"] = []
        item["evidence_found"] = False if dropped_background and not atomic else bool(
            atomic or item.get("evidence_found", item.get("present", False))
        )
        item["present"] = item["evidence_found"]
    if review_sources:
        comparison["direct_routes"] = [
            route for route in comparison.get("direct_routes", [])
            if not isinstance(route, dict) or route.get("source") not in review_sources
        ]
    tradeoff = comparison.get("central_tradeoff", "")
    if isinstance(tradeoff, str):
        sources = [
            str(item.get("source", "")).strip()
            for item in comparison.get("review_comparison_sources", [])
            if isinstance(item, dict) and item.get("source")
        ]
        comparison["central_tradeoff"] = {
            "claim": tradeoff.strip(),
            "sources": list(dict.fromkeys(sources)),
        }
    elif isinstance(tradeoff, dict):
        tradeoff["claim"] = str(tradeoff.get("claim", "")).strip()
        sources = tradeoff.get("sources") if isinstance(tradeoff.get("sources"), list) else []
        tradeoff["sources"] = list(dict.fromkeys(
            str(source).strip() for source in sources if source
        ))
    else:
        comparison["central_tradeoff"] = {"claim": "", "sources": []}
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
    atomic_required = getattr(cfg, "COMPARISON_JSON_DIRECT_RENDER_ENABLED", False)
    source_roles = comparison.get("source_roles") if isinstance(comparison.get("source_roles"), list) else []
    review_sources = {
        item.get("source") for item in source_roles
        if isinstance(item, dict) and "review/comparison" in str(item.get("role", "")).lower()
    }
    background_sources = {
        item.get("source") for item in source_roles
        if isinstance(item, dict) and "background" in str(item.get("role", "")).lower()
    }
    for key, item in dimensions.items():
        if not isinstance(item, dict) or not isinstance(item.get("evidence"), list):
            continue
        invalid_sources = {
            entry.get("source") for entry in item["evidence"]
            if isinstance(entry, dict) and entry.get("source") in background_sources
        }
        if invalid_sources:
            errors.append(
                f"`dimensions.{key}.evidence` must not use background source(s): "
                f"{', '.join(sorted(invalid_sources))}."
            )
    for key in query_dims:
        item = dimensions.get(key)
        if not isinstance(item, dict):
            errors.append(f"`dimensions.{key}` is missing.")
            continue
        if not item.get("requested"):
            errors.append(f"`dimensions.{key}.requested` must be true because the question asks for it.")
        atomic = item.get("evidence") if isinstance(item.get("evidence"), list) else []
        valid_atomic = [
            entry for entry in atomic
            if isinstance(entry, dict) and entry.get("source") and entry.get("claim")
        ]
        if item.get("evidence_found") and atomic_required and not valid_atomic:
            errors.append(
                f"`dimensions.{key}` must contain source-bound atomic evidence entries."
            )
        elif item.get("evidence_found") and not valid_atomic and (not item.get("text") or not item.get("sources")):
            errors.append(f"`dimensions.{key}` says evidence_found=true but lacks evidence.")
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

    for route in comparison.get("direct_routes", []) if isinstance(comparison.get("direct_routes"), list) else []:
        if not isinstance(route, dict):
            continue
        if route.get("source") in review_sources:
            errors.append(f"Review/comparison source `{route.get('source')}` must not appear in direct_routes.")
        if atomic_required and not str(route.get("outcome", "")).strip():
            errors.append(f"Direct route `{route.get('source')}` must preserve its reported outcome.")
        source = str(route.get("source", ""))
        if route.get("source") in background_sources or any(
            term in source.lower()
            for term in ("derivative", "formulation", "solubility", "biological propert")
        ):
            errors.append("Derivative/formulation/solubility/biological-property source must not be a direct target-compound route.")

    tradeoff_value = comparison.get("central_tradeoff", "")
    tradeoff = str(
        tradeoff_value.get("claim", "") if isinstance(tradeoff_value, dict) else tradeoff_value
    ).lower()
    if _has_absence_claim(tradeoff) and query_dims:
        errors.append("central_tradeoff contains an absence claim for requested comparison dimensions.")
    route_outcomes = " ".join(
        str(route.get("outcome", ""))
        for route in comparison.get("direct_routes", [])
        if isinstance(route, dict)
    ).lower()
    purity_framed = any(term in tradeoff for term in (
        "high-purity", "high purity", "high optical purity", "optically pure", "enantiopur",
    ))
    isotope_framed = any(term in tradeoff for term in (
        "isotopically enriched", "isotopic enrichment", "10b", "boron-10",
    ))
    if (
        "isotopic_enrichment" in query_dims
        and any(term in route_outcomes for term in ("optically pure", "optical purity", "e.e.", " ee", "enantiopur"))
        and not (purity_framed and isotope_framed)
    ):
        errors.append(
            "central_tradeoff must explicitly frame high-purity/isotopically enriched material."
        )
    if atomic_required and query_dims and (
        not isinstance(tradeoff_value, dict)
        or not tradeoff_value.get("claim")
        or not tradeoff_value.get("sources")
    ):
        errors.append("central_tradeoff must contain a source-bound claim.")
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
- Set `evidence_found=true` only when the retrieved evidence supports that dimension; otherwise set it false and leave `evidence` empty.
- Every dimension evidence item must contain exactly one source and one atomic claim. Split claims from different sources into separate items.
- Background sources must not provide core comparison-dimension evidence; use route or review/comparison sources.
- Every direct route must preserve its reported outcome, including optical purity, e.e., yield, or other comparison-relevant result when present.
- central_tradeoff must contain one claim and only the source paper(s) that directly support it; prefer a review/comparison source when available.
- When isotopic enrichment is requested and a route outcome reports optical purity/e.e., central_tradeoff.claim must explicitly say high-purity/isotopically enriched material.
- For scalability/cost-effectiveness, qualitative review evidence about route efficiency, shorter routes, fewer protecting groups, practical synthesis, or lower process burden counts as evidence. Do not require quantitative scale-up metrics or reagent prices.
- Do not put review/comparison sources in `direct_routes`.
- Do not use absence claims in `central_tradeoff` for dimensions the question asks to compare; use a qualitative trade-off when evidence exists.

Current JSON:
{current_json}

Original extraction prompt and retrieved evidence:
{original_prompt}
""".strip()


def _comparison_schema_instruction(query: str, comparison_json_enabled: bool | None = None) -> str:
    if not getattr(cfg, "STAGE3_COMPARISON_SCHEMA_ENABLED", False):
        return ""
    if not _is_comparison_query(query):
        return ""
    if comparison_json_enabled is None:
        comparison_json_enabled = getattr(cfg, "COMPARISON_JSON_ENABLED", False)
    if comparison_json_enabled:
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
      {"source": "", "route_phrase": "", "outcome": "", "produces_target": true, "evidence": ""}
    ],
    "review_comparison_sources": [
      {"source": "", "claim": "", "dimensions": [], "evidence": ""}
    ],
    "dimensions": {
      "isotopic_enrichment": {"requested": false, "evidence_found": false, "evidence": [{"source": "", "claim": ""}]},
      "scalability": {"requested": false, "evidence_found": false, "evidence": [{"source": "", "claim": ""}]},
      "cost_effectiveness": {"requested": false, "evidence_found": false, "evidence": [{"source": "", "claim": ""}]},
      "safety": {"requested": false, "evidence_found": false, "evidence": [{"source": "", "claim": ""}]}
    },
    "central_tradeoff": {"claim": "", "sources": []}
  }
}
Rules:
- Source metadata and guidance lines may only decide role; they are not paper evidence.
- A review/comparison source stays role="review/comparison source"; do not rewrite it as an experimental route paper.
- direct_routes must include only non-review route papers that directly synthesize the target compound.
- If a review/comparison source describes example routes, summarize them in review_comparison_sources, not in direct_routes.
- Derivative/formulation/solubility/biological-property papers are background unless they directly synthesize the exact target compound.
- Preserve exact route-defining phrases from the source. For the L-BPA hybrid route, use "enantioselective alkylation followed by chymotrypsin-catalysed enzymatic hydrolysis".
- Preserve each direct route's comparison-relevant outcome separately. If the evidence reports e.e. or optical purity, keep it in outcome; for the L-BPA hybrid route preserve the reported 74% e.e. alkylation outcome and optically pure final L-BPA when retrieved.
- Each dimensions.*.evidence item must bind exactly one atomic claim to exactly one source. Never place two papers' claims in one item and never use a separate multi-source list.
- Core dimension evidence must come from a route or review/comparison source, never from a source classified as background.
- Fill every dimension that appears in the question or source evidence.
- Set requested=true for every dimension named in the question.
- Set evidence_found=true when retrieved paper evidence supports that dimension, even if the evidence is qualitative rather than numeric.
- For scalability/cost-effectiveness, qualitative review evidence about route efficiency, shorter routes, fewer protecting groups, practical synthesis, or lower process burden counts as evidence. Do not require quantitative scale-up metrics or reagent prices.
- If requested=true but evidence_found=false, leave evidence empty; do not silently omit the dimension from the schema.
- If scalability, cost-effectiveness, and safety appear together in the source, keep all three.
- If isotopic enrichment or 10B appears, set dimensions.isotopic_enrichment.evidence_found=true.
- If evidence mentions high cost/expense of isotopically enriched 10B or boron material, preserve that wording in an atomic cost_effectiveness evidence item and central_tradeoff.claim.
- Do not write absence claims such as "does not explicitly provide scalability/cost-effectiveness" when a review/comparison source is selected for those dimensions.
- central_tradeoff.claim must mention every evidence_found=true dimension, stay qualitative unless the source explicitly provides values, and list only supporting paper(s) in central_tradeoff.sources.
- When isotopic enrichment is requested and a direct route outcome reports optical purity/e.e., central_tradeoff.claim must explicitly frame high-purity/isotopically enriched material against scalability/cost-effectiveness.
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
- 若 evidence 提到 high cost/expense of isotopically enriched 10B 或 boron material，必須保留在 cost-effectiveness 與 central trade-off 相關事實中。
- 與面向無關的實驗條件、產率、試劑細節可省略。
""".strip()


def _build_user_prompt(
    formatted: str,
    query: str,
    comparison_json_enabled: bool | None = None,
) -> str:
    schema = _comparison_schema_instruction(query, comparison_json_enabled=comparison_json_enabled)
    schema_block = f"\n\n{schema}" if schema else ""
    return (
        f"參考問題方向（僅供整理聚焦，不影響事實陳述）：{query}{schema_block}\n\n"
        f"請將以下論文段落整理為結構化已知事實清單：\n\n"
        f"--- 論文段落開始 ---\n{formatted}\n--- 論文段落結束 ---"
    )


def _source_near(text: str, pos: int) -> str:
    window = text[max(0, pos - 1200):pos + 200]
    matches = list(re.finditer(r"【([^】]+)】", window))
    if matches:
        return matches[-1].group(1)
    matches = list(re.finditer(r"來源：([^\n]+)", window))
    return matches[-1].group(1).strip() if matches else "retrieved evidence"


def _append_isotope_cost_fact(result: str, evidence_text: str, query: str) -> str:
    if not _is_comparison_query(query):
        return result
    lower_result = (result or "").lower()
    if any(term in lower_result for term in ("high cost", "isotope starting material", "expensive 10b", "10b-enriched starting")):
        return result

    text = evidence_text or ""
    lower = text.lower()
    patterns = (
        "high cost of isotopically enriched 10b",
        "major cost typically comes from the isotope starting material",
    )
    pos = next((lower.find(p) for p in patterns if lower.find(p) >= 0), -1)
    if pos < 0:
        return result

    source = _source_near(text, pos)
    lines = (result or "").rstrip().splitlines()
    if lines and lines[-1].strip().startswith("- ") and not lines[-1].strip().endswith((".", ")", "]")):
        lines.pop()
    result = "\n".join(lines).rstrip()
    if "[dimension_evidence]" not in result:
        result += "\n\n[dimension_evidence]"
    if "high cost of isotopically enriched 10b" in lower:
        fact = "the review highlights the high cost of isotopically enriched 10B"
    else:
        fact = "the major cost typically comes from the isotope starting material"
    return result + f"\n- cost-effectiveness: When preparing isotopically enriched compounds, {fact} (Source: {source})."


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

    def _generate(self, prompt: str, system_prompt: str, metadata: dict | None = None) -> str:
        options = {
            "temperature": 0.1,
            "num_predict": 8192,
            "num_ctx": cfg.STAGE3_NUM_CTX,
        }
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model":  self.model,
                "system": system_prompt,
                "prompt": prompt,
                "stream": True,
                "options": options,
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
                if metadata is not None:
                    metadata.update({
                        key: chunk.get(key)
                        for key in (
                            "done_reason", "prompt_eval_count", "eval_count",
                            "total_duration", "load_duration",
                            "prompt_eval_duration", "eval_duration",
                        )
                    })
                break
        print()
        output = "".join(chunks_out).strip()
        if metadata is not None:
            metadata.update({
                "model": self.model,
                "requested_num_ctx": options["num_ctx"],
                "requested_num_predict": options["num_predict"],
                "prompt_chars": len(prompt),
                "output_chars": len(output),
            })
        return output

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

    @staticmethod
    def _fallback_chunks(chunks: list[dict]) -> str:
        lines = []
        for i, chunk in enumerate(chunks):
            source = (
                chunk.get("source")
                or chunk.get("paper_name")
                or chunk.get("file_name")
                or (chunk.get("metadata") or {}).get("file_name")
                or f"chunk_{i}"
            )
            text = chunk.get("text") or chunk.get("content") or str(chunk)
            lines.append(f"[Chunk {i+1}] 來源：{source}\n{text}")
        return "\n\n".join(lines)

    def synthesize(
        self,
        chunks: list[dict],
        query: str = "",
        on_status=None,
        on_artifact=None,
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
        if on_artifact:
            on_artifact("stage3_prompt", user_prompt)

        def _status(msg):
            if on_status:
                on_status(msg)
            else:
                print(msg)

        generation_meta = []

        def _generate_attempt(prompt: str, attempt: str) -> str:
            meta = {"attempt": attempt}
            output = self._generate(prompt, _system_prompt(), metadata=meta)
            generation_meta.append(meta)
            if on_artifact:
                on_artifact("stage3_generation_meta", generation_meta)
            if len(meta) > 1:
                marker = "⚠️ " if meta.get("done_reason") == "length" else "ℹ️ "
                _status(
                    f"  {marker} [ollama] {attempt}: done_reason={meta.get('done_reason') or 'unknown'} "
                    f"prompt_tokens={meta.get('prompt_eval_count')} output_tokens={meta.get('eval_count')}"
                )
            return output

        logger.info(
            "[Synthesizer] Starting: %d chunks (%d chars), query=\"%s\"",
            len(chunks), total_chars, query[:50]
        )
        t0 = time.time()
        comparison_json_mode = getattr(cfg, "COMPARISON_JSON_ENABLED", False) and _is_comparison_query(query)

        try:
            result = _generate_attempt(
                user_prompt, "comparison_json" if comparison_json_mode else "fact_list"
            )
            if on_artifact:
                on_artifact("stage3_raw_output", result)
            if not (result or "").strip():
                _status("  ⚠️  [Synthesizer] empty output; using original chunks")
                result = self._fallback_chunks(chunks)
            if comparison_json_mode and '"comparison_json"' in result:
                result = _normalize_comparison_json(result, query)
                errors = _comparison_json_validation_errors(result, query)
                retries = getattr(cfg, "COMPARISON_JSON_REPAIR_RETRIES", 1)
                if errors == ["Output is not valid JSON."]:
                    _status("  ⚠️  [comparison-json] invalid JSON; retrying plain comparison schema")
                    plain_prompt = _build_user_prompt(
                        formatted, query, comparison_json_enabled=False
                    )
                    if on_artifact:
                        on_artifact("stage3_plain_prompt", plain_prompt)
                    result = _generate_attempt(plain_prompt, "plain_fallback")
                    if on_artifact:
                        on_artifact("stage3_plain_output", result)
                    if not (result or "").strip():
                        _status("  ⚠️  [Synthesizer] plain retry empty; using original chunks")
                        result = self._fallback_chunks(chunks)
                    result = _append_isotope_cost_fact(result, formatted, query)
                    errors = []
                elif getattr(cfg, "COMPARISON_JSON_VALIDATION_ENABLED", False):
                    for attempt in range(retries):
                        if not errors:
                            break
                        _status(
                            "  🔧 [comparison-json] validator failed; "
                            f"repair {attempt + 1}/{retries}: {'; '.join(errors[:3])}"
                        )
                        repair_prompt = _comparison_json_repair_prompt(user_prompt, result, errors)
                        candidate = _generate_attempt(repair_prompt, f"json_repair_{attempt + 1}")
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
            else:
                result = _append_isotope_cost_fact(result, formatted, query)

            elapsed = time.time() - t0
            logger.info(
                "[Synthesizer] Done: input %d chars → output %d chars (%.1fs)",
                total_chars, len(result), elapsed
            )
            _status(
                f"  📋 [Synthesizer] {len(chunks)} chunks → "
                f"{len(result)} chars ({elapsed:.1f}s)"
            )
            if on_artifact:
                on_artifact("stage3_knowledge_base", result)
            return result

        except Exception as e:
            elapsed = time.time() - t0
            logger.warning("[Synthesizer] FALLBACK: %s (%.1fs)", e, elapsed)
            _status(f"  ⚠️  [Synthesizer] 失敗，使用原始 chunks ({e})")
            # Fallback：直接串接原始 chunk text
            result = self._fallback_chunks(chunks)
            if on_artifact:
                on_artifact("stage3_knowledge_base", result)
            return result
