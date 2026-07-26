import json
import re
import unicodedata


_FACT_RE = re.compile(
    r"^\s*\[(?:Fact|事實)\s*\d+\]\s*(?P<claim>.*?)\s*"
    r"\((?:Source|來源)\s*[:：]\s*(?P<source>.*?)\)\s*\.?\s*$",
    re.IGNORECASE,
)
_SNIPPET_RE = re.compile(r"\[Snippet \d+\]\s*(.*?)(?=\s*\[Snippet \d+\]|\Z)", re.DOTALL)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for",
    "from", "in", "is", "it", "its", "of", "on", "or", "that", "the", "these",
    "this", "those", "to", "was", "were", "when", "which", "while", "with",
}
_NEGATIONS = {"absence", "neither", "never", "no", "nor", "not", "without"}
_CONTINUATION_RE = re.compile(
    r"(?P<head>.*\b(?:generating|yielding|reaching|resulting\s+in))\b",
    re.IGNORECASE,
)
_FRAGMENT_RE = re.compile(
    r"^(?:approximately|about|roughly|nearly|up\s+to|at\s+least|at\s+most|\d)",
    re.IGNORECASE,
)
_PDF_INTERRUPTION_RE = re.compile(
    r"\b(?:fig(?:ure)?\.?|table|journal\s+of|chromatographic\s+trace|mechanistic\s+pathway)\b",
    re.IGNORECASE,
)
_BROKEN_FRAGMENT_RE = re.compile(
    r"(?<!\bvs)[.!?]\s+(?:approximately|about|roughly|nearly|up\s+to|at\s+least|at\s+most|\d)",
    re.IGNORECASE,
)
_INLINE_REFERENCE_RE = re.compile(
    r"\b(?:fig(?:ure)?s?|eq(?:uation)?s?|table)\.?\s+(?=\d)",
    re.IGNORECASE,
)
_FOCUS_STOPWORDS = _STOPWORDS | {
    "according", "answer", "describe", "detail", "explain", "finding", "findings",
    "give", "given", "how", "include", "including", "key", "main", "paper", "papers",
    "question", "report", "reported", "specific", "study", "used", "using", "what",
}
_METHOD_QUERY_RE = re.compile(
    r"\b(?:method|process|procedure|protocol|synthesi\w*|preparation|reaction route|"
    r"key steps?|reactants?|reagents?)\b",
    re.IGNORECASE,
)
_CONDITION_QUERY_RE = re.compile(
    r"\b(?:conditions?|solvents?|catalysts?|loading|temperature|reaction time|"
    r"duration|concentration|pressure|pH)\b",
    re.IGNORECASE,
)
_RELATION_QUERY_RE = re.compile(
    r"\b(?:mechanism|role|relationship|interact\w*|bind\w*|inhibit\w*|"
    r"influence|why|how does|how do)\b",
    re.IGNORECASE,
)
_OUTCOME_QUERY_RE = re.compile(
    r"\b(?:data|effect|efficacy|outcome|result|survival|tumou?r|accumulation|"
    r"retention|potency|ic50|yield|selectivity|purity|impurit\w*|storage|stability)\b",
    re.IGNORECASE,
)
_METHOD_ACTION_RE = re.compile(
    r"\b(?:react(?:ed|ion)|treat(?:ed|ment)|convert(?:ed|sion)|hydroly\w*|"
    r"alkylat\w*|coupl\w*|deprotect\w*|esterif\w*|prepar\w*|synthesi\w*|"
    r"cycli[sz]\w*|oxid\w*|reduc\w*)\b",
    re.IGNORECASE,
)
_STEP_DEFINING_ACTION_RE = re.compile(
    r"\b(?:reacted|treat(?:ed|ment)|convert(?:ed|sion)|hydroly\w*|"
    r"alkylat\w*|coupl\w*|deprotect\w*|esterif\w*|cycli[sz]\w*|"
    r"oxid\w*|reduc\w*)\b",
    re.IGNORECASE,
)
_CHEMICAL_TRANSFORM_ACTION_RE = re.compile(
    r"\b(?:reacted|alkylat\w*|coupl\w*|oxid\w*|reduc\w*)\b",
    re.IGNORECASE,
)
_CONDITION_EVIDENCE_RE = re.compile(
    r"\b(?:solvent|catalyst|loading|equiv(?:alent)?s?|mmol|mol|hours?|hrs?|"
    r"minutes?|mins?|temperature|room temperature|stirred|heated|cooled|"
    r"thf|methanol|ethanol|dichloromethane|water)\b|°\s*c",
    re.IGNORECASE,
)
_OUTCOME_EVIDENCE_RE = re.compile(
    r"\b(?:yield|e\.?\s*e\.?|enantiomeric excess|purity|selectivity|"
    r"afford(?:ed)?|furnish(?:ed)?|obtained|produced|survival|tumou?r|"
    r"accumulation|retention|inhibition|potency|ic50|vmax|km|degrad\w*|"
    r"impurit\w*|stability|stable|formation|formed|detected|concentration)\b",
    re.IGNORECASE,
)
_CONTROL_EVIDENCE_RE = re.compile(
    r"\b(?:control|comparison|compared|versus|vs\.?|without|absence|untreated)\b",
    re.IGNORECASE,
)
_RELATION_EVIDENCE_RE = re.compile(
    r"\b(?:because|due to|through|via|thereby|leads? to|results? in|"
    r"interact\w*|bind\w*|bond\w*|exchange|inhibit\w*|uptake|efflux|"
    r"cross-?link\w*|collapse\w*|de-?crosslink\w*|reconstruct\w*|reform\w*)\b",
    re.IGNORECASE,
)
_BINDING_EVIDENCE_RE = re.compile(
    r"\b(?:bind\w*|bond\w*|affinity|recognition|receptor|occup\w*|interact\w*)\b",
    re.IGNORECASE,
)
_DYNAMIC_EXCHANGE_RE = re.compile(
    r"\bexchange\w*\b",
    re.IGNORECASE,
)
_NETWORK_FORMATION_RE = re.compile(
    r"\b(?:hydrogel|network)\b.{0,100}\b(?:cross-?link\w*|formation|form\w*)\b"
    r"|\b(?:cross-?link\w*|formation|form\w*)\b.{0,100}\b(?:hydrogel|network)\b",
    re.IGNORECASE,
)
_NETWORK_DISRUPTION_RE = re.compile(
    r"\b(?:collapse\w*|de-?crosslink\w*)\b",
    re.IGNORECASE,
)
_NETWORK_RECOVERY_RE = re.compile(
    r"\b(?:reconstruct\w*|reform\w*|recover\w*|regenerat\w*)\b",
    re.IGNORECASE,
)
_STRUCTURE_IDENTITY_RE = re.compile(
    r"\b(?:dehydrat\w*|complex\w*|transform\w*)\b",
    re.IGNORECASE,
)
_STABILITY_VALUE_RE = re.compile(
    r"\b(?:pH|days?|weeks?|months?)\b",
    re.IGNORECASE,
)
_CHEMICAL_STEP_RE = re.compile(
    r"\b(?:react(?:ed|ion)|alkylat\w*|coupl\w*|oxid\w*|reduc\w*)\b",
    re.IGNORECASE,
)
_ENZYMATIC_STEP_RE = re.compile(
    r"\b(?:enzyme|enzymatic|hydroly\w*|chymotrypsin|trypsin|aminoacylase)\b",
    re.IGNORECASE,
)


def build_fact_contract_requirements(
    query: str,
    focus_questions: list[str] | None = None,
) -> list[dict]:
    texts = list(dict.fromkeys(
        str(value).strip()
        for value in [query, *(focus_questions or [])]
        if str(value).strip()
    ))
    combined = " ".join(texts)
    requirements = []

    def add(kind: str, label: str, minimum: int = 1):
        key = (kind, _plain(label))
        if any(item["_key"] == key for item in requirements):
            return
        requirements.append({
            "_key": key,
            "kind": kind,
            "label": label,
            "minimum": minimum,
        })

    method_query = bool(_METHOD_QUERY_RE.search(combined))
    relation_query = bool(_RELATION_QUERY_RE.search(combined))
    outcome_query = bool(_OUTCOME_QUERY_RE.search(combined))
    if method_query:
        add("method_transform", "Exact reactants, reagents, and step-defining transformations", 2)
        hybrid_query = bool(re.search(
            r"\bhybrid\s+(?:method|process|route|synthesi\w*)\b",
            combined,
            re.IGNORECASE,
        ))
        if hybrid_query:
            add("chemical_step", "Chemical transformation in the hybrid process", 1)
            add("enzymatic_step", "Enzymatic transformation in the hybrid process", 1)
        explicit_conditions = bool(_CONDITION_QUERY_RE.search(query or ""))
        if explicit_conditions or re.search(r"\bkey steps?\b", query or "", re.IGNORECASE):
            add(
                "method_conditions",
                (
                    "Condition for the chemical transformation in the hybrid process"
                    if hybrid_query
                    else "Step-specific solvent, catalyst, temperature, time, or loading"
                ),
                2 if explicit_conditions else 1,
            )
        add("method_outcomes", "Step-specific yield, selectivity, purity, or product outcome", 2)
        if _CONTROL_EVIDENCE_RE.search(query or ""):
            add("control", "Control or comparison outcome", 1)
    if relation_query:
        label = next((text for text in texts if _RELATION_QUERY_RE.search(text)), query)
        add("relation", label or "Requested mechanism or relation", 2)
        if re.search(r"\bbind\w*\b", combined, re.IGNORECASE):
            add("binding_relation", label or "Requested binding relation", 1)
        dynamic_network = bool(
            re.search(r"\b(?:dynamic|covalent|exchange\w*)\b", combined, re.IGNORECASE)
            and re.search(
                r"\b(?:hydrogel|network|cross-?link\w*|gel[–-]sol)\b",
                combined,
                re.IGNORECASE,
            )
        )
        if dynamic_network:
            add("dynamic_exchange", label or "Requested dynamic exchange", 1)
            add("network_formation", label or "Requested network formation", 1)
            add("network_disruption", label or "Requested network disruption", 1)
            add("network_recovery", label or "Requested network recovery", 1)
        if re.search(
            r"\bwater-?stable\b",
            combined,
            re.IGNORECASE,
        ):
            add("structure_identity", label or "Requested stable structure", 2)
            add("stability_values", label or "Requested stability evidence", 2)
        elif re.search(
            r"\b(?:hydrogel|network|cross-?link\w*|gel[–-]sol)\b",
            combined,
            re.IGNORECASE,
        ):
            add("network_formation", label or "Requested network formation", 1)
    if outcome_query and not method_query:
        label = next((text for text in texts if _OUTCOME_QUERY_RE.search(text)), query)
        add("quantitative_outcome", label or "Requested quantitative outcome", 2)

    for focus in texts[1:]:
        if (
            (method_query and _METHOD_QUERY_RE.search(focus))
            or (relation_query and _RELATION_QUERY_RE.search(focus))
            or (outcome_query and _OUTCOME_QUERY_RE.search(focus))
        ):
            continue
        add("facet", focus, 1)
    if not requirements:
        add("facet", query or (texts[0] if texts else "Requested answer"), 2)

    return [
        {
            "id": f"R{index}",
            "kind": item["kind"],
            "label": item["label"],
            "minimum": item["minimum"],
        }
        for index, item in enumerate(requirements[:8], 1)
    ]


def _requirement_score(requirement: dict, evidence: str) -> int:
    kind = str(requirement.get("kind", "facet"))
    focus_tokens = _tokens(str(requirement.get("label", ""))) - _FOCUS_STOPWORDS
    evidence_tokens = _tokens(evidence)
    overlap = len(focus_tokens & evidence_tokens)
    numbers = bool(_numbers(evidence))

    if kind == "method_transform":
        actions = len(_METHOD_ACTION_RE.findall(evidence))
        if not actions:
            return 0
        generic = bool(re.search(
            r"\b(?:in summary|present synthetic method|advantage|few steps|"
            r"ease of performance|ease of workup)\b",
            evidence,
            re.IGNORECASE,
        ))
        return 20 + 4 * actions + 2 * numbers - (12 if generic else 0)
    if kind == "method_conditions":
        action_pattern = (
            _CHEMICAL_TRANSFORM_ACTION_RE
            if "chemical transformation" in _plain(str(requirement.get("label", "")))
            else _STEP_DEFINING_ACTION_RE
        )
        actions = len(action_pattern.findall(evidence))
        return (
            20 + 10 * actions + 3 * overlap + 2 * numbers
            if _CONDITION_EVIDENCE_RE.search(evidence)
            else 0
        )
    if kind == "method_outcomes":
        return 20 + 3 * overlap + 3 * numbers if _OUTCOME_EVIDENCE_RE.search(evidence) else 0
    if kind == "chemical_step":
        return 20 + 3 * overlap + 2 * numbers if _CHEMICAL_STEP_RE.search(evidence) else 0
    if kind == "enzymatic_step":
        return 20 + 3 * overlap + 2 * numbers if _ENZYMATIC_STEP_RE.search(evidence) else 0
    if kind == "control":
        return 20 + 3 * overlap + 2 * numbers if _CONTROL_EVIDENCE_RE.search(evidence) else 0
    if kind == "relation":
        return 20 + 4 * overlap + 2 * numbers if _RELATION_EVIDENCE_RE.search(evidence) else 0
    if kind == "binding_relation":
        anchors = re.findall(
            r"([a-z0-9+.-]+)\s+bind(?:s|ing)?\b",
            _plain(str(requirement.get("label", ""))),
        )
        if anchors and not set(anchors) & evidence_tokens:
            return 0
        markers = len(_BINDING_EVIDENCE_RE.findall(evidence))
        return 20 + 5 * markers + 3 * overlap if markers else 0
    if kind == "dynamic_exchange":
        markers = len(_DYNAMIC_EXCHANGE_RE.findall(evidence))
        return 20 + 5 * markers + 3 * overlap if markers else 0
    if kind == "network_formation":
        return 20 + 3 * overlap if _NETWORK_FORMATION_RE.search(evidence) else 0
    if kind == "network_disruption":
        return 20 + 3 * overlap if _NETWORK_DISRUPTION_RE.search(evidence) else 0
    if kind == "network_recovery":
        return 20 + 3 * overlap if _NETWORK_RECOVERY_RE.search(evidence) else 0
    if kind == "structure_identity":
        label = _plain(str(requirement.get("label", "")))
        if "water-stable" in label and not re.search(
            r"\b(?:water|aqueous|dehydrat\w*)\b",
            evidence,
            re.IGNORECASE,
        ):
            return 0
        markers = len(_STRUCTURE_IDENTITY_RE.findall(evidence))
        return 20 + 4 * markers + 3 * overlap if markers else 0
    if kind == "stability_values":
        pH_range = bool(re.search(
            r"\d+(?:\.\d+)?\s*<\s*pH\s*<\s*\d+(?:\.\d+)?",
            evidence,
            re.IGNORECASE,
        ))
        duration = bool(re.search(
            r"\b\d+(?:\.\d+)?[-\s]+(?:days?|weeks?|months?)\b",
            evidence,
            re.IGNORECASE,
        ))
        return (
            20 + 20 * pH_range + 10 * duration + 3 * overlap
            if numbers and _STABILITY_VALUE_RE.search(evidence)
            else 0
        )
    if kind == "quantitative_outcome":
        if not numbers or not _OUTCOME_EVIDENCE_RE.search(evidence):
            return 0
        return 20 + 4 * overlap

    minimum_overlap = 1 if len(focus_tokens) < 6 else 2
    return 5 * overlap + 2 * numbers if overlap >= minimum_overlap else 0


def _rank_requirement(requirement: dict, catalog: list[dict]) -> list[dict]:
    ranked = [
        (_requirement_score(requirement, item["text"]), -index, item)
        for index, item in enumerate(catalog)
    ]
    return [
        item for score, _, item in sorted(ranked, reverse=True)
        if score > 0
    ]


def _plain(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").lower().replace("μ", "µ")
    value = re.sub(r"\\text\{([^{}]*)\}", r"\1", value)
    value = value.replace(r"\pm", "±").replace("_", "")
    value = re.sub(r"\(?see\s+fig\.\s*\d+\.?\)?", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\[(?:source\s*:\s*)?[^\]]+\]", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"(?<=[a-z0-9])-\s+(?=[a-z0-9])", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _tokens(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[a-z0-9]+", _plain(text)):
        if token in _STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "is")):
            token = token[:-1]
        tokens.add(token)
    return tokens


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?!\w|\.\d)", _plain(text)))


def _sentence_blocks(text: str) -> list[str]:
    snippets = _SNIPPET_RE.findall(text or "")
    return snippets or [text or ""]


def _interrupted_sentence_windows(sentences: list[str]) -> tuple[list[str], set[str]]:
    """Restore clauses split by embedded PDF figures without joining normal sentences."""
    windows, superseded = [], set()
    for index, fragment in enumerate(sentences):
        if not _FRAGMENT_RE.match(fragment):
            continue
        fragment_tokens = _tokens(fragment)
        start = max(0, index - 8)
        for previous_index in range(index - 1, start - 1, -1):
            previous = sentences[previous_index]
            continuation = _CONTINUATION_RE.match(previous)
            if not continuation:
                continue
            interruptions = sentences[previous_index + 1:index]
            if any(
                len(_tokens(item)) > 3 and not _PDF_INTERRUPTION_RE.search(_plain(item))
                for item in interruptions
            ):
                continue
            head = continuation.group("head").rstrip(" ,")
            if not (_tokens(head) & fragment_tokens):
                continue
            windows.append(f"{head} {fragment}".strip())
            superseded.add(fragment)
            break
    return windows, superseded


def _catalog_noise(sentence: str) -> bool:
    plain = _plain(sentence)
    if (
        "corresponding author" in plain
        or "a r t i c l e i n f o" in plain
        or "sub-question:" in plain
        or "source metadata" in plain
        or "retrieved evidence snippets" in plain
        or "use only snippets below" in plain
        or re.search(
            r"\breceived\s*:?\s*(?:january|february|march|april|may|june|july|"
            r"august|september|october|november|december|\d)",
            plain,
        )
        or (
            re.search(r"\breceived\s*:", plain)
            and re.search(r"\baccepted\s*:", plain)
        )
        or ("department of" in plain and "university" in plain)
        or re.search(r"\b(?:supplementary\s+)?fig(?:ure)?\.?$", plain)
        or re.search(r"^\d+(?:\.\d+)?\s*(?:mmol|mg|ml|g)?\s*\)", plain)
        or re.search(r"(?:~{4,}|~[^A-Za-z0-9]{0,3}~|\[\(?x\]|\[lit\.,)", sentence, re.IGNORECASE)
        or (
            sentence.count("[") > sentence.count("]")
            and re.search(r"\[[^\]]{0,40}$", sentence)
        )
    ):
        return True
    plot_labels = re.search(r"\b(?:mau|time\s*\(min\))\b", plain, re.IGNORECASE)
    figure = re.search(r"\bfig\.", plain, re.IGNORECASE)
    return bool(plot_labels and figure)


def _trim_scheme_ocr(sentence: str) -> str:
    if not re.search(r"~{4,}", sentence):
        return sentence
    scheme = re.search(r"\s+\b(?:was|were)\s+\)\(", sentence, re.IGNORECASE)
    if not scheme:
        return sentence
    return sentence[:scheme.start()].rstrip(" ,;") + "."


def build_evidence_catalog(chunks: list[dict]) -> list[dict]:
    catalog, seen = [], set()
    for chunk in chunks:
        source = str(chunk.get("source", "")).strip()
        if not source:
            continue
        for block in _sentence_blocks(str(chunk.get("text", ""))):
            block = re.sub(r"\x03(?=g(?:/|\b))", "µ", block)
            clean = re.sub(r"\s+", " ", block).strip()
            sentences = re.split(
                r"(?<!\bFig\.)(?<!\bFigs\.)(?<!\bEq\.)(?<!\bDr\.)(?<!\bMr\.)(?<!\bvs\.)"
                r"(?<!\bg\.)(?<!\bmg\.)(?<!\bml\.)(?<!\bmmol\.)(?<!\bmin\.)"
                r"(?<!\bh\.)(?<=[.!?])\s+",
                clean,
                flags=re.IGNORECASE,
            )
            sentences = [sentence.strip(" -\n") for sentence in sentences]
            windows, superseded = _interrupted_sentence_windows(sentences)
            for sentence in sentences + windows:
                original_sentence = sentence
                sentence = _trim_scheme_ocr(sentence)
                if (
                    original_sentence in superseded
                    or (
                        sentence[:1].islower()
                        and not re.match(
                            r"(?:pH\b|mRNA\b|m/z\b|in vitr[oa]\b|tert-|cis-|trans-|[α-ω])",
                            sentence,
                        )
                    )
                    or (
                        sentence.count("(") > sentence.count(")")
                        and re.search(r"\([^)]{0,80}$", sentence)
                    )
                    or _catalog_noise(sentence)
                ):
                    continue
                sentence = re.sub(
                    r"^\(See\s+Fig\.\s*\d+\.?\)\s*", "", sentence, flags=re.IGNORECASE
                )
                normalized = _plain(sentence)
                if len(normalized) < 20 or normalized.startswith((
                    "source metadata", "sub-question", "use only snippets",
                )):
                    continue
                key = (source.casefold(), normalized)
                if key in seen:
                    continue
                seen.add(key)
                catalog.append({
                    "id": f"E{len(catalog) + 1}",
                    "source": source,
                    "text": sentence,
                })
    return catalog


def format_evidence_catalog(catalog: list[dict]) -> str:
    return "\n".join(
        f"{item['id']} | source={item['source']} | {item['text']}" for item in catalog
    )


def fact_contract_schema(
    catalog: list[dict],
    requirements: list[dict] | None = None,
) -> dict:
    evidence_ids = [item["id"] for item in catalog]
    properties = {
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string", "enum": evidence_ids},
            "minItems": 1,
            "maxItems": min(20, len(evidence_ids)),
        },
    }
    required = ["evidence_ids"]
    if requirements:
        properties["requirement_evidence"] = {
            "type": "object",
            "properties": {
                item["id"]: {
                    "type": "array",
                    "items": {"type": "string", "enum": evidence_ids},
                    "maxItems": min(
                        max(1, int(item.get("minimum", 1))),
                        len(evidence_ids),
                    ),
                }
                for item in requirements
            },
            "required": [item["id"] for item in requirements],
            "additionalProperties": False,
        }
        required.append("requirement_evidence")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def fact_contract_prompt(
    catalog: list[dict],
    query: str,
    recovery_hint: str = "",
    focus_questions: list[str] | None = None,
    requirements: list[dict] | None = None,
) -> str:
    recovery = f"\nRecovery focus: {recovery_hint}\n" if recovery_hint else ""
    focus = [
        str(value).strip()
        for value in (focus_questions or [])
        if str(value).strip() and str(value).strip() != (query or "").strip()
    ]
    focus_block = ""
    if focus:
        focus_block = "\nPlanned coverage facets:\n" + "\n".join(
            f"- {value}" for value in dict.fromkeys(focus)
        )
    requirements = requirements or build_fact_contract_requirements(query, focus)
    requirement_block = "\nAtomic coverage requirements:\n" + "\n".join(
        f"- {item['id']} ({item['minimum']} witness"
        f"{'es' if item['minimum'] != 1 else ''} when available): {item['label']}"
        for item in requirements
    )
    return f"""Select all and only the evidence IDs needed to answer the question.

Question: {query}
{recovery}{focus_block}{requirement_block}

Rules:
- Return evidence_ids as the union of all selected IDs, and group them by requirement_evidence.
- Select direct evidence, not background or inferred relationships.
- Include every sentence needed for requested values, conditions, outcomes, and comparisons.
- Treat each atomic requirement as an independent checklist. Evidence for one yield, condition, or relation does not cover another requirement.
- For a mechanism, role, key-step, or supporting-data facet, prefer the concrete relation, operation, or result over a generic summary.
- Prefer one concise complete sentence over its incomplete or noisy fragments.
- Omit background that does not answer the question.

Evidence sentences:
{format_evidence_catalog(catalog)}
"""


def parse_fact_contract(text: str) -> dict | None:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _support_score(claim: str, evidence: str) -> tuple[float, str]:
    fragment_scan = _INLINE_REFERENCE_RE.sub("", _plain(claim))
    if _BROKEN_FRAGMENT_RE.search(fragment_scan):
        return 0.0, "claim contains a broken evidence fragment"
    claim_tokens = _tokens(claim)
    evidence_tokens = _tokens(evidence)
    if len(claim_tokens) < 3:
        return 0.0, "claim too short"
    missing_numbers = _numbers(claim) - _numbers(evidence)
    if missing_numbers:
        return 0.0, f"numbers absent from evidence: {', '.join(sorted(missing_numbers))}"
    if claim_tokens & _NEGATIONS and not evidence_tokens & _NEGATIONS:
        return 0.0, "negation or absence relation absent from evidence"
    coverage = len(claim_tokens & evidence_tokens) / len(claim_tokens)
    if coverage < 0.68:
        return coverage, f"lexical coverage {coverage:.2f} below 0.68"
    return coverage, ""


def _requirement_coverage(
    requirements: list[dict],
    catalog: list[dict],
    facts: list[dict],
) -> list[dict]:
    selected_ids = {
        fact.get("evidence_id") for fact in facts if fact.get("evidence_id")
    }
    coverage = []
    for requirement in requirements:
        available = _rank_requirement(requirement, catalog)
        target = min(int(requirement.get("minimum", 1)), len(available))
        required = (
            available[:target]
            if requirement.get("kind") == "stability_values"
            else available
        )
        matched = [item["id"] for item in required if item["id"] in selected_ids]
        coverage.append({
            **requirement,
            "available_count": len(available),
            "selected_evidence_ids": matched,
            "covered": bool(target) and len(matched) >= target,
        })
    return coverage


def _contract_report(
    facts: list[dict],
    rejected: list[dict],
    catalog: list[dict],
    requirements: list[dict] | None = None,
) -> dict:
    deduplicated = _deduplicate_facts(facts)
    report = {
        "schema": "fact_contract_v1",
        "facts": deduplicated,
        "rejected": rejected,
        "evidence_count": len(catalog),
    }
    if requirements:
        report["requirement_coverage"] = _requirement_coverage(
            requirements,
            catalog,
            deduplicated,
        )
    return report


def validate_fact_contract(
    data: dict | None,
    catalog: list[dict],
    requirements: list[dict] | None = None,
) -> dict:
    evidence = {item["id"]: item for item in catalog}
    rows = data.get("evidence_ids") if isinstance(data, dict) else None
    accepted, rejected = [], []
    if not isinstance(rows, list):
        rows = []
        rejected.append({"reason": "top-level evidence_ids must be a list"})
    else:
        rows = list(rows)

    if requirements:
        mapping = data.get("requirement_evidence") if isinstance(data, dict) else None
        if not isinstance(mapping, dict):
            rejected.append({"reason": "top-level requirement_evidence must be an object"})
        else:
            rows = []
            expected = {item["id"] for item in requirements}
            for requirement_id in mapping:
                if requirement_id not in expected:
                    rejected.append({
                        "requirement_id": requirement_id,
                        "reason": "unknown requirement_id",
                    })
            for requirement in requirements:
                requirement_id = requirement["id"]
                mapped = mapping.get(requirement_id)
                if not isinstance(mapped, list):
                    rejected.append({
                        "requirement_id": requirement_id,
                        "reason": "requirement evidence must be a list",
                    })
                    continue
                rows.extend(mapped[:max(1, int(requirement.get("minimum", 1)))])

    selected = set()
    for index, evidence_id in enumerate(rows, 1):
        if not isinstance(evidence_id, str) or evidence_id not in evidence:
            rejected.append({"index": index, "evidence_id": evidence_id, "reason": "unknown evidence_id"})
            continue
        if evidence_id in selected:
            continue
        selected.add(evidence_id)
        item = evidence.get(evidence_id)
        accepted.append({
            "claim": item["text"].rstrip(" ."),
            "source": item["source"],
            "evidence_id": evidence_id,
            "evidence": item["text"],
            "coverage": 1.0,
        })

    return _contract_report(accepted, rejected, catalog, requirements)


def complete_fact_contract(
    contract: dict,
    catalog: list[dict],
    focus_questions: list[str] | list[dict] | None,
    max_per_focus: int = 2,
) -> dict:
    """Fill uncovered requirements with the highest-scoring direct evidence."""
    if not focus_questions or max_per_focus < 1:
        return contract

    structured_requirements = all(isinstance(item, dict) for item in focus_questions)
    requirements = (
        focus_questions
        if structured_requirements
        else build_fact_contract_requirements("", focus_questions)
    )
    selected_ids = {
        fact.get("evidence_id")
        for fact in contract.get("facts", [])
        if fact.get("evidence_id")
    }
    additions = []

    for requirement in requirements:
        ranked = _rank_requirement(requirement, catalog)
        minimum = int(requirement.get("minimum", 1))
        target = min(
            minimum if structured_requirements else min(minimum, max_per_focus),
            len(ranked),
        )
        required = (
            ranked[:target]
            if requirement.get("kind") == "stability_values"
            else ranked
        )
        matched = {item["id"] for item in required if item["id"] in selected_ids}
        for item in required:
            if len(matched) >= target or len(additions) >= 8:
                break
            if item["id"] in selected_ids:
                matched.add(item["id"])
                continue
            additions.append({
                "claim": item["text"].rstrip(" ."),
                "source": item["source"],
                "evidence_id": item["id"],
                "evidence": item["text"],
                "coverage": 1.0,
            })
            selected_ids.add(item["id"])
            matched.add(item["id"])

    if not additions:
        completed = dict(contract)
        completed["requirement_coverage"] = _requirement_coverage(
            requirements,
            catalog,
            list(contract.get("facts", [])),
        )
        return completed
    completed = _contract_report(
        list(contract.get("facts", [])) + additions,
        list(contract.get("rejected", [])),
        catalog,
        requirements,
    )
    completed["supplemented_evidence_ids"] = [
        item["evidence_id"] for item in additions
    ]
    return completed


def _deduplicate_facts(facts: list[dict]) -> list[dict]:
    token_sets = [_tokens(fact["claim"]) for fact in facts]
    number_sets = [_numbers(fact["claim"]) for fact in facts]
    dropped = set()
    for shorter, short_tokens in enumerate(token_sets):
        if len(short_tokens) < 5:
            continue
        for longer, long_tokens in enumerate(token_sets):
            if shorter == longer or facts[shorter]["source"] != facts[longer]["source"]:
                continue
            if (
                short_tokens < long_tokens
                and number_sets[shorter] <= number_sets[longer]
            ):
                dropped.add(shorter)
                break

    kept = []
    for index, fact in enumerate(facts):
        if index in dropped:
            continue
        duplicate = next((
            current for current in kept
            if current["source"] == fact["source"]
            and current["evidence_id"] == fact["evidence_id"]
            and _numbers(current["claim"]) == number_sets[index]
            and len(_tokens(current["claim"]) & token_sets[index])
                / max(1, min(len(_tokens(current["claim"])), len(token_sets[index]))) >= 0.9
        ), None)
        if duplicate is None:
            kept.append(fact)
    return kept


def bind_fact_list(knowledge_base: str, catalog: list[dict]) -> dict:
    accepted, rejected = [], []
    for line in (knowledge_base or "").splitlines():
        match = _FACT_RE.match(line)
        if not match:
            continue
        source = match.group("source").strip()
        claim = match.group("claim").strip()
        best = None
        for item in catalog:
            if item["source"] != source:
                continue
            score, reason = _support_score(claim, item["text"])
            if not reason and (best is None or score > best[0]):
                best = (score, item)
        if not best:
            rejected.append({"claim": claim, "source": source, "reason": "no single supporting evidence"})
            continue
        score, item = best
        accepted.append({
            "claim": item["text"].rstrip(" ."),
            "source": source,
            "evidence_id": item["id"],
            "evidence": item["text"],
            "coverage": round(score, 3),
        })
    return _contract_report(accepted, rejected, catalog)


def fact_list_from_contract(contract: dict) -> str:
    return "\n\n".join(
        f"[Fact {index}] {fact['claim']} (Source: {fact['source']})"
        for index, fact in enumerate(contract.get("facts", []), 1)
    )


def contract_is_usable(contract: dict) -> bool:
    accepted = len(contract.get("facts", []))
    rejected = len(contract.get("rejected", []))
    return accepted >= 2 and accepted / max(1, accepted + rejected) >= 0.5


def render_fact_contract(contract: dict) -> tuple[str, list[str]]:
    claims = [
        f"{fact['claim'].rstrip(' .')} [Source: {fact['source']}]."
        for fact in contract.get("facts", [])
    ]
    if not claims:
        return "", []
    return "## [Direct Paper Evidence]\n\n" + "\n".join(f"- {claim}" for claim in claims), claims
