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
    r"[.!?]\s+(?:approximately|about|roughly|nearly|up\s+to|at\s+least|at\s+most|\d)",
    re.IGNORECASE,
)


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
    return set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", _plain(text)))


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
    if "corresponding author" in plain:
        return True
    plot_labels = re.search(r"\b(?:mau|time\s*\(min\))\b", plain, re.IGNORECASE)
    figure = re.search(r"\bfig\.", plain, re.IGNORECASE)
    return bool(plot_labels and figure)


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
                r"(?<!\bFig\.)(?<!\bEq\.)(?<!\bDr\.)(?<!\bMr\.)(?<=[.!?])\s+",
                clean,
                flags=re.IGNORECASE,
            )
            sentences = [sentence.strip(" -\n") for sentence in sentences]
            windows, superseded = _interrupted_sentence_windows(sentences)
            for sentence in sentences + windows:
                if (
                    sentence in superseded
                    or (sentence[:1].islower() and _FRAGMENT_RE.match(sentence))
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


def fact_contract_schema(catalog: list[dict]) -> dict:
    evidence_ids = [item["id"] for item in catalog]
    return {
        "type": "object",
        "properties": {
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string", "enum": evidence_ids},
                "minItems": 1,
                "maxItems": min(20, len(evidence_ids)),
            },
        },
        "required": ["evidence_ids"],
        "additionalProperties": False,
    }


def fact_contract_prompt(catalog: list[dict], query: str, recovery_hint: str = "") -> str:
    recovery = f"\nRecovery focus: {recovery_hint}\n" if recovery_hint else ""
    return f"""Select all and only the evidence IDs needed to answer the question.

Question: {query}
{recovery}

Rules:
- Return evidence_ids only; the application will restore source and text.
- Select direct evidence, not background or inferred relationships.
- Include every sentence needed for requested values, conditions, outcomes, and comparisons.
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
    if _BROKEN_FRAGMENT_RE.search(_plain(claim)):
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


def _contract_report(facts: list[dict], rejected: list[dict], catalog: list[dict]) -> dict:
    return {
        "schema": "fact_contract_v1",
        "facts": _deduplicate_facts(facts),
        "rejected": rejected,
        "evidence_count": len(catalog),
    }


def validate_fact_contract(data: dict | None, catalog: list[dict]) -> dict:
    evidence = {item["id"]: item for item in catalog}
    rows = data.get("evidence_ids") if isinstance(data, dict) else None
    accepted, rejected = [], []
    if not isinstance(rows, list):
        rows = []
        rejected.append({"reason": "top-level evidence_ids must be a list"})

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

    return _contract_report(accepted, rejected, catalog)


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
