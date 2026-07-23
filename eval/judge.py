# eval/judge.py
# Correctness evaluation only. This module is not part of the product pipeline.

import json
import os
import re
import sys
import unicodedata

try:
    import requests
except ModuleNotFoundError:  # Offline self-checks do not need the HTTP client.
    requests = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg
from rag.query_grounding_flow import split_into_sentences


_JUDGE_SYSTEM = (
    "You are a scientific answer grader. Compare the CANDIDATE against the REFERENCE, which is "
    "the sole ground truth. Score only coverage and agreement. Penalize only contradictions, "
    "incorrect reference facts, or omitted key reference facts. Ignore style, language, citations, "
    "and extra details; grounding is evaluated separately."
)

_RUBRIC = (
    "Score 1-5:\n"
    "5 = all key reference facts present and correct, none contradicted\n"
    "4 = mostly covered, with one minor fact missing or imprecise\n"
    "3 = some key facts missing, or one notable contradiction/error\n"
    "2 = most key facts missing or wrong\n"
    "1 = fails to address or broadly contradicts the reference\n"
    "For false-premise questions, score 5 when the candidate flags the premise and does not fabricate."
)

_FACT_JUDGE_SYSTEM = (
    "You are a scientific fact auditor. The numbered REFERENCE FACTS are the sole ground truth. "
    "Audit every fact independently against the entire CANDIDATE. A fact may be supported by more "
    "than one candidate passage. Ignore style, language, citations, and extra details. Return JSON only."
)

_TRANSLATION_JUDGE_SYSTEM = (
    "You are a scientific translation auditor. The English SOURCE is the sole ground truth and the "
    "TARGET is intended to be a Taiwan Traditional Chinese translation. Find semantic errors only. "
    "Audit the translation literally; never fact-check, reinterpret, or correct the SOURCE. "
    "A technical term left in English, or shown as Chinese plus English, is semantically faithful and "
    "MUST NOT be reported as an error. If direction words and values are preserved, do not invent an "
    "error based on how the scientific variable might be interpreted. Ignore style, fluency, markdown, "
    "and citations. Taiwan amino-acid names using 胺酸 are valid; for example, tyrosine=酪胺酸 and "
    "phenylalanine=苯丙胺酸. Do not require Mainland Chinese 氨酸 variants. Return JSON only."
)

_TRANSLATION_AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "errors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "mistranslation", "omission", "addition", "number_unit",
                            "negation_relation", "untranslated",
                        ],
                    },
                    "severity": {"type": "string", "enum": ["minor", "material"]},
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                    "target_ids": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": ["type", "severity", "source_ids", "target_ids", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["errors"],
    "additionalProperties": False,
}


def _generate(system: str, prompt: str, model: str, base_url: str,
              timeout: int, json_mode: bool | dict = False) -> tuple[str | None, str | None]:
    if requests is None:
        return None, "judge call failed: requests is not installed"
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 2048 if json_mode else 1024,
            "num_ctx": 16384,
            "thinking": False,
        },
    }
    if json_mode:
        payload["format"] = json_mode if isinstance(json_mode, dict) else "json"
    try:
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", ""), None
    except Exception as exc:
        return None, f"judge call failed: {exc}"


def _json_object(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        start, end = (text or "").find("{"), (text or "").rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _normalized(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = re.sub(
        r"(?<=[a-z])-\s+(?=(?!(?:and|or)\b)[a-z])",
        "",
        value,
    )
    return " ".join(value.split())


_CONTRACT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in", "is",
    "it", "of", "on", "or", "that", "the", "their", "this", "to", "under", "was", "were", "with",
}
_CONTRACT_CONDITION_GROUPS = (
    ("combined", (
        "combined", "combination", "addition of preincubation", "pre-plus",
        "synergistically enhances the co-incubation",
    )),
    ("alkaline", ("alkaline", "alkali", "naoh", "basic condition")),
    ("oxidative", ("oxidative", "oxidation", "h2o2")),
    ("acidic", ("acidic", "hcl", "acetic acid")),
    ("dark", ("in the dark", "dark storage")),
)
_CONTRACT_OUTCOME_GROUPS = (
    ("stable", ("stable", "stability", "no detectable degradation")),
    ("rapid", ("rapid", "rapidly")),
    ("slow", ("slow", "slowly")),
)
_POSITIVE_RECOVERY_GROUPS = (
    ("enhancement", ("enhanc", "augment")),
)


def _without_citations(text: str) -> str:
    return re.sub(r"\[[^\]]+\]|【[^】]+】", " ", str(text or ""))


def _contract_numbers(text: str) -> set[str]:
    plain = _without_citations(text)
    plain = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", plain)
    plain = re.sub(r"(?<=[A-Za-z])_\{?(\d+)\}?", r"\1", plain)
    values = re.findall(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?", plain)
    return {value.rstrip("0").rstrip(".") if "." in value else value for value in values}


def _contract_tokens(text: str) -> set[str]:
    normalized = _normalized(_without_citations(text))
    return {
        token for token in re.findall(r"[a-z][a-z0-9]+", normalized)
        if token not in _CONTRACT_STOPWORDS
    }


def _missing_contract_groups(fact: str, evidence: str, groups) -> set[str]:
    fact_text = _normalized(fact)
    evidence_text = _normalized(evidence)
    return {
        name for name, aliases in groups
        if any(alias in fact_text for alias in aliases)
        and not any(alias in evidence_text for alias in aliases)
    }


def _apply_fact_contract(fact: str, verdict: str, evidence: list[str]) -> tuple[str, str]:
    evidence_text = " ".join(evidence)
    if verdict == "covered":
        missing_numbers = _contract_numbers(fact) - _contract_numbers(evidence_text)
        dependent_facets = set(re.findall(r"\b([a-z][a-z0-9]*)[- ]dependent\b", _normalized(fact)))
        missing_facets = dependent_facets - _contract_tokens(evidence_text)
        missing_groups = _missing_contract_groups(
            fact,
            evidence_text,
            _CONTRACT_CONDITION_GROUPS + _CONTRACT_OUTCOME_GROUPS,
        )
        if missing_numbers or missing_facets or missing_groups:
            missing = sorted(missing_numbers | missing_facets | missing_groups)
            return "missing", f"deterministic contract missing required elements: {', '.join(missing)}"
    elif verdict == "contradicted":
        missing_groups = _missing_contract_groups(
            fact, evidence_text, _CONTRACT_CONDITION_GROUPS
        )
        fact_text = _normalized(fact)
        evidence_normalized = _normalized(evidence_text)
        if (
            " alone" in f" {fact_text}"
            and " alone" not in f" {evidence_normalized}"
            and re.search(
                r"\b(?:combined|combination|pre-plus)\b|pre\s*\+\s*co|addition of preincubation",
                evidence_normalized,
            )
        ):
            missing_groups.add("standalone")
        if missing_groups:
            return "missing", (
                "deterministic contract found insufficient condition scope for contradiction: "
                + ", ".join(sorted(missing_groups))
            )
        fact_tokens = _contract_tokens(fact)
        overlap = len(fact_tokens & _contract_tokens(evidence_text)) / len(fact_tokens) if fact_tokens else 0.0
        if overlap < 0.4:
            return "missing", "deterministic contract found insufficient entity/condition overlap for contradiction"
    return verdict, ""


def _positive_contract_witness(fact: str, candidate_items: list[dict]) -> dict | None:
    fact_text = _normalized(fact)
    active_groups = [
        group for group in _POSITIVE_RECOVERY_GROUPS
        if any(alias in fact_text for alias in group[1])
    ]
    if not active_groups:
        return None
    fact_tokens = _contract_tokens(fact)
    for item in candidate_items:
        evidence = str(item.get("text", ""))
        evidence_text = _normalized(evidence)
        if (
            re.search(r"\b(?:no|not|never|without)\b", evidence_text)
            and not re.search(r"\b(?:no|not|never|without)\b", fact_text)
        ):
            continue
        if _missing_contract_groups(fact, evidence, active_groups):
            continue
        if _apply_fact_contract(fact, "covered", [evidence])[0] != "covered":
            continue
        overlap = len(fact_tokens & _contract_tokens(evidence)) / len(fact_tokens) if fact_tokens else 0.0
        if overlap >= 0.4:
            return item
    return None


def _fact_items(reference_facts: list) -> list[dict]:
    return [
        {"id": f"F{i}", "fact": str(fact).strip()}
        for i, fact in enumerate(reference_facts or [], 1)
        if str(fact).strip()
    ]


def _candidate_items(candidate: str) -> list[dict]:
    """Number candidate passages so the judge selects evidence instead of copying it."""
    texts = split_into_sentences(candidate)
    for line in (candidate or "").splitlines():
        heading = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", line.strip()).strip()
        if (
            heading.endswith(":")
            and len(re.findall(r"\w+", heading)) > 3
            and heading not in texts
        ):
            texts.append(heading)
    items = [{"id": f"C{i}", "text": text} for i, text in enumerate(texts, 1)]
    if not items and (candidate or "").strip():
        items.append({"id": "C1", "text": candidate.strip()})
    return items


def _validate_fact_audit(
    data: dict | None,
    facts: list[dict],
    candidate: str,
    candidate_items: list[dict] | None = None,
    stable_protocol: bool | None = None,
) -> tuple[list, list]:
    expected = {item["id"]: item["fact"] for item in facts}
    rows = data.get("facts") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return [], ["top-level 'facts' must be a list"]

    if stable_protocol is None:
        stable_protocol = getattr(cfg, "STRUCTURED_JUDGE_STABLE_PROTOCOL_ENABLED", False)
    candidate_items = candidate_items or _candidate_items(candidate)
    candidate_lookup = {item["id"]: item["text"] for item in candidate_items}
    found, errors = {}, []
    candidate_norm = _normalized(candidate)
    for row in rows:
        if not isinstance(row, dict):
            errors.append("every fact result must be an object")
            continue
        fact_id = str(row.get("id", ""))
        verdict = str(row.get("verdict", "")).lower()
        if fact_id not in expected:
            errors.append(f"unexpected fact id {fact_id!r}")
            continue
        if fact_id in found:
            errors.append(f"duplicate fact id {fact_id}")
            continue
        if verdict not in {"covered", "missing", "contradicted"}:
            errors.append(f"{fact_id} has invalid verdict {verdict!r}")
            continue

        evidence_ids = []
        if stable_protocol:
            evidence_ids = row.get("evidence_ids", [])
            if isinstance(evidence_ids, str):
                evidence_ids = [evidence_ids]
            if not isinstance(evidence_ids, list) or any(not isinstance(x, str) for x in evidence_ids):
                errors.append(f"{fact_id} evidence_ids must be a string list")
                continue
            evidence_ids = [value.strip().upper() for value in evidence_ids if value.strip()]
            unknown_ids = [value for value in evidence_ids if value not in candidate_lookup]
            if unknown_ids:
                errors.append(f"{fact_id} has unknown evidence_ids: {', '.join(unknown_ids)}")
                continue
            evidence = [candidate_lookup[value] for value in evidence_ids]
        else:
            evidence = row.get("evidence", [])
            if isinstance(evidence, str):
                evidence = [evidence]
            if not isinstance(evidence, list) or any(not isinstance(x, str) for x in evidence):
                errors.append(f"{fact_id} evidence must be a string list")
                continue
            evidence = [x.strip() for x in evidence if x.strip()]
            bad_quotes = [quote for quote in evidence if _normalized(quote) not in candidate_norm]
            if bad_quotes:
                errors.append(f"{fact_id} evidence is not a verbatim candidate excerpt")
                continue

        if verdict in {"covered", "contradicted"} and not evidence:
            errors.append(f"{fact_id} {verdict} verdict requires candidate evidence")
            continue
        if verdict == "missing" and evidence:
            errors.append(f"{fact_id} missing verdict requires empty evidence")
            continue
        judge_verdict = verdict
        contract_reason = ""
        if stable_protocol:
            witness = (
                _positive_contract_witness(expected[fact_id], candidate_items)
                if verdict == "missing" else None
            )
            if witness:
                verdict = "covered"
                evidence_ids = [witness["id"]]
                evidence = [witness["text"]]
                contract_reason = "deterministic contract found an explicit positive relation witness"
            else:
                verdict, contract_reason = _apply_fact_contract(expected[fact_id], verdict, evidence)
            if verdict == "missing":
                evidence = []
                evidence_ids = []
        found[fact_id] = {
            "id": fact_id,
            "fact": expected[fact_id],
            "verdict": verdict,
            "evidence": evidence,
            "reason": contract_reason or str(row.get("reason", "")).strip()[:300],
        }
        if verdict != judge_verdict:
            found[fact_id]["judge_verdict"] = judge_verdict
        if stable_protocol:
            found[fact_id]["evidence_ids"] = evidence_ids

    missing_ids = [fact_id for fact_id in expected if fact_id not in found]
    if missing_ids:
        errors.append(f"missing fact results: {', '.join(missing_ids)}")
    return [found[item["id"]] for item in facts if item["id"] in found], errors


def _fact_prompt(
    question: str,
    candidate: str,
    facts: list[dict],
    review: bool = False,
    correction: str = "",
    candidate_items: list[dict] | None = None,
    stable_protocol: bool | None = None,
) -> str:
    fact_text = "\n".join(f'{item["id"]}: {item["fact"]}' for item in facts)
    task = (
        "This is a second-pass review of negative verdicts. Search the entire candidate for support "
        "the first pass may have missed; overturn a verdict only when exact candidate excerpts support it."
        if review else
        "Audit every numbered fact."
    )
    repair = f"\nYour previous JSON was invalid: {correction}\nReturn a corrected complete audit." if correction else ""
    if stable_protocol is None:
        stable_protocol = getattr(cfg, "STRUCTURED_JUDGE_STABLE_PROTOCOL_ENABLED", False)
    if stable_protocol:
        candidate_items = candidate_items or _candidate_items(candidate)
        candidate_text = "\n".join(f'{item["id"]}: {item["text"]}' for item in candidate_items)
        schema = (
            '{"facts":[{"id":"F1","verdict":"covered|missing|contradicted",'
            '"evidence_ids":["C1"],"reason":"short reason"}]}\n'
            "Use only supplied candidate passage IDs. Covered or contradicted requires one or more evidence_ids; "
            "missing requires an empty evidence_ids list."
        )
        candidate_label = "CANDIDATE PASSAGES"
    else:
        candidate_text = candidate
        schema = (
            '{"facts":[{"id":"F1","verdict":"covered|missing|contradicted",'
            '"evidence":["exact contiguous excerpt copied from CANDIDATE"],"reason":"short reason"}]}\n'
            "Evidence may contain multiple exact excerpts when support is distributed. For missing, return an empty evidence list."
        )
        candidate_label = "CANDIDATE"
    return (
        f"{task}{repair}\n\nQUESTION:\n{question}\n\nREFERENCE FACTS:\n{fact_text}\n\n"
        f"{candidate_label}:\n{candidate_text}\n\nReturn exactly this JSON shape:\n{schema}\n"
        "Use every supplied fact id exactly once. Use covered only when the complete fact is expressed; "
        "use missing only when the fact is absent, and use contradicted only for an opposing claim about "
        "the same entity under the same experimental or storage conditions. A different condition is missing, "
        "not contradicted. "
        "Do not output a score."
    )


def _request_fact_audit(question: str, candidate: str, facts: list[dict], model: str,
                        base_url: str, timeout: int, review: bool = False) -> tuple[list | None, str | None, int]:
    correction = ""
    stable_protocol = getattr(cfg, "STRUCTURED_JUDGE_STABLE_PROTOCOL_ENABLED", False)
    candidate_items = _candidate_items(candidate)
    for attempt in range(1, 3):
        output, error = _generate(
            _FACT_JUDGE_SYSTEM,
            _fact_prompt(
                question,
                candidate,
                facts,
                review,
                correction,
                candidate_items=candidate_items,
                stable_protocol=stable_protocol,
            ),
            model,
            base_url,
            timeout,
            json_mode=True,
        )
        if error:
            return None, error, attempt
        audit, errors = _validate_fact_audit(
            _json_object(output),
            facts,
            candidate,
            candidate_items=candidate_items,
            stable_protocol=stable_protocol,
        )
        if not errors:
            return audit, None, attempt
        correction = "; ".join(errors)
    return None, f"invalid structured judge output: {correction}", 2


def _score_fact_audit(audit: list[dict]) -> tuple[int, float, str]:
    total = len(audit)
    covered = [item["id"] for item in audit if item["verdict"] == "covered"]
    missing = [item["id"] for item in audit if item["verdict"] == "missing"]
    contradicted = [item["id"] for item in audit if item["verdict"] == "contradicted"]
    ratio = len(covered) / total if total else 0.0
    if ratio == 1.0 and not contradicted:
        raw = 5
    elif ratio >= 0.8 and not contradicted:
        raw = 4
    elif ratio >= 0.5:
        raw = 3
    elif ratio > 0:
        raw = 2
    else:
        raw = 1
    if contradicted:
        raw = min(raw, 3)
    parts = [f"covered {len(covered)}/{total}"]
    if missing:
        parts.append(f"missing {', '.join(missing)}")
    if contradicted:
        parts.append(f"contradicted {', '.join(contradicted)}")
    return raw, (raw - 1) / 4.0, "; ".join(parts)


def _judge_holistic(
    question: str,
    candidate: str,
    reference: str,
    model: str,
    base_url: str,
    timeout: int,
    mode: str = "legacy_holistic",
) -> dict:
    prompt = (
        f"{_RUBRIC}\n\nQUESTION:\n{question}\n\nREFERENCE (ground truth):\n{reference}\n\n"
        f"CANDIDATE (system answer):\n{candidate}\n\n"
        "Output exactly two lines:\nSCORE: <1-5>\nREASON: <one sentence>"
    )
    output, error = _generate(_JUDGE_SYSTEM, prompt, model, base_url, timeout)
    return _parse_scalar_score(output, error, mode)


def _parse_scalar_score(output: str | None, error: str | None, mode: str) -> dict:
    if error:
        return {"score": None, "raw": None, "reason": error, "mode": mode}
    match = re.search(r"SCORE:\s*([1-5])", output or "") or re.search(r"\b([1-5])\s*/\s*5\b", output or "")
    if not match:
        return {
            "score": None,
            "raw": None,
            "reason": f"unparseable judge output: {(output or '')[:120]}",
            "mode": mode,
        }
    raw = int(match.group(1))
    reason_match = re.search(r"REASON:\s*(.+)", output or "", re.S)
    reason = reason_match.group(1).strip()[:300] if reason_match else (output or "")[:200].strip()
    return {"score": (raw - 1) / 4.0, "raw": raw, "reason": reason, "mode": mode}


def _judge_structured(question: str, candidate: str, reference: str, reference_facts: list,
                      model: str, base_url: str, timeout: int) -> dict:
    facts = _fact_items(reference_facts)
    audit, error, first_attempts = _request_fact_audit(
        question, candidate, facts, model, base_url, timeout,
    )
    if error:
        if (
            getattr(cfg, "STRUCTURED_JUDGE_STABLE_PROTOCOL_ENABLED", False)
            and error.startswith("invalid structured judge output:")
        ):
            fallback = _judge_holistic(
                question,
                candidate,
                reference,
                model,
                base_url,
                timeout,
                mode="legacy_holistic_fallback",
            )
            fallback["structured_error"] = error
            fallback["judge_attempts"] = first_attempts + 1
            return fallback
        return {
            "score": None,
            "raw": None,
            "reason": error,
            "mode": "structured_fact_audit_v1",
        }

    reviewed_ids = []
    disputed = [item for item in facts if next(x for x in audit if x["id"] == item["id"])["verdict"] != "covered"]
    review_attempts = 0
    review_error = None
    if disputed:
        review, review_error, review_attempts = _request_fact_audit(
            question, candidate, disputed, model, base_url, timeout, review=True,
        )
        if review:
            initial = {item["id"]: item for item in audit}
            for item in review:
                original = initial[item["id"]]
                if item["verdict"] == "covered":
                    item["initial_verdict"] = original["verdict"]
                    initial[item["id"]] = item
                else:
                    original["review_verdict"] = item["verdict"]
                    original["review_reason"] = item["reason"]
                reviewed_ids.append(item["id"])
            audit = [initial[item["id"]] for item in facts]

    raw, score, reason = _score_fact_audit(audit)
    result = {
        "score": score,
        "raw": raw,
        "reason": reason,
        "mode": "structured_fact_audit_v1",
        "fact_audit": audit,
        "reviewed_ids": reviewed_ids,
        "judge_attempts": first_attempts + review_attempts,
    }
    if review_error:
        result["review_error"] = review_error
    return result


def judge_correctness(question: str, candidate: str, reference: str,
                      model: str = None, base_url: str = None, timeout: int = 600,
                      reference_facts: list | None = None) -> dict:
    """Return score in 0..1. Curated reference_facts enable auditable structured judging."""
    model = model or getattr(cfg, "JUDGE_MODEL", cfg.VERIFY_MODEL)
    base_url = base_url or cfg.OLLAMA_BASE_URL
    if not (reference or "").strip() or not (candidate or "").strip():
        return {"score": None, "raw": None, "reason": "missing reference or candidate"}
    if _fact_items(reference_facts):
        return _judge_structured(
            question,
            candidate,
            reference,
            reference_facts,
            model,
            base_url,
            timeout,
        )
    return _judge_holistic(question, candidate, reference, model, base_url, timeout)


def _translation_items(text: str, prefix: str) -> list[dict]:
    items = split_into_sentences(text)
    if not items and (text or "").strip():
        items = [text.strip()]
    return [{"id": f"{prefix}{i}", "text": item} for i, item in enumerate(items, 1)]


def _number_unit_signature(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    normalized = unicodedata.normalize("NFKC", _without_citations(text)).lower().replace("μ", "u").replace("µ", "u")
    numbers = tuple(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", normalized))
    units = tuple(re.findall(
        r"(?<![a-z])(?:%|°\s*[cf]|[numk]?m|kg|mg|ug|g|ml|l|h|hr|hours?|min|minutes?|days?|months?|years?)(?![a-z])",
        normalized,
    ))
    return numbers, units


_TAIWAN_TERM_EQUIVALENTS = {
    "tyrosine": ("酪胺酸", "酪氨酸"),
    "phenylalanine": ("苯丙胺酸", "苯丙氨酸"),
}
_TRANSLATION_PHRASE_EQUIVALENTS = (
    ("bench scale", ("bench scale", "實驗室規模", "小試規模")),
    ("room temperature", ("room temperature", "室溫")),
    ("water-stable", ("water-stable", "water stable", "水穩定", "耐水")),
    ("later-stage", ("later-stage", "later stage", "後期", "後階段")),
    (
        "cell membrane disruption",
        ("cell membrane disruption", "細胞膜破裂", "細胞膜破壞", "細胞膜受損"),
    ),
)


def _translation_term_false_positive(kind: str, reason: str, source: str, target: str) -> bool:
    if kind not in {"mistranslation", "untranslated"} or not re.search(
        r"translat|technical term|chemical name|corresponds to", reason, re.I
    ):
        return False
    source_lower, reason_lower, target_lower = source.lower(), reason.lower(), target.lower()
    for term, aliases in _TAIWAN_TERM_EQUIVALENTS.items():
        if term in source_lower and term in reason_lower and (
            term in target_lower or any(alias in target for alias in aliases)
        ):
            return True
    retained_terms = re.findall(r"\b[A-Za-z][A-Za-z0-9-]{4,}\b", source)
    return any(term.lower() in reason_lower and term.lower() in target_lower for term in retained_terms)


def _translation_structural_omission_false_positive(kind: str, reason: str) -> bool:
    return bool(
        kind == "omission"
        and re.search(
            r"\b(?:content|meaning|information)\b.{0,120}\b(?:merged|present|included|conveyed)\b",
            reason,
            re.I,
        )
        and re.search(
            r"\b(?:structure|standalone|caption|sentence|distinct elements?|format)\b",
            reason,
            re.I,
        )
    )


def _translation_omission_witness_present(
    kind: str,
    reason: str,
    source: str,
    target: str,
) -> bool:
    if kind != "omission":
        return False
    quoted = [
        next(value for value in match if value).strip()
        for match in re.findall(r"'([^']+)'|\"([^\"]+)\"|`([^`]+)`", reason or "")
        if any(match)
    ]
    source_text = _normalized(source)
    phrases = [
        phrase for phrase in quoted
        if _normalized(phrase) and _normalized(phrase) in source_text
    ]
    if not phrases:
        return False

    target_text = _normalized(target)
    target_numbers = _contract_numbers(target)
    for phrase in phrases:
        phrase_text = _normalized(phrase)
        if phrase_text in target_text:
            continue
        groups = [
            aliases for key, aliases in _TRANSLATION_PHRASE_EQUIVALENTS
            if key in phrase_text
        ]
        if not groups:
            return False
        if any(
            not any(_normalized(alias) in target_text for alias in aliases)
            for aliases in groups
        ):
            return False
        if _contract_numbers(phrase) - target_numbers:
            return False
    return True


def _validate_translation_audit(data: dict | None, source: str, target: str) -> tuple[list, list]:
    rows = data.get("errors") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return [], ["top-level 'errors' must be a list"]
    source_items = _translation_items(source, "S")
    target_items = _translation_items(target, "T")
    source_lookup = {item["id"]: item["text"] for item in source_items}
    target_lookup = {item["id"]: item["text"] for item in target_items}
    valid, errors = [], []
    allowed_types = {"mistranslation", "omission", "addition", "number_unit", "negation_relation", "untranslated"}
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            errors.append(f"error {index} must be an object")
            continue
        kind = str(row.get("type", "")).lower()
        severity = str(row.get("severity", "")).lower()
        source_ids = row.get("source_ids", [])
        target_ids = row.get("target_ids", [])
        if kind not in allowed_types or severity not in {"minor", "material"}:
            errors.append(f"error {index} has invalid type or severity")
            continue
        if not isinstance(source_ids, list) or not isinstance(target_ids, list):
            errors.append(f"error {index} sentence ids must be lists")
            continue
        if any(value not in source_lookup for value in source_ids) or any(value not in target_lookup for value in target_ids):
            errors.append(f"error {index} has unknown sentence ids")
            continue
        if not source_ids and not target_ids:
            errors.append(f"error {index} must cite at least one sentence id")
            continue
        source_text = " ".join(source_lookup[value] for value in source_ids)
        target_text = " ".join(target_lookup[value] for value in target_ids)
        reason = str(row.get("reason", "")).strip()[:300]
        if _translation_term_false_positive(kind, reason, source_text, target):
            continue
        if _translation_structural_omission_false_positive(kind, reason):
            continue
        if (
            getattr(cfg, "TRANSLATION_OMISSION_WITNESS_FILTER_ENABLED", False)
            and _translation_omission_witness_present(kind, reason, source_text, target)
        ):
            continue
        if kind == "number_unit" and getattr(cfg, "TRANSLATION_EXACT_VALUE_FILTER_ENABLED", False):
            source_signature = _number_unit_signature(source_text)
            if source_signature[0] and source_signature == _number_unit_signature(target_text):
                continue
        valid.append({
            "type": kind,
            "severity": severity,
            "source_ids": source_ids,
            "target_ids": target_ids,
            "source": [source_lookup[value] for value in source_ids],
            "target": [target_lookup[value] for value in target_ids],
            "reason": reason,
        })
    return valid, errors


def _translation_prompt(source: str, target: str, correction: str = "") -> str:
    source_items = _translation_items(source, "S")
    target_items = _translation_items(target, "T")
    source_text = "\n".join(f'{item["id"]}: {item["text"]}' for item in source_items)
    target_text = "\n".join(f'{item["id"]}: {item["text"]}' for item in target_items)
    repair = f"\nYour previous JSON was invalid: {correction}\n" if correction else ""
    return (
        "Report only meaning-changing scientific errors: mistranslation, omission, addition, "
        "number/unit error, negation/relation error, or a substantially untranslated sentence. "
        "Do not report retained English technical terms, wording preference, style, or fluency. "
        "Use severity=minor only when scientific meaning remains intact; otherwise material. "
        "Sentence IDs are alignment anchors, not a one-to-one mapping: a source sentence may be "
        "merged into a neighboring target sentence. Search the entire target before reporting an omission. "
        "Never report omission solely because sentence, caption, or paragraph boundaries changed; if the "
        "content is merged or present, name a specific absent semantic detail or return no error.\n"
        f"{repair}\nENGLISH SOURCE SENTENCES:\n{source_text}\n\n"
        f"TRADITIONAL CHINESE TARGET SENTENCES:\n{target_text}\n\n"
        "Return exactly this JSON shape and no score:\n"
        '{"errors":[{"type":"mistranslation|omission|addition|number_unit|negation_relation|untranslated",'
        '"severity":"minor|material","source_ids":["S1"],"target_ids":["T1"],"reason":"short reason"}]}\n'
        "Return an empty errors list when the translation is semantically faithful."
    )


def _score_translation_audit(audit: list[dict]) -> tuple[int, float, str]:
    material = sum(item["severity"] == "material" for item in audit)
    minor = len(audit) - material
    if not audit:
        raw = 5
    elif material == 0 and minor == 1:
        raw = 4
    elif material <= 1:
        raw = 3
    elif material <= 3:
        raw = 2
    else:
        raw = 1
    reason = f"{material} material and {minor} minor semantic errors"
    if audit and audit[0].get("reason"):
        reason += f"; {audit[0]['reason']}"
    return raw, (raw - 1) / 4.0, reason


def judge_translation_fidelity(source: str, target: str, model: str = None,
                               base_url: str = None, timeout: int = 600) -> dict:
    """Return a deterministic score over a structured translation-error audit."""
    model = model or getattr(cfg, "JUDGE_MODEL", cfg.VERIFY_MODEL)
    base_url = base_url or cfg.OLLAMA_BASE_URL
    if not (source or "").strip() or not (target or "").strip():
        return {"score": None, "raw": None, "reason": "missing source or translation", "mode": "translation_fidelity_v2"}
    correction = ""
    for attempt in range(1, 3):
        output, error = _generate(
            _TRANSLATION_JUDGE_SYSTEM,
            _translation_prompt(source, target, correction),
            model,
            base_url,
            timeout,
            json_mode=_TRANSLATION_AUDIT_SCHEMA,
        )
        if error:
            return {"score": None, "raw": None, "reason": error, "mode": "translation_fidelity_v2"}
        audit, errors = _validate_translation_audit(_json_object(output), source, target)
        if not errors:
            raw, score, reason = _score_translation_audit(audit)
            return {
                "score": score,
                "raw": raw,
                "reason": reason,
                "mode": "translation_fidelity_v2",
                "error_audit": audit,
                "judge_attempts": attempt,
            }
        correction = "; ".join(errors)
    return {
        "score": None,
        "raw": None,
        "reason": f"invalid translation audit: {correction}",
        "mode": "translation_fidelity_v2",
        "judge_attempts": 2,
    }


if __name__ == "__main__":
    candidate = "Producing high-purity, isotopically enriched 10B material is a challenge."
    facts = _fact_items(["High-purity isotopically enriched 10B material is difficult to produce."])
    valid = {"facts": [{
        "id": "F1",
        "verdict": "covered",
        "evidence_ids": ["C1"],
        "reason": "directly stated",
    }]}
    audit, errors = _validate_fact_audit(valid, facts, candidate, stable_protocol=True)
    assert not errors and _score_fact_audit(audit)[:2] == (5, 1.0)
    assert _validate_fact_audit({"facts": []}, facts, candidate)[1]
    assert judge_correctness("q", "ans", "")["score"] is None
    print("judge.py self-check OK")
