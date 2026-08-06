import json
import re

import config as cfg


_ABSENCE_MARKERS = (
    "does not explicitly", "not explicitly", "does not contain", "not contain",
    "did not provide", "does not provide", "not provide", "not reported",
    "unaddressed", "lacks", "missing",
)
_EVIDENCE_DIMENSION_MARKERS = {
    "isotopic_enrichment": ("isotop", "enrich"),
    "scalability": ("scalab", " scale", "scale-", "workup", "few reaction steps"),
    "cost_effectiveness": ("cost", "expens", "economic"),
    "safety": ("safety", "safe", "risk", "toxic", "contamination"),
}
_ISOTOPE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"(?P<mass>\d{1,3})-?(?P<symbol>Cl|Br|H|B|C|N|O|F|P|S|I)"
    r"|(?P<name>hydrogen|boron|carbon|nitrogen|oxygen|fluorine|phosphorus|sulfur|chlorine|bromine|iodine)"
    r"\s*-?\s*(?P<name_mass>\d{1,3})"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_ISOTOPE_CONTEXT_MARKERS = (
    "isotop", "enrich", "label", "radio", "neutron", "bnct",
)
_ISOTOPE_SYMBOLS = {
    "h": "H", "b": "B", "c": "C", "n": "N", "o": "O", "f": "F",
    "p": "P", "s": "S", "cl": "Cl", "br": "Br", "i": "I",
    "hydrogen": "H", "boron": "B", "carbon": "C", "nitrogen": "N",
    "oxygen": "O", "fluorine": "F", "phosphorus": "P", "sulfur": "S",
    "chlorine": "Cl", "bromine": "Br", "iodine": "I",
}
_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_NAMED_INTERACTION_RE = re.compile(
    r"\b(?P<relation>(?:halogen|hydrogen|covalent|ionic)\s+bond|salt bridge)\b"
    r"[^.!?]{0,120}\b(?:with|between)\b[^.!?]{0,60}"
    r"\b(?P<anchor>(?:Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|"
    r"Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)\d+)\b",
    re.IGNORECASE,
)
_STRATEGY_QUALIFIER_RE = re.compile(
    r"\b(?:competitiv\w*|minimally toxic|self[- ]?assembl\w*|proliferat\w*|"
    r"structural basis|targeting motif)\b",
    re.IGNORECASE,
)
_TARGET_ACTION = (
    r"(?:inhibit\w*|suppress\w*|block\w*|antagoni[sz]\w*|degrad\w*|"
    r"silenc\w*|knock(?:down|ed)?|bind\w*|bound|sensiti[sz]\w*)"
)
_STRATEGY_ACTION = rf"(?:{_TARGET_ACTION}|design\w*|develop\w*|conjugat\w*)"


def query_dimension_keys(query: str) -> set[str]:
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


def query_requests_mechanism(query: str) -> bool:
    text = (query or "").lower()
    return "mechanism" in text or "mechanistic" in text or "機制" in text


def query_target(query: str) -> str:
    match = re.search(
        r"\btarget(?:s|ed|ing)?\s+(?:the\s+)?([A-Za-z][A-Za-z0-9-]{2,})\b",
        query or "",
        re.IGNORECASE,
    )
    return match.group(1) if match else ""


def direct_route_targets_query_target(route: dict, target: str) -> bool:
    if not target:
        return True
    text = " ".join(
        str(route.get(key, ""))
        for key in ("route_phrase", "outcome", "evidence")
    )
    target_pattern = rf"\b{re.escape(target)}\b"
    if not re.search(target_pattern, text, re.IGNORECASE):
        return bool(re.search(_TARGET_ACTION, text, re.IGNORECASE))
    return any(
        re.search(target_pattern, sentence, re.IGNORECASE)
        and re.search(_TARGET_ACTION, sentence, re.IGNORECASE)
        for sentence in re.split(r"(?<=[.!?])\s+", text)
    )


def is_synthetic_route_query(query: str) -> bool:
    text = (query or "").lower()
    return any(term in text for term in (
        "synthesi", "synthetic route", "preparation method", "manufactur", "合成", "製備",
    ))


def comparison_json_payload(text: str):
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
        return json.loads(stripped, strict=False)
    except json.JSONDecodeError:
        return None


def exact_isotope_terms(text: str, require_context: bool = True) -> list[str]:
    normalized = str(text or "").translate(_SUPERSCRIPT_DIGITS)
    terms = []
    for match in _ISOTOPE_RE.finditer(normalized):
        prefix = normalized[max(0, match.start() - 16):match.start()]
        suffix = normalized[match.end():match.end() + 12]
        if re.search(r"(?:fig(?:ure)?|scheme|table)\.?\s*$", prefix, re.IGNORECASE):
            continue
        if re.match(r"\s*-?\s*NMR\b", suffix, re.IGNORECASE):
            continue
        context = normalized[max(0, match.start() - 120):match.end() + 120].lower()
        if require_context and not any(marker in context for marker in _ISOTOPE_CONTEXT_MARKERS):
            continue
        mass = match.group("mass") or match.group("name_mass")
        element = match.group("symbol") or match.group("name")
        term = f"{int(mass)}{_ISOTOPE_SYMBOLS[element.lower()]}"
        if term not in terms:
            terms.append(term)
    return terms


def _citable_evidence(text: str) -> str:
    marker = "Retrieved evidence snippets:"
    value = str(text or "")
    return value.split(marker, 1)[1] if marker in value else value


def _evidence_units(text: str) -> list[str]:
    parts = re.split(r"(?m)^\s*\[Snippet \d+\]\s*", str(text or ""))
    units = [part.strip() for part in parts if part.strip()]
    return units or [str(text or "")]


def _evidence_sentences(text: str) -> list[str]:
    boundary = re.compile(
        r"(?<!\bFig\.)(?<!\bFigs\.)(?<!\bet al\.)(?<=[.!?])"
        r"(?:\d+(?:[,–-]\d+)*)?\s+(?=[A-Z])"
    )
    return [
        sentence.strip()
        for unit in _evidence_units(text)
        for sentence in boundary.split(unit)
        if sentence.strip()
    ]


def build_comparison_requirements(query: str, chunks: list[dict]) -> dict:
    rows = []
    for index, chunk in enumerate(chunks or []):
        source = str(chunk.get("source") or f"chunk_{index}").strip()
        text = str(chunk.get("text") or chunk.get("content") or "")
        rows.append({
            "source": source,
            "review": "role_hint=review/comparison source" in text.lower(),
            "evidence": _citable_evidence(text),
        })

    requested = sorted(query_dimension_keys(query))
    review_rows = [row for row in rows if row["review"]]
    review_sources = list(dict.fromkeys(row["source"] for row in review_rows))
    dimension_sources = {}
    for key in requested:
        markers = _EVIDENCE_DIMENSION_MARKERS[key]
        sources = [
            row["source"] for row in review_rows
            if (
                any(marker in row["evidence"].lower() for marker in markers)
                or (
                    key == "isotopic_enrichment"
                    and bool(exact_isotope_terms(row["evidence"]))
                )
            )
        ]
        if sources:
            dimension_sources[key] = list(dict.fromkeys(sources))

    isotope_scope = review_rows or rows
    isotopes = []
    relation_isotopes_by_source = {}
    if "isotopic_enrichment" in requested:
        for row in isotope_scope:
            for term in exact_isotope_terms(row["evidence"]):
                if term not in isotopes:
                    isotopes.append(term)
            if "cost_effectiveness" in requested:
                for unit in _evidence_units(row["evidence"]):
                    if not any(
                        marker in unit.lower()
                        for marker in _EVIDENCE_DIMENSION_MARKERS["cost_effectiveness"]
                    ):
                        continue
                    for term in exact_isotope_terms(unit, require_context=False):
                        source_terms = relation_isotopes_by_source.setdefault(row["source"], [])
                        if term not in source_terms:
                            source_terms.append(term)

    relation_requirements = []
    if {"isotopic_enrichment", "cost_effectiveness"}.issubset(requested):
        shared_sources = (
            set(dimension_sources.get("isotopic_enrichment", []))
            & set(dimension_sources.get("cost_effectiveness", []))
        )
        for source in review_sources:
            anchors = [
                term for term in relation_isotopes_by_source.get(source, [])
                if term in isotopes
            ]
            if source in shared_sources and anchors:
                relation_requirements.append({
                    "dimension": "cost_effectiveness",
                    "source": source,
                    "anchors": anchors,
                })

    mechanism_requirements = []
    if query_requests_mechanism(query):
        for row in rows:
            found = None
            for sentence in _evidence_sentences(row["evidence"]):
                match = _NAMED_INTERACTION_RE.search(sentence)
                if match:
                    found = {
                        "source": row["source"],
                        "anchors": [
                            match.group("relation"),
                            match.group("anchor"),
                        ],
                        "claim": re.sub(r"\s+", " ", sentence).strip(),
                    }
                    break
            if found:
                mechanism_requirements.append(found)

    target = query_target(query)
    strategy_requirements = []
    if target and query_requests_mechanism(query) and not is_synthetic_route_query(query):
        for row in rows:
            candidates = []
            for index, sentence in enumerate(_evidence_sentences(row["evidence"])):
                claim = re.sub(r"\s+", " ", sentence).strip()
                words = claim.split()
                qualifiers = _STRATEGY_QUALIFIER_RE.findall(claim)
                if (
                    not qualifiers
                    or not (6 <= len(words) <= 120)
                    or re.match(r"^\d+[A-Za-z]?\)?,", claim)
                ):
                    continue
                target_present = bool(re.search(
                    rf"\b{re.escape(target)}\b",
                    claim,
                    re.IGNORECASE,
                ))
                action_present = bool(re.search(_STRATEGY_ACTION, claim, re.IGNORECASE))
                if not action_present:
                    continue
                score = (
                    6 * target_present
                    + 2 * len({value.lower() for value in qualifiers})
                    + 3 * int(bool(re.search(
                        r"\b(?:design\w*|develop\w*|conjugat\w*)\b",
                        claim,
                        re.IGNORECASE,
                    )))
                    + 2 * int(bool(re.search(r"\bproliferation\b", claim, re.IGNORECASE)))
                    + int(bool(re.search(r"\b(?:via|through|thereby|by)\b", claim, re.IGNORECASE)))
                )
                candidates.append((score, -index, claim))
            selected = []
            for _, _, claim in sorted(candidates, reverse=True):
                tokens = set(re.findall(r"[a-z0-9]+", claim.lower()))
                if any(
                    len(tokens & seen) / max(1, min(len(tokens), len(seen))) >= 0.7
                    for seen, _ in selected
                ):
                    continue
                selected.append((tokens, claim))
                if len(selected) == 2:
                    break
            strategy_requirements.extend({
                "source": row["source"],
                "claim": claim,
            } for _, claim in selected)

    return {
        "version": 4,
        "query_target": target,
        "requested_dimensions": requested,
        "review_sources": review_sources,
        "dimension_sources": dimension_sources,
        "exact_isotopes": isotopes,
        "relation_requirements": relation_requirements,
        "mechanism_requirements": mechanism_requirements,
        "strategy_requirements": strategy_requirements,
    }


def attach_comparison_requirements(data: dict, requirements: dict) -> None:
    data["comparison_requirements"] = requirements


def _has_absence_claim(text: str) -> bool:
    lower = str(text or "").lower()
    return any(marker in lower for marker in _ABSENCE_MARKERS)


def _has_review_comparison_source(comparison: dict) -> bool:
    source_roles = comparison.get("source_roles") if isinstance(comparison.get("source_roles"), list) else []
    return (
        any(
            isinstance(item, dict) and "review/comparison" in str(item.get("role", "")).lower()
            for item in source_roles
        )
        or bool(comparison.get("review_comparison_sources"))
    )


def comparison_json_validation_errors(text: str, query: str = "") -> list[str]:
    data = comparison_json_payload(text)
    if not isinstance(data, dict):
        return ["Output is not valid JSON."]
    comparison = data.get("comparison_json")
    if not isinstance(comparison, dict):
        return ["Missing root object: comparison_json."]

    errors = []
    for field in ("source_roles", "direct_routes", "review_comparison_sources"):
        if not isinstance(comparison.get(field), list):
            errors.append(f"`{field}` must be a list.")
    mechanisms = comparison.get("supporting_mechanisms", [])
    if not isinstance(mechanisms, list):
        errors.append("`supporting_mechanisms` must be a list.")
        mechanisms = []

    dimensions = comparison.get("dimensions")
    if not isinstance(dimensions, dict):
        errors.append("`dimensions` must be an object.")
        dimensions = {}

    requirements = data.get("comparison_requirements")
    requirements = requirements if isinstance(requirements, dict) else {}
    query_dims = set(requirements.get("requested_dimensions", [])) or query_dimension_keys(query)
    has_review_source = _has_review_comparison_source(comparison)
    atomic_required = getattr(cfg, "COMPARISON_JSON_DIRECT_RENDER_ENABLED", False)
    source_roles = comparison.get("source_roles") if isinstance(comparison.get("source_roles"), list) else []
    roles_by_source = {
        str(item.get("source", "")).strip(): str(item.get("role", "")).lower()
        for item in source_roles
        if isinstance(item, dict) and item.get("source")
    }
    review_sources = {
        source for source, role in roles_by_source.items() if "review/comparison" in role
    }
    route_sources = {
        source for source, role in roles_by_source.items() if role == "route"
    }
    background_sources = {
        source for source, role in roles_by_source.items() if "background" in role
    }
    target = str(requirements.get("query_target") or query_target(query)).strip()
    direct_route_sources = {
        str(item.get("source", "")).strip()
        for item in comparison.get("direct_routes", [])
        if isinstance(item, dict) and item.get("source")
    }
    review_entry_sources = {
        str(item.get("source", "")).strip()
        for item in comparison.get("review_comparison_sources", [])
        if isinstance(item, dict) and item.get("source")
    }

    valid_mechanisms = []
    for item in mechanisms:
        if not isinstance(item, dict):
            errors.append("Every supporting mechanism must be an object.")
            continue
        source = str(item.get("source", "")).strip()
        claim = str(item.get("claim", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        if not source or not claim or not evidence:
            errors.append("Every supporting mechanism requires source, claim, and evidence.")
            continue
        if source not in roles_by_source:
            errors.append(f"Supporting mechanism source `{source}` is missing from source_roles.")
            continue
        if source in background_sources:
            errors.append(f"Supporting mechanism source `{source}` must not have role=background.")
            continue
        valid_mechanisms.append(item)
    if query_requests_mechanism(query) and not valid_mechanisms:
        errors.append("The question asks for mechanism differences; add source-bound supporting_mechanisms evidence.")
    for requirement in requirements.get("mechanism_requirements", []):
        if not isinstance(requirement, dict):
            continue
        source = str(requirement.get("source", "")).strip()
        anchors = [
            str(anchor).strip().lower()
            for anchor in requirement.get("anchors", [])
            if str(anchor).strip()
        ]
        candidates = [
            item for item in valid_mechanisms
            if isinstance(item, dict)
            and str(item.get("source", "")).strip() == source
        ]
        if source and anchors and not any(
            all(
                anchor in (
                    f"{item.get('claim', '')} {item.get('evidence', '')}".lower()
                )
                for anchor in anchors
            )
            for item in candidates
        ):
            errors.append(
                "Mechanism evidence from "
                f"`{source}` must preserve the source relation "
                f"`{' + '.join(anchors)}` in one source-bound item."
            )

    for source in requirements.get("review_sources", []):
        if "review/comparison" not in roles_by_source.get(source, ""):
            errors.append(f"Source `{source}` must retain role=review/comparison source.")
        if source not in review_entry_sources:
            errors.append(f"Review/comparison source `{source}` is missing from review_comparison_sources.")
    for source in route_sources - direct_route_sources:
        errors.append(f"Route source `{source}` is missing from direct_routes.")
    for source in review_sources - review_entry_sources:
        errors.append(f"Review/comparison source `{source}` is missing from review_comparison_sources.")
    for requirement in requirements.get("strategy_requirements", []):
        if not isinstance(requirement, dict):
            continue
        source = str(requirement.get("source", "")).strip()
        if source and source not in roles_by_source:
            errors.append(f"Strategy evidence source `{source}` is missing from source_roles.")
        elif source in background_sources:
            errors.append(f"Strategy evidence source `{source}` must not have role=background.")

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

    dimension_sources = requirements.get("dimension_sources", {})
    dimension_sources = dimension_sources if isinstance(dimension_sources, dict) else {}
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
            errors.append(f"`dimensions.{key}` must contain source-bound atomic evidence entries.")
        elif item.get("evidence_found") and not valid_atomic and (not item.get("text") or not item.get("sources")):
            errors.append(f"`dimensions.{key}` says evidence_found=true but lacks evidence.")
        expected_sources = set(dimension_sources.get(key, []))
        actual_sources = {str(entry.get("source", "")).strip() for entry in valid_atomic}
        missing_sources = sorted(expected_sources - actual_sources)
        if missing_sources:
            errors.append(
                f"`dimensions.{key}.evidence` omitted retrieved support from: "
                f"{', '.join(missing_sources)}."
            )
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

    for requirement in requirements.get("relation_requirements", []):
        if not isinstance(requirement, dict):
            continue
        key = str(requirement.get("dimension", "")).strip()
        source = str(requirement.get("source", "")).strip()
        anchors = {
            str(term) for term in requirement.get("anchors", [])
            if isinstance(term, str) and term
        }
        item = dimensions.get(key)
        evidence = item.get("evidence", []) if isinstance(item, dict) else []
        matched = any(
            isinstance(entry, dict)
            and str(entry.get("source", "")).strip() == source
            and anchors.intersection(
                exact_isotope_terms(str(entry.get("claim", "")), require_context=False)
            )
            and any(
                marker in str(entry.get("claim", "")).lower()
                for marker in _EVIDENCE_DIMENSION_MARKERS.get(key, ())
            )
            for entry in evidence
        )
        if anchors and source and not matched:
            errors.append(
                f"`dimensions.{key}.evidence` must connect an exact isotope identifier "
                f"({', '.join(sorted(anchors))}) to {key} in one source-bound atomic claim "
                f"from `{source}`; do not leave the related facts in separate dimensions."
            )

    for route in comparison.get("direct_routes", []) if isinstance(comparison.get("direct_routes"), list) else []:
        if not isinstance(route, dict):
            continue
        source = str(route.get("source", "")).strip()
        if source in review_sources:
            errors.append(f"Review/comparison source `{source}` must not appear in direct_routes.")
        if source in roles_by_source and source not in route_sources:
            errors.append(f"Direct route `{source}` must have role=route in source_roles.")
        if atomic_required and not str(route.get("route_phrase", "")).strip():
            errors.append(f"Direct route `{source}` must preserve its route-defining phrase.")
        if atomic_required and not str(route.get("outcome", "")).strip():
            errors.append(f"Direct route `{source}` must preserve its reported outcome.")
        if target and not direct_route_targets_query_target(route, target):
            errors.append(
                f"Direct strategy `{source}` describes target-mediated delivery or uptake "
                f"without an intervention acting on `{target}`; classify it as background."
            )
        if is_synthetic_route_query(query) and (
            source in background_sources or any(
                term in source.lower()
                for term in ("derivative", "formulation", "solubility", "biological propert")
            )
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
        errors.append("central_tradeoff must explicitly frame high-purity/isotopically enriched material.")
    if atomic_required and query_dims and (
        not isinstance(tradeoff_value, dict)
        or not tradeoff_value.get("claim")
        or not tradeoff_value.get("sources")
    ):
        errors.append("central_tradeoff must contain a source-bound claim.")
    return errors
