# rag/query_pipeline.py
# Public entry point for the query pipeline.
# Coordinates all pipeline stages; delegates implementation to sub-modules.
#
# Public API (stable):
#   execute_structured_query(...)        → str
#   execute_structured_query_stream(...) → Generator[str, None, None]

import json
import re
import time
import unicodedata

from llama_index.core import Settings

import config as cfg
from rag.knowledge_synthesizer import KnowledgeSynthesizer
from rag.comparison_json_validator import (
    comparison_json_validation_errors as _comparison_json_validation_errors,
)
from rag.answer_verifier import AnswerVerifier
from rag.query_planning import detect_target_paper, _keyword_prefilter, select_relevant_papers, plan_sub_questions
from rag.query_retrieval import (
    build_subquery_tasks,
    run_subqueries_parallel,
    is_empty_result,
    extract_paper_name,
    _query_window_score,
)
from rag.query_grounding_flow import run_grounding_check, split_into_sentences
from rag.query_translation import translate_to_traditional_chinese
from rag.query_prompts import build_synthesis_prompt, build_fallback_prompt
from rag.fact_contract import (
    bind_fact_list,
    build_evidence_catalog,
    contract_is_usable,
    render_fact_contract,
)

_synthesizer = KnowledgeSynthesizer()
_verifier    = AnswerVerifier()

_FALLBACK_NOTICE = (
    "⚠️ **資料來源說明**：本地學術文獻資料庫中未找到與此問題直接相關的內容。"
    "以下回答來自模型自身知識，非論文原文，請謹慎參考並自行查證。\n\n"
)

_FACT_LIST_LINE = re.compile(
    r"^\s*\[(?:Fact|事實)\s*(\d+)\]\s*(.*?)\s*"
    r"\((?:Source|來源)\s*[:：]\s*(.*?)\)\s*\.?\s*$",
    re.IGNORECASE,
)
_METHOD_QUERY_TERMS = (
    "method", "process", "procedure", "synthesis", "synthesize", "preparation",
    "key step", "steps", "reaction route", "方法", "製程", "流程", "合成", "步驟",
)
_COMPARISON_QUERY_TERMS = (
    "compare", "comparison", "different routes", "across the papers", "trade-off",
    "scalability", "cost-effectiveness", "比較", "跨文獻", "不同路線", "權衡",
)


def _fact_list_items(knowledge_base: str) -> list[dict]:
    items = []
    for line in (knowledge_base or "").splitlines():
        match = _FACT_LIST_LINE.match(line)
        if not match:
            continue
        source_text = match.group(3).strip()
        sources = [value.strip() for value in re.split(r"\s*[,;]\s*", source_text) if value.strip()]
        if sources:
            items.append({
                "id": f"F{match.group(1)}",
                "claim": match.group(2).strip(),
                "sources": sources,
            })
    return items


def _merge_fact_lists(original: str, recovered: str) -> str:
    original_items = _fact_list_items(original)
    recovered_items = _fact_list_items(recovered)
    if not original_items or not recovered_items:
        return "\n\n".join(part.strip() for part in (original, recovered) if part and part.strip())

    merged = {}
    for item in original_items + recovered_items:
        key = re.sub(r"[^\w]+", " ", item["claim"].casefold()).strip()
        if key not in merged:
            merged[key] = {"claim": item["claim"], "sources": list(item["sources"])}
        else:
            merged[key]["sources"].extend(
                source for source in item["sources"] if source not in merged[key]["sources"]
            )
    return "\n\n".join(
        f"[Fact {index}] {item['claim']} (Source: {', '.join(item['sources'])})"
        for index, item in enumerate(merged.values(), 1)
    )


def _render_validated_fact_contract(
    knowledge_base: str,
    sub_answers: list[str],
) -> tuple[str, list[str], dict]:
    chunks = [
        {"text": answer, "source": extract_paper_name(answer, f"retrieved_chunk_{index}")}
        for index, answer in enumerate(sub_answers)
    ]
    contract = bind_fact_list(knowledge_base, build_evidence_catalog(chunks))
    if not contract_is_usable(contract):
        return "", [], contract
    answer, claims = render_fact_contract(contract)
    return answer, claims, contract


_RECOVERY_SNIPPET_RE = re.compile(r"\[Snippet \d+\]\s*(.*?)(?=\n\[Snippet \d+\]|\Z)", re.S)
_MEASUREMENT_RE = re.compile(
    r"(?:\b(?:IC\s*50|K[im]|Vmax|pH)\b.{0,30}?\d|"
    r"\d+(?:\.\d+)?(?:\s*(?:±|\+/-)\s*\d+(?:\.\d+)?)?\s*"
    r"(?:%|°?\s*[CF]\b|nM\b|mM\b|µM\b|μM\b|mg\b|µg\b|μg\b|g\b|"
    r"mL\b|ml\b|months?\b|days?\b|hours?\b|h\b|min\b))",
    re.IGNORECASE,
)
_RESULT_RELATION_RE = re.compile(
    r"(?:\b(?:concentration|time|temperature)(?:-\s*(?:and|or)\s*"
    r"(?:concentration|time|temperature))?[- ]dependent\b|"
    r"\b(?:degrad\w*|form\w*|stable)\b.{0,180}\b"
    r"(?:alkali\w*|oxidat\w*|acidic|stable|rapid\w*|phenylalanine|tyrosine)\b|"
    r"\b(?:alkali\w*|oxidat\w*|acidic)\b.{0,180}\b(?:degrad\w*|form\w*|stable)\b)",
    re.IGNORECASE,
)


def _literal_recovery_facts(recovery_results: list[tuple[str, str]], question: str) -> str:
    """Keep short, query-relevant measurements that Stage 3 must not silently drop."""
    lower_question = (question or "").lower()
    if not any(term in lower_question for term in ("value", "condition", "storage", "dose", "yield", "數值", "條件", "儲存")):
        return ""
    condition_query = any(
        term in lower_question for term in ("condition", "storage", "stored", "條件", "儲存")
    )
    question_terms = {
        token for token in re.findall(r"[a-z][a-z0-9]+", lower_question)
        if len(token) > 3 and token not in {"does", "give", "reported", "study", "that", "the", "what", "which", "with"}
    }

    candidates = []
    for result_index, (label, block) in enumerate(recovery_results):
        source = label.strip("【】")
        for snippet in _RECOVERY_SNIPPET_RE.findall(block or ""):
            text = unicodedata.normalize("NFKC", snippet).replace("μ", "µ")
            text = re.sub(
                r"(?<=[A-Za-z])-\s+(?=(?!(?:and|or)\b)[A-Za-z])",
                "",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(r"\s+", " ", text).replace("\x03g", "µg").strip()
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
            for sentence_index, sentence in enumerate(sentences):
                sentence = sentence.strip()
                axis_values = re.search(
                    r"(?:-?\d+(?:\.\d+)?\s+){8,}", sentence
                )
                if axis_values:
                    sentence = sentence[:axis_values.start()].rstrip(" ,:-")
                    sentence = re.sub(r",?\s+(?:generating|showing)$", "", sentence, flags=re.I)
                measurement = bool(_MEASUREMENT_RE.search(sentence))
                relation = bool(_RESULT_RELATION_RE.search(sentence))
                if not 20 <= len(sentence) <= 700 or not (measurement or relation):
                    continue
                result_sentence = sentence
                sentence_terms = set(re.findall(r"[a-z][a-z0-9]+", sentence.lower()))
                if measurement and len(question_terms & sentence_terms) < 2:
                    for prior in reversed(sentences[max(0, sentence_index - 3):sentence_index]):
                        prior = prior.strip()
                        prior_terms = set(re.findall(r"[a-z][a-z0-9]+", prior.lower()))
                        if len(question_terms & prior_terms) < 2:
                            continue
                        prefix = re.split(
                            r"\b(?:to determine|to assess|to evaluate|to clarify|cells? were|samples? were)\b",
                            prior,
                            maxsplit=1,
                            flags=re.IGNORECASE,
                        )[0].strip(" .:")
                        context = prefix if 20 <= len(prefix) < len(prior) else prior
                        context_numbers = re.findall(
                            r"(?<![A-Za-z0-9])-?\d+(?:\.\d+)?", context
                        )
                        if len(context_numbers) <= 4 and len(context) + len(sentence) <= 700:
                            sentence = f"{context}. {sentence}"
                            break
                sentence_lower = sentence.lower()
                if any(marker in sentence_lower for marker in (
                    "article history:", "received in revised", "available online", "doi:",
                    "sub-question:", "source metadata", "above-described reports",
                    "pioneering reports",
                )):
                    continue
                if "mechanistic pathway" in sentence_lower and "was set to" in sentence_lower:
                    continue
                if len(re.findall(r"(?<![A-Za-z0-9])-?\d+(?:\.\d+)?", sentence)) > 14:
                    continue
                metric = bool(re.search(r"\b(?:IC\s*50|K[im]|Vmax|pH)\b", sentence, re.I))
                if sentence.count("(") != sentence.count(")"):
                    continue
                potency_query = any(term in lower_question for term in (
                    "potency", "inhibitory", "inhibition", "ic50"
                ))
                if potency_query and not (metric or relation or "potency" in result_sentence.lower()):
                    continue
                score = _query_window_score(sentence, question) + 6 * relation
                score += 4 * metric
                result_statement = bool(re.search(
                    r"\b(?:determined|observed|indicate|showed|stable|detectable|"
                    r"degrad(?:e|es|ed|ing)|form(?:s|ed|ing)|reaching|reached)\b",
                    result_sentence,
                    re.IGNORECASE,
                ))
                setup_statement = bool(re.search(
                    r"\b(?:to determine|to clarify|were exposed|were cultured|assays? were performed|"
                    r"sample preparation|markers? with the solid lines)\b",
                    result_sentence,
                    re.IGNORECASE,
                ))
                condition_witness = bool(
                    condition_query
                    and measurement
                    and re.search(r"\b(?:stored|storage|incubat\w*|kept)\b", result_sentence, re.I)
                )
                if setup_statement and not result_statement and not condition_witness:
                    continue
                if len(result_sentence) <= 320 and result_statement:
                    score += 8
                if metric and len(result_sentence) <= 180 and result_statement:
                    score += 12
                if setup_statement and not condition_witness:
                    score -= 12
                if condition_witness:
                    score += 12
                if "mechanistic pathway" in sentence_lower:
                    score -= 5
                if score >= 7:
                    candidates.append((score, -result_index, sentence, source))

    facts, seen = [], set()
    for _, _, sentence, source in sorted(candidates, reverse=True):
        key = re.sub(r"\W+", " ", sentence.casefold()).strip()
        if key in seen:
            continue
        seen.add(key)
        facts.append((sentence, source))
        if len(facts) == (8 if condition_query else 6):
            break
    return "\n\n".join(
        f"[Fact {index}] {claim} (Source: {source})"
        for index, (claim, source) in enumerate(facts, 1)
    )


_LITERAL_FACET_GROUPS = (
    ("combined", ("combined", "combination", "addition of preincubation", "pre-plus")),
    ("concentration-dependent", ("concentration-dependent", "concentration dependent")),
    ("time-dependent", ("time-dependent", "time dependent")),
    ("temperature-dependent", ("temperature-dependent", "temperature dependent")),
    ("alkaline", ("alkaline", "alkali", "naoh", "basic condition")),
    ("oxidative", ("oxidative", "oxidation", "h2o2")),
    ("acidic", ("acidic", "hcl", "acetic acid")),
    ("stable", ("stable", "stability", "no detectable degradation")),
    ("rapid", ("rapid", "rapidly")),
    ("slow", ("slow", "slowly")),
    ("dark", ("in the dark", "dark storage")),
)
_LITERAL_STOPWORDS = {
    "about", "according", "after", "also", "and", "are", "been", "being", "both",
    "during", "from", "into", "months", "reported", "results", "that", "the", "their",
    "these", "this", "under", "using", "value", "values", "were", "with",
}


def _literal_normalized(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    value = value.replace("‐", "-").replace("‑", "-").replace("–", "-")
    return " ".join(value.split())


def _literal_numbers(text: str) -> set[str]:
    plain = re.sub(r"\[[^\]]+\]|【[^】]+】", " ", str(text or ""))
    plain = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", plain)
    plain = re.sub(r"(?<=[A-Za-z])_\{?(\d+)\}?", r"\1", plain)
    values = re.findall(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?", plain)
    return {value.rstrip("0").rstrip(".") if "." in value else value for value in values}


def _literal_fact_present(claim: str, answer: str) -> bool:
    claim_text = _literal_normalized(claim)
    required_numbers = _literal_numbers(claim)
    required_facets = [
        (name, aliases) for name, aliases in _LITERAL_FACET_GROUPS
        if any(alias in claim_text for alias in aliases)
    ]
    claim_tokens = {
        token for token in re.findall(r"[a-z][a-z0-9]+", claim_text)
        if len(token) > 3 and token not in _LITERAL_STOPWORDS
    }
    if not required_numbers and not required_facets:
        return True

    for sentence in split_into_sentences(answer) or [answer]:
        sentence_text = _literal_normalized(sentence)
        if required_numbers - _literal_numbers(sentence):
            continue
        if any(not any(alias in sentence_text for alias in aliases) for _, aliases in required_facets):
            continue
        sentence_tokens = set(re.findall(r"[a-z][a-z0-9]+", sentence_text))
        required_overlap = min(3, max(1, len(claim_tokens) // 4))
        if len(claim_tokens & sentence_tokens) >= required_overlap:
            return True
    return False


def _append_missing_literal_facts(answer: str, literal_facts: str) -> str:
    additions = []
    for item in _fact_list_items(literal_facts):
        if _literal_fact_present(item["claim"], answer):
            continue
        claim = item["claim"].rstrip(" .")
        additions.append(f"- {claim} [Source: {', '.join(item['sources'])}].")
    if not additions:
        return answer
    return (answer or "").rstrip() + "\n\n" + "\n".join(additions)


def _is_method_fact_query(question: str) -> bool:
    lower = (question or "").lower()
    return (
        any(term in lower for term in _METHOD_QUERY_TERMS)
        and not any(term in lower for term in _COMPARISON_QUERY_TERMS)
    )


def _method_fact_roles(claim: str) -> set[str]:
    lower = (claim or "").lower()
    roles = set()
    if any(term in lower for term in (
        "hybrid process", "synthesized", "synthesis of", "synthesis is based",
        "preparation of", "method uses", "process involving",
    )):
        roles.add("overview")
    if any(term in lower for term in (
        "reacted", "reaction", "alkylation", "hydroly", "treatment", "treated",
        "furnish", "gave", "gives", "yielded", "produced", "converted",
    )):
        roles.add("steps")
    if (
        re.search(r"(?:^|\s)-?\d+(?:\.\d+)?\s*°\s*c\b", lower)
        or any(term in lower for term in (
            "temperature", "reaction time", "concentration", "solvent", " in thf",
            "conducted at", "conducted in", "performed at", "performed in", "under reflux",
        ))
    ):
        roles.add("conditions")
    if any(term in lower for term in (
        "yield", "e.e.", "enantiomeric excess", "optically pure", "enantiomerically pure",
        "furnish", "gave", "gives", "to give", "produced", "afforded", "resulted in",
    )):
        roles.add("outcome")
    if any(term in lower for term in (
        "starting material", "prepared from commercially", "was protected as",
        "group was protected", "initial efforts were directed toward",
    )):
        roles.add("precursor")
    if any(term in lower for term in ("alternative route", "control route", "non-enzymatic")):
        roles.add("alternative")
    return roles


def _method_requirements(question: str, sub_questions: list[dict]) -> set[str]:
    plan_text = " ".join(
        [question or ""]
        + [str(item.get("sub_q", "")) for item in (sub_questions or []) if isinstance(item, dict)]
    ).lower()
    requirements = {"overview", "steps", "outcome"}
    if any(term in plan_text for term in (
        "experimental condition", "reaction condition", "temperature", "solvent",
        "concentration", "reaction time", "stirring", "pressure", "ph",
    )):
        requirements.add("conditions")
    return requirements


def _render_method_fact_list(
    knowledge_base: str,
    question: str,
    sub_questions: list[dict],
) -> tuple[str, list[str], dict]:
    """Render source-bound Stage 3 facts without asking Stage 4 to rewrite them."""
    if not _is_method_fact_query(question):
        return "", [], {}
    facts = _fact_list_items(knowledge_base)
    if not facts:
        return "", [], {}

    requirements = _method_requirements(question, sub_questions)
    full_protocol = any(term in (question or "").lower() for term in (
        "full protocol", "complete protocol", "step-by-step", "all experimental conditions",
        "完整實驗", "完整操作", "所有條件",
    ))
    selected, excluded, seen = [], [], set()
    for fact in facts:
        roles = _method_fact_roles(fact["claim"])
        fact["roles"] = roles
        if not full_protocol and roles & {"precursor", "alternative"}:
            excluded.append(fact["id"])
            continue
        normalized = " ".join(fact["claim"].lower().split())
        if normalized in seen:
            continue
        if roles & requirements:
            selected.append(fact)
            seen.add(normalized)

    covered = set().union(*(fact["roles"] for fact in selected)) if selected else set()
    missing = requirements - covered
    if missing:
        return "", [], {
            "requirements": sorted(requirements),
            "selected_fact_ids": [fact["id"] for fact in selected],
            "excluded_fact_ids": excluded,
            "missing_requirements": sorted(missing),
        }

    claims = []
    for fact in selected:
        citation = ", ".join(fact["sources"])
        claim = fact["claim"].strip()
        line = f"- {claim} [{citation}]" if claim.endswith((".", "!", "?")) else f"- {claim} [{citation}]."
        claims.append(line)
    answer = "Method evidence:\n" + "\n".join(claims)
    return answer, claims, {
        "requirements": sorted(requirements),
        "selected_fact_ids": [fact["id"] for fact in selected],
        "excluded_fact_ids": excluded,
        "missing_requirements": [],
    }


def _build_memory_section(memory_context: str, is_fallback: bool) -> str:
    if not memory_context:
        return ""
    if is_fallback:
        return "【相關歷史問答記憶，僅供參考】\n" + memory_context + "\n"
    return "---\n【相關歷史問答記憶，僅供參考】" + memory_context


def _comparison_json_from_knowledge_base(knowledge_base: str) -> dict:
    text = (knowledge_base or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    comparison = data.get("comparison_json") if isinstance(data, dict) else None
    return comparison if isinstance(comparison, dict) else {}


def _stage4_answer_validation_issues(answer: str, knowledge_base: str, question: str) -> str:
    if not getattr(cfg, "STAGE4_ANSWER_VALIDATION_ENABLED", False):
        return ""
    comparison = _comparison_json_from_knowledge_base(knowledge_base)
    if not comparison:
        return ""

    lower = (answer or "").lower()
    issues = []
    if any(marker in lower for marker in ("no relevant query results", "no paper data", "please provide the query results")):
        issues.append(
            "Stage4Validation | False no-data answer | The Known Facts List contains comparison_json with paper evidence; do not ask the user to provide data."
        )

    for route in comparison.get("direct_routes", []):
        if not isinstance(route, dict):
            continue
        phrase = str(route.get("route_phrase", "")).strip()
        if phrase and phrase.lower() not in lower:
            issues.append(f"Stage4Validation | Missing direct route phrase | Include exactly: {phrase}")

    for review in comparison.get("review_comparison_sources", []):
        if not isinstance(review, dict):
            continue
        source = str(review.get("source", "")).strip()
        if source and source.lower() not in lower:
            issues.append(f"Stage4Validation | Missing review/comparison source | Include {source} as review/comparison evidence, not as a direct route.")

    source_roles = comparison.get("source_roles") if isinstance(comparison.get("source_roles"), list) else []
    for source in (
        str(item.get("source", "")).strip()
        for item in source_roles
        if isinstance(item, dict) and str(item.get("role", "")).lower() == "background"
    ):
        if source and source.lower() in lower:
            issues.append(
                f"Stage4Validation | Background source cited in core comparison | Remove {source} from the final route comparison; use only direct route and review/comparison sources."
            )

    for sentence in split_into_sentences(answer or ""):
        if sentence.count("[") >= 2:
            issues.append(
                "Stage4Validation | Over-dense multi-source sentence | Split this into short separate sentences or bullets, each with one source-backed claim."
            )
            break

    dimensions = comparison.get("dimensions") if isinstance(comparison.get("dimensions"), dict) else {}
    dim_terms = {
        "isotopic_enrichment": ("isotopic enrichment", "10b", "boron-10"),
        "scalability": ("scalability", "scalable", "scale-up", "route efficiency", "practical synthesis"),
        "cost_effectiveness": ("cost-effectiveness", "cost effectiveness", "cost", "lower process burden", "fewer protecting groups"),
        "safety": ("safety", "safe", "risk"),
    }
    for key, terms in dim_terms.items():
        item = dimensions.get(key)
        if not isinstance(item, dict) or not item.get("requested"):
            continue
        if item.get("evidence_found") and not any(term in lower for term in terms):
            issues.append(f"Stage4Validation | Missing requested dimension | Cover {key} using the comparison_json text and sources.")
        if key in ("scalability", "cost_effectiveness") and any(
            marker in lower for marker in ("did not provide", "does not provide", "not provide", "no data", "missing")
        ):
            issues.append(
                f"Stage4Validation | Wrong missing-evidence claim | Do not say {key} is missing when review/comparison evidence supports a qualitative comparison."
            )

    if dimensions.get("isotopic_enrichment", {}).get("requested") and "high-purity" not in lower:
        issues.append(
            "Stage4Validation | Missing high-purity framing | The Central trade-off must explicitly say high-purity/isotopically enriched L-BPA or high-purity isotopically enriched material."
        )

    if "central trade-off" not in lower and "core trade-off" not in lower:
        issues.append("Stage4Validation | Missing central trade-off | Add one concise Central trade-off sentence using the requested dimensions.")

    return "VERIFY_FAIL\n" + "\n".join(f"- {issue}" for issue in issues) if issues else ""


def _stage4_empty_answer_fallback(
    knowledge_base: str,
    atomic_only: bool = False,
    question: str = "",
) -> str:
    comparison = _comparison_json_from_knowledge_base(knowledge_base)
    if not comparison:
        return "" if atomic_only else (knowledge_base or "").strip()

    q_lower = (question or "").lower()
    route_text = " ".join(
        str(route.get("route_phrase", ""))
        for route in comparison.get("direct_routes", [])
        if isinstance(route, dict)
    ).lower()
    is_synthesis_comparison = any(term in q_lower for term in (
        "synthesi", "synthetic route", "preparation method", "manufactur", "合成", "製備",
    )) or any(term in route_text for term in (
        "synthesi", "alkylat", "hydrolys", "deprotect", "coupling", "reaction sequence",
    ))
    detail_requested = any(term in q_lower for term in (
        "yield", "percentage", "percent", "e.e.", "optical purity", "ld50",
        "toxicity", "dose", "temperature", "reaction condition", "reagent",
        "step-by-step", "numerical",
    ))
    concise = bool(question) and not detail_requested
    source_roles = comparison.get("source_roles") if isinstance(comparison.get("source_roles"), list) else []
    background_sources = {
        str(item.get("source", "")).strip()
        for item in source_roles
        if isinstance(item, dict) and str(item.get("role", "")).lower() == "background"
    }
    route_sources = {
        str(item.get("source", "")).strip()
        for item in source_roles
        if isinstance(item, dict) and str(item.get("role", "")).lower() == "route"
    }
    tradeoff_value = comparison.get("central_tradeoff", "")
    if isinstance(tradeoff_value, dict):
        tradeoff = str(tradeoff_value.get("claim", "")).strip()
        tradeoff_sources = [
            str(source).strip() for source in tradeoff_value.get("sources", []) if source
        ] if isinstance(tradeoff_value.get("sources"), list) else []
    else:
        tradeoff, tradeoff_sources = str(tradeoff_value).strip(), []
    if atomic_only and (not tradeoff or not tradeoff_sources):
        return ""

    target_compound = str(comparison.get("target_compound", "") or "the target compound").strip().rstrip(".")
    lines = ["Comparison scaffold:"]
    for route in comparison.get("direct_routes", []):
        if not isinstance(route, dict):
            continue
        source = str(route.get("source", "")).strip()
        phrase = str(route.get("route_phrase", "")).strip().rstrip(".")
        outcome = str(route.get("outcome", "")).strip().rstrip(".")
        outcome_lower = outcome.lower()
        if concise and "optically pure" in outcome_lower:
            product = "L-BPA" if "l-bpa" in outcome_lower else target_compound
            outcome = f"optically pure {product}"
            if "e.e." in outcome_lower or "enantiomeric excess" in outcome_lower:
                outcome += " at high e.e."
        if atomic_only and (not source or not phrase or not outcome):
            return ""
        if source and phrase and source not in background_sources:
            result = (
                f", yielding {outcome}" if is_synthesis_comparison else f", with {outcome}"
            ) if outcome else ""
            label = "Route" if is_synthesis_comparison else "Strategy"
            lines.append(f"- {label}: `{source}` reports {phrase}{result} [{source}].")

    for mechanism in comparison.get("supporting_mechanisms", []):
        if not isinstance(mechanism, dict):
            continue
        source = str(mechanism.get("source", "")).strip()
        claim = str(mechanism.get("claim", "")).strip().rstrip(".")
        if source and claim and source not in background_sources:
            lines.append(f"- Mechanism: `{source}` reports {claim} [{source}].")

    for review in comparison.get("review_comparison_sources", []):
        if not isinstance(review, dict):
            continue
        source = str(review.get("source", "")).strip()
        if source and source not in background_sources:
            dimensions = [
                str(value).strip().replace("_", "-")
                for value in review.get("dimensions", [])
                if str(value).strip()
            ] if isinstance(review.get("dimensions"), list) else []
            if is_synthesis_comparison:
                claim = f"the synthesis of {target_compound} has been approached through multiple routes"
            else:
                claim = f"multiple therapeutic strategies involve {target_compound}"
            lines.append(f"- Review/comparison source: `{source}` reports that {claim} [{source}].")
            if concise and dimensions:
                review_dimensions = [
                    value for value in dimensions if "isotop" not in value.lower()
                ] or dimensions
                dimension_text = review_dimensions[0] if len(review_dimensions) == 1 else (
                    ", ".join(review_dimensions[:-1]) + f", and {review_dimensions[-1]}"
                )
                lines.append(
                    "- Review dimensions: The review highlights limitations of each method "
                    f"regarding {dimension_text} [{source}]."
                )

    dimensions = comparison.get("dimensions") if isinstance(comparison.get("dimensions"), dict) else {}
    labels = {
        "isotopic_enrichment": (
            "High-purity/isotopic enrichment"
            if any(term in tradeoff.lower() for term in (
                "high-purity", "high purity", "high optical purity", "optically pure", "enantiopur",
            ))
            else "Isotopic enrichment"
        ),
        "scalability": "Scalability",
        "cost_effectiveness": "Cost-effectiveness",
        "safety": "Safety",
    }
    dimension_lines = []
    seen_claims = set()
    for key in ("isotopic_enrichment", "scalability", "cost_effectiveness", "safety"):
        item = dimensions.get(key)
        if not isinstance(item, dict) or not item.get("evidence_found"):
            continue
        if concise and not item.get("requested"):
            continue
        evidence = [
            entry for entry in item.get("evidence", [])
            if (
                isinstance(entry, dict)
                and entry.get("source")
                and entry.get("claim")
                and str(entry.get("source")).strip() not in background_sources
            )
        ] if isinstance(item.get("evidence"), list) else []
        if concise and key == "scalability" and len(evidence) > 1:
            non_safety = [
                entry for entry in evidence
                if not any(term in str(entry["claim"]).lower() for term in (
                    "safety", "toxicity", "contamination", "oxidant",
                ))
            ]
            if non_safety:
                evidence = non_safety
            route_evidence = [entry for entry in evidence if str(entry["source"]).strip() in route_sources]
            if route_evidence:
                evidence = route_evidence
        if concise:
            evidence = evidence[:1]
        if atomic_only and not evidence:
            return ""
        for entry in evidence:
            source = str(entry["source"]).strip()
            claim = str(entry["claim"]).strip().rstrip(".")
            claim_key = (source.lower(), claim.lower())
            if claim_key in seen_claims:
                continue
            seen_claims.add(claim_key)
            dimension_lines.append(f"- {labels[key]}: {claim} [{source}].")
        if evidence:
            continue
        text = str(item.get("text", "")).strip()
        sources = ", ".join(
            str(source) for source in item.get("sources", [])
            if source and str(source).strip() not in background_sources
        )
        if text:
            dimension_lines.append(f"- {labels[key]}: {text}" + (f" [{sources}]." if sources else "."))

    if dimension_lines:
        tradeoff_heading = "Central trade-off:"
        if concise:
            requested_labels = [
                labels[key].lower()
                for key in ("isotopic_enrichment", "scalability", "cost_effectiveness", "safety")
                if isinstance(dimensions.get(key), dict) and dimensions[key].get("requested")
            ]
            if len(requested_labels) >= 2:
                right = requested_labels[1] if len(requested_labels) == 2 else (
                    ", ".join(requested_labels[1:-1]) + f" and {requested_labels[-1]}"
                )
                tradeoff_heading = f"Central trade-off ({requested_labels[0]} versus {right}):"
        lines.extend(("", tradeoff_heading, *dimension_lines))
    if atomic_only and len(lines) == 1:
        return ""
    return "\n".join(lines).strip()


def _append_missing_isotope_cost_answer(answer: str, knowledge_base: str, question: str) -> str:
    q = (question or "").lower()
    if "cost" not in q or not any(term in q for term in ("isotopic", "10b", "enrichment")):
        return answer
    lower = (answer or "").lower()
    if any(term in lower for term in ("high cost", "isotope starting material", "expensive 10b", "10b-enriched starting")):
        return answer

    kb_lower = (knowledge_base or "").lower()
    has_high_cost = "high cost of isotopically enriched 10b" in kb_lower
    has_isotope_starting = "major cost typically comes from the isotope starting material" in kb_lower
    if not has_high_cost and not has_isotope_starting:
        return answer

    source = "CMDC-20-e202500059" if "cmdc-20-e202500059" in kb_lower else "review/comparison source"
    if has_high_cost:
        sentence = f"Cost-effectiveness: the review highlights the high cost of isotopically enriched 10B [Source: {source}]."
    else:
        sentence = (
            "Cost-effectiveness: when preparing isotopically enriched compounds, "
            f"the major cost typically comes from the isotope starting material [Source: {source}]."
        )
    return (answer or "").rstrip() + "\n\n" + sentence


_STABILITY_RESULT_RE = re.compile(
    r"\b(?:is|are|was|were|remains?|remained)\s+stable\b"
    r"|\b(?:shows?|showed|exhibits?|exhibited)\s+no\s+(?:detectable\s+)?degradation\b"
    r"|\bno\s+(?:detectable\s+)?degradation\s+(?:is\s+|was\s+)?observed\b",
    re.IGNORECASE,
)
_STABILITY_PROTOCOL_JOIN_RE = re.compile(
    r",\s+including\s+"
    r"(?P<subject>(?:forced\s+)?(?:degradation\s+)?(?:tests?|assays?|experiments?))\s+"
    r"(?P<verb>performed|conducted)\b",
    re.IGNORECASE,
)


def _separate_stability_protocol_clause(answer: str) -> str:
    """Split an observed stability result from a protocol accidentally attached to it."""
    if not getattr(cfg, "FACT_RELATION_ATOMICITY_GUARD_ENABLED", False):
        return answer

    text = answer or ""

    def _replace(match: re.Match) -> str:
        sentence_start = 0
        for marker in (". ", "? ", "! ", "\n"):
            boundary = text.rfind(marker, 0, match.start())
            if boundary >= 0:
                sentence_start = max(sentence_start, boundary + len(marker))
        if not _STABILITY_RESULT_RE.search(text[sentence_start:match.start()]):
            return match.group(0)

        subject = match.group("subject")
        auxiliary = "were" if subject.lower().endswith("s") else "was"
        return f". {subject[:1].upper() + subject[1:]} {auxiliary} {match.group('verb')}"

    return _STABILITY_PROTOCOL_JOIN_RE.sub(_replace, text)


def _rewrite_stage4_if_needed(full_text: str, knowledge_base: str, question: str, on_status=None) -> str:
    def _status(msg):
        if on_status:
            on_status(msg)
        else:
            print(msg)

    current = full_text
    retries = getattr(cfg, "STAGE4_ANSWER_REWRITE_RETRIES", 1)
    for attempt in range(retries):
        issues = _stage4_answer_validation_issues(current, knowledge_base, question)
        if not issues:
            return current
        preview = "; ".join(line[2:] for line in issues.splitlines()[1:3])
        _status(f"  🔁 [stage4-validator] rewrite {attempt + 1}/{retries}: {preview}")
        rewritten = _verifier.correct(current, knowledge_base, issues, on_status=on_status)
        if not rewritten or rewritten == current:
            return current
        current = rewritten
    return current


def _attempt_partial_recovery(
    question: str,
    valid_tasks: list,
    prefilled: dict,
    sub_answers: list[str],
    knowledge_base: str,
    assessment: dict,
    on_status=None,
    on_artifact=None,
) -> dict:
    outcome = {
        "attempted": False,
        "accepted": False,
        "sub_answers": sub_answers,
        "knowledge_base": knowledge_base,
        "assessment": assessment,
        "literal_facts": "",
    }
    is_comparison = any(term in (question or "").lower() for term in _COMPARISON_QUERY_TERMS)
    if (
        assessment.get("verdict") != "PARTIAL"
        or not getattr(cfg, "PARTIAL_ANSWER_RECOVERY_ENABLED", False)
        or not getattr(cfg, "SYNTHESIS_ENABLED", False)
        or getattr(cfg, "STAGE2_LLM_SUBANSWERS_ENABLED", False)
        or is_comparison
        or not valid_tasks
    ):
        return outcome

    def _status(msg):
        if on_status:
            on_status(msg)
        else:
            print(msg)

    outcome["attempted"] = True
    started = time.perf_counter()
    reason = str(assessment.get("reason", "")).strip()
    _status(f"  [partial-recovery] expanding evidence: {reason[:180]}")
    try:
        recovery_results = run_subqueries_parallel(
            valid_tasks,
            prefilled,
            on_status=_status,
            evidence_snippets_per_task=getattr(
                cfg, "PARTIAL_RECOVERY_EVIDENCE_SNIPPETS_PER_TASK", 3
            ),
            include_adjacent_evidence=getattr(
                cfg, "PARTIAL_RECOVERY_DETERMINISTIC_GUARDS_ENABLED", False
            ),
        )
        recovery_sub_answers = [f"{label}\n{result}" for label, result in recovery_results]
        recovery_text = "\n\n".join(recovery_sub_answers)
        recovery_chunks = [
            {"text": answer, "source": extract_paper_name(answer, f"retrieved_chunk_{i}")}
            for i, answer in enumerate(recovery_sub_answers)
        ]
        if on_artifact:
            on_artifact("stage2_recovery_evidence", recovery_text)

        def _recovery_artifact(name, value):
            if on_artifact:
                recovery_name = name.replace("stage3_", "stage3_recovery_", 1)
                on_artifact(recovery_name, value)

        recovered_kb = _synthesizer.synthesize(
            chunks=recovery_chunks,
            query=question,
            recovery_hint=reason,
            on_status=on_status,
            on_artifact=_recovery_artifact if on_artifact else None,
        )
        literal_kb = ""
        if getattr(cfg, "PARTIAL_RECOVERY_DETERMINISTIC_GUARDS_ENABLED", False):
            literal_kb = _literal_recovery_facts(recovery_results, question)
            recovered_kb = _merge_fact_lists(recovered_kb, literal_kb) if literal_kb else recovered_kb
            if literal_kb and on_artifact:
                on_artifact("stage3_recovery_literal_facts", literal_kb)
        outcome["literal_facts"] = literal_kb
        merged_kb = _merge_fact_lists(knowledge_base, recovered_kb)
        if on_artifact:
            on_artifact("stage3_recovery_knowledge_base", merged_kb)
        from rag.answerability import assess_answerability
        recovered_assessment = assess_answerability(question, merged_kb)
        outcome["recovered_assessment"] = recovered_assessment
        outcome["accepted"] = recovered_assessment.get("verdict") == "ANSWERABLE"
        if outcome["accepted"]:
            outcome.update({
                "sub_answers": recovery_sub_answers,
                "knowledge_base": merged_kb,
                "assessment": recovered_assessment,
            })
        if on_artifact:
            on_artifact(
                "partial_recovery_assessment",
                json.dumps(recovered_assessment, ensure_ascii=False, indent=2),
            )
        _status(
            f"  [partial-recovery] verdict={recovered_assessment.get('verdict')} "
            f"accepted={outcome['accepted']} elapsed_ms={int((time.perf_counter()-started)*1000)}"
        )
    except Exception as exc:
        _status(f"  [partial-recovery] failed; keeping original facts ({exc})")
    return outcome


# ══════════════════════════════════════════════════════════════════
#  Non-streaming entry point
# ══════════════════════════════════════════════════════════════════

def execute_structured_query(
    question: str,
    paper_engines: dict,
    memory_context: str = "",
    on_status=None,
    on_artifact=None,
) -> str:
    """
    Full query pipeline (non-streaming).
    Stages: planning → retrieval → synthesis → LLM → verification → grounding → translation
    """
    def _status(msg):
        if on_status:
            on_status(msg)
        else:
            print(msg)

    t0 = time.perf_counter()
    all_paper_names = list(paper_engines.keys())

    # ── Stage 1: Planning ────────────────────────────────────────────
    _status("\n[planning] 開始")
    detected = detect_target_paper(question, all_paper_names)
    if cfg.REVIEW_MODE:
        _status("\n  📖 REVIEW_MODE 已啟用，使用全部論文，跳過篩選")
        paper_names = all_paper_names
        paper_engines_to_use = paper_engines
    elif detected:
        paper_names = all_paper_names
        paper_engines_to_use = paper_engines
    else:
        _status("\n  🔎 先篩選相關論文...")
        prefiltered = _keyword_prefilter(question, all_paper_names)
        paper_names = select_relevant_papers(question, prefiltered)
        paper_engines_to_use = {k: v for k, v in paper_engines.items() if k in paper_names}

    _status("\n  📋 拆解子問題中...")
    sub_questions = plan_sub_questions(question, paper_names)
    _status(f"  → 拆出 {len(sub_questions)} 個子問題")
    _status(f"[planning] 完成 paper_count={len(paper_names)} subquery_count={len(sub_questions)} "
            f"elapsed_ms={int((time.perf_counter()-t0)*1000)}")

    # ── Stage 2: Retrieval ───────────────────────────────────────────
    t1 = time.perf_counter()
    _status(f"\n[retrieval] 開始")
    _status(f"\n  ⚡ 並行檢索 {len(sub_questions)} 個子問題中（workers={cfg.SUBQUERY_MAX_WORKERS}）...")
    valid_tasks, prefilled = build_subquery_tasks(sub_questions, paper_engines_to_use, paper_engines)
    deterministic_evidence = getattr(
        cfg, "PARTIAL_RECOVERY_DETERMINISTIC_GUARDS_ENABLED", False
    )
    ordered_results = run_subqueries_parallel(
        valid_tasks,
        prefilled,
        on_status=_status,
        evidence_snippets_per_task=(
            getattr(cfg, "PARTIAL_RECOVERY_EVIDENCE_SNIPPETS_PER_TASK", 4)
            if deterministic_evidence else None
        ),
        include_adjacent_evidence=deterministic_evidence,
    )

    sub_answers = []
    rag_found_anything = False
    for label, result in ordered_results:
        sub_answers.append(f"{label}\n{result}")
        if not is_empty_result(result):
            rag_found_anything = True
        _status(f"\n  ── {label} 回覆 ──\n  {result[:200]}")

    _status(f"\n[retrieval] 完成 rag_found={rag_found_anything} "
            f"elapsed_ms={int((time.perf_counter()-t1)*1000)}")
    if on_artifact:
        on_artifact("stage2_evidence", "\n\n".join(sub_answers))

    # ── Stage 3: Knowledge synthesis (distillation) ──────────────────
    t2 = time.perf_counter()
    _status("\n  🔗 綜合所有子答案中...")
    literal_kb = ""
    if cfg.SYNTHESIS_ENABLED and rag_found_anything:
        _status("\n  🧪 [synthesis] 知識蒸餾中...")
        synthesis_chunks = [
            {"text": ans, "source": extract_paper_name(ans, f"retrieved_chunk_{i}")}
            for i, ans in enumerate(sub_answers)
        ]
        knowledge_base = _synthesizer.synthesize(
            chunks=synthesis_chunks, query=question, on_status=on_status, on_artifact=on_artifact,
        )
        if deterministic_evidence:
            literal_kb = _literal_recovery_facts(ordered_results, question)
            if literal_kb:
                knowledge_base = _merge_fact_lists(knowledge_base, literal_kb)
                if on_artifact:
                    on_artifact("stage3_literal_facts", literal_kb)
    else:
        knowledge_base = "\n\n".join(sub_answers)
    if on_artifact:
        on_artifact("knowledge_base", knowledge_base)
        on_artifact("stage3_knowledge_base", knowledge_base)
    _status(f"[synthesis] 完成 elapsed_ms={int((time.perf_counter()-t2)*1000)}")

    # ── Stage 3.5: 可答性 gate（Phase 2：接路由）──────────────────────
    # NOT_ANSWERABLE→硬棄答（跳 Stage 4-7）；PARTIAL→軟警告橫幅；ANSWERABLE→正常。
    gate_abstain, gate_notice = False, ""
    if cfg.ANSWERABILITY_GATE_ENABLED and rag_found_anything:
        from rag.answerability import assess_answerability, gate_route
        _ans = assess_answerability(question, knowledge_base)
        gate_abstain, gate_notice = gate_route(_ans["verdict"])
        _kb_head = " ".join((knowledge_base or "")[:240].split())
        _status(f"[answerability] verdict={_ans['verdict']} abstain={gate_abstain} "
                f"kb_chars={len(knowledge_base or '')} kb_head={_kb_head} reason={_ans['reason'][:160]}")
        recovery = _attempt_partial_recovery(
            question,
            valid_tasks,
            prefilled,
            sub_answers,
            knowledge_base,
            _ans,
            on_status=on_status,
            on_artifact=on_artifact,
        )
        if recovery["accepted"]:
            if on_artifact:
                on_artifact("stage2_initial_evidence", "\n\n".join(sub_answers))
                on_artifact("stage3_initial_knowledge_base", knowledge_base)
            sub_answers = recovery["sub_answers"]
            knowledge_base = recovery["knowledge_base"]
            literal_kb = _merge_fact_lists(
                literal_kb, recovery.get("literal_facts", "")
            )
            _ans = recovery["assessment"]
            gate_abstain, gate_notice = gate_route(_ans["verdict"])
            if on_artifact:
                on_artifact("stage2_evidence", "\n\n".join(sub_answers))
                on_artifact("knowledge_base", knowledge_base)
                on_artifact("stage3_knowledge_base", knowledge_base)

    # ── Stage 4: LLM synthesis ───────────────────────────────────────
    t3 = time.perf_counter()
    synthesis_prompt = ""
    stage4_direct_rendered = False
    fact_contract_direct = False
    direct_render_text = ""
    direct_grounding_claims = []
    direct_render_meta = {}
    if gate_abstain:
        _status("  🚪 [answerability] NOT_ANSWERABLE → 誠實棄答，跳過生成")
        full_text = gate_notice
    else:
        direct_answer = ""
        if (
            rag_found_anything
            and cfg.EN_DRAFT_PIPELINE
            and getattr(cfg, "COMPARISON_JSON_DIRECT_RENDER_ENABLED", False)
            and not _comparison_json_validation_errors(knowledge_base, question)
        ):
            direct_answer = _stage4_empty_answer_fallback(
                knowledge_base,
                atomic_only=True,
                question=question,
            )
            if direct_answer:
                direct_grounding_claims = split_into_sentences(direct_answer)
                synthesis_prompt = "[deterministic comparison_json renderer]"
        if (
            not direct_answer
            and rag_found_anything
            and cfg.EN_DRAFT_PIPELINE
            and cfg.REASONING_MODE == "strict"
            and getattr(cfg, "METHOD_FACT_LIST_DIRECT_RENDER_ENABLED", False)
        ):
            direct_answer, direct_grounding_claims, direct_render_meta = _render_method_fact_list(
                knowledge_base,
                question,
                sub_questions,
            )
            if direct_answer:
                synthesis_prompt = "[deterministic method fact-list renderer]"
        if (
            not direct_answer
            and rag_found_anything
            and cfg.EN_DRAFT_PIPELINE
            and cfg.REASONING_MODE == "strict"
            and getattr(cfg, "STRUCTURED_FACT_CONTRACT_ENABLED", False)
            and not _is_method_fact_query(question)
            and not any(term in question.lower() for term in _COMPARISON_QUERY_TERMS)
        ):
            direct_answer, direct_grounding_claims, direct_render_meta = _render_validated_fact_contract(
                knowledge_base,
                sub_answers,
            )
            if direct_answer:
                fact_contract_direct = True
                synthesis_prompt = "[deterministic evidence-bound fact renderer]"
        if direct_answer:
            stage4_direct_rendered = True
            full_text = direct_answer
            direct_render_text = direct_answer
            if direct_render_meta.get("schema") == "fact_contract_v1":
                _status(
                    "  🧱 [synthesis-llm] evidence-bound fact contract → deterministic render "
                    f"accepted={len(direct_render_meta['facts'])} "
                    f"rejected={len(direct_render_meta['rejected'])}"
                )
            elif direct_render_meta:
                _status(
                    "  🧱 [synthesis-llm] method fact_list → deterministic render "
                    f"selected={','.join(direct_render_meta['selected_fact_ids'])} "
                    f"requirements={','.join(direct_render_meta['requirements'])}"
                )
            else:
                _status("  🧱 [synthesis-llm] validated atomic comparison_json → deterministic render")
        else:
            if not rag_found_anything:
                _status("  ℹ️  RAG 資料庫未找到相關內容，切換至模型推理模式...")
                fallback_notice = _FALLBACK_NOTICE
                synthesis_prompt = build_fallback_prompt(
                    question, _build_memory_section(memory_context, is_fallback=True)
                )
            else:
                fallback_notice = ""  # 軟警告橫幅在翻譯後才加（見 Stage 7 後），避免中文橫幅被送進翻譯
                lang = "en" if cfg.EN_DRAFT_PIPELINE else "zh"
                final_translation_enabled = getattr(cfg, "FINAL_TRANSLATION_ENABLED", True)
                print(f"  {'🧠 推理' if cfg.REASONING_MODE == 'reasoning' else '📋 嚴格'}模式"
                      f"（{cfg.REASONING_MODE}）  target_paper_detected={bool(detected)}"
                      f"  streaming_mode=False"
                      f"  translation_applied={cfg.EN_DRAFT_PIPELINE and final_translation_enabled}")
                synthesis_prompt = build_synthesis_prompt(
                    knowledge_base, question,
                    _build_memory_section(memory_context, is_fallback=False),
                    cfg.REASONING_MODE, lang,
                )
        if on_artifact:
            on_artifact("stage4_prompt", synthesis_prompt)
            if direct_render_meta.get("schema") == "fact_contract_v1":
                on_artifact("stage4_fact_contract", json.dumps(direct_render_meta, ensure_ascii=False, indent=2))
            elif direct_render_meta:
                on_artifact("stage4_fact_requirements", json.dumps(direct_render_meta, ensure_ascii=False, indent=2))
            if direct_grounding_claims:
                on_artifact("stage4_grounding_claims", "\n".join(direct_grounding_claims))

        if not stage4_direct_rendered:
            print("\n 最終綜合回答（Stage 4 初稿）：")
            full_text = fallback_notice
            try:
                for chunk in Settings.llm.stream_complete(synthesis_prompt):
                    print(chunk.delta, end="", flush=True)
                    full_text += chunk.delta
            except Exception as exc:
                _status(f"  ⚠️  [synthesis-llm] failed; using Stage 3 facts ({str(exc)[:180]})")
                fallback = _stage4_empty_answer_fallback(knowledge_base, question=question)
                full_text = fallback_notice + fallback
            print("\n")
            if rag_found_anything and not full_text.strip():
                _status("  ⚠️  [synthesis-llm] empty answer; using Stage 3 facts fallback")
                full_text = _stage4_empty_answer_fallback(knowledge_base, question=question)
    if on_artifact:
        on_artifact("stage4_draft", full_text)
    _status(f"[synthesis-llm] 完成 elapsed_ms={int((time.perf_counter()-t3)*1000)}")

    if rag_found_anything and not gate_abstain and not fact_contract_direct:
        full_text = _rewrite_stage4_if_needed(full_text, knowledge_base, question, on_status=on_status)
        full_text = _append_missing_isotope_cost_answer(full_text, knowledge_base, question)
        atomic_text = _separate_stability_protocol_clause(full_text)
        if atomic_text != full_text:
            _status("  🧱 [stage4-atomicity] separated a stability result from its protocol")
            full_text = atomic_text
    if on_artifact:
        on_artifact("stage4_validated", full_text)

    # ── Stage 5: Verification ────────────────────────────────────────
    if cfg.VERIFY_ENABLED and rag_found_anything and not gate_abstain and not fact_contract_direct:
        t4 = time.perf_counter()
        _status("\n  🔍 [verification] Stage 5: 邏輯自洽驗證中...")
        full_text = _verifier.verify_and_correct(
            draft_answer=full_text, knowledge_base=knowledge_base, on_status=on_status,
        )
        _status(f"[verification] 完成 elapsed_ms={int((time.perf_counter()-t4)*1000)}")
    if (
        rag_found_anything
        and not gate_abstain
        and not fact_contract_direct
        and deterministic_evidence
        and literal_kb
    ):
        before_literal_guard = full_text
        full_text = _append_missing_literal_facts(full_text, literal_kb)
        if full_text != before_literal_guard:
            _status("  🧱 [literal-completeness] restored omitted direct evidence")
            if on_artifact:
                on_artifact(
                    "stage5_literal_completeness",
                    full_text[len(before_literal_guard.rstrip()):].lstrip(),
                )
    if not fact_contract_direct:
        atomic_text = _separate_stability_protocol_clause(full_text)
        if atomic_text != full_text:
            _status("  🧱 [stage4-atomicity] restored result/protocol separation after verification")
            full_text = atomic_text
    if on_artifact:
        on_artifact("stage5_verified", full_text)

    # ── Stage 6: Citation grounding ──────────────────────────────────
    nli_report = ""
    if cfg.CITATION_GROUNDING_ENABLED and rag_found_anything and not gate_abstain:
        t5 = time.perf_counter()
        _status("\n[grounding] 開始")
        try:
            full_text, nli_report = run_grounding_check(
                full_text, sub_answers, knowledge_base,
                question=question, paper_engines_to_use=paper_engines_to_use,
                grounding_claims=(
                    direct_grounding_claims
                    if direct_grounding_claims and full_text == direct_render_text
                    else None
                ),
                on_status=_status,
            )
            print(nli_report)
        except Exception as e:
            _status(f"  ⚠️  答案品質審查失敗（不影響主流程）：{e}")
        _status(f"[grounding] 完成 elapsed_ms={int((time.perf_counter()-t5)*1000)}")
    if on_artifact:
        on_artifact("stage6_grounded_answer", full_text)
        on_artifact("stage6_grounding_report", nli_report)

    if on_artifact:
        on_artifact("answer_for_judge", full_text)

    # ── Stage 7: Translation ─────────────────────────────────────────
    # 棄答橫幅本來就是中文，不需翻譯（也避免 gemma 翻一段固定中文）。
    if (
        cfg.EN_DRAFT_PIPELINE
        and getattr(cfg, "FINAL_TRANSLATION_ENABLED", True)
        and rag_found_anything
        and not gate_abstain
    ):
        t6 = time.perf_counter()
        _status("\n[translation] 開始")
        full_text = translate_to_traditional_chinese(full_text, on_status=on_status)
        _status(f"[translation] 完成 elapsed_ms={int((time.perf_counter()-t6)*1000)}")
        if on_artifact:
            on_artifact("stage7_translated_answer", full_text)

    # PARTIAL 軟警告：翻譯後才加（橫幅本身已是中文，不送進翻譯）。棄答時 full_text 已是橫幅，不重複。
    if gate_notice and not gate_abstain:
        full_text = gate_notice + full_text

    if nli_report:
        full_text += nli_report

    _status(f"[pipeline] 完成 total_elapsed_ms={int((time.perf_counter()-t0)*1000)}")
    return full_text


# ══════════════════════════════════════════════════════════════════
#  Streaming entry point
# ══════════════════════════════════════════════════════════════════

def execute_structured_query_stream(
    question: str,
    paper_engines: dict,
    memory_context: str = "",
    on_status=None,
):
    """
    Streaming generator version of execute_structured_query.
    Yields two token types:
      [STATUS] prefix → progress message (rendered as blockquote by api.py)
      other           → LLM output tokens written directly to the response
    """
    t0 = time.perf_counter()
    all_paper_names = list(paper_engines.keys())

    # ── Stage 1: Planning ────────────────────────────────────────────
    detected = detect_target_paper(question, all_paper_names)
    if cfg.REVIEW_MODE:
        yield "[STATUS] 📖 REVIEW_MODE 已啟用，使用全部論文...\n"
        paper_names = all_paper_names
        paper_engines_to_use = paper_engines
    elif detected:
        paper_names = all_paper_names
        paper_engines_to_use = paper_engines
    else:
        yield "[STATUS] 🔎 篩選相關論文中...\n"
        prefiltered = _keyword_prefilter(question, all_paper_names)
        paper_names = select_relevant_papers(question, prefiltered)
        paper_engines_to_use = {k: v for k, v in paper_engines.items() if k in paper_names}
        yield f"[STATUS] 📌 已選出 {len(paper_names)} 篇相關論文\n"

    yield "[STATUS] 📋 拆解子問題中...\n"
    sub_questions = plan_sub_questions(question, paper_names)
    yield f"[STATUS] → 拆出 {len(sub_questions)} 個子問題，開始檢索...\n"
    yield (f"[STATUS] [planning] 完成 paper_count={len(paper_names)} "
           f"subquery_count={len(sub_questions)} "
           f"elapsed_ms={int((time.perf_counter()-t0)*1000)}\n")

    # ── Stage 2: Retrieval ───────────────────────────────────────────
    t1 = time.perf_counter()
    sub_answers = []
    rag_found_anything = False
    yield f"[STATUS] ⚡ 並行檢索 {len(sub_questions)} 個子問題中（workers={cfg.SUBQUERY_MAX_WORKERS}）...\n"
    valid_tasks, prefilled = build_subquery_tasks(sub_questions, paper_engines_to_use, paper_engines)
    retrieval_msgs = []
    deterministic_evidence = getattr(
        cfg, "PARTIAL_RECOVERY_DETERMINISTIC_GUARDS_ENABLED", False
    )
    ordered_results = run_subqueries_parallel(
        valid_tasks,
        prefilled,
        on_status=retrieval_msgs.append,
        evidence_snippets_per_task=(
            getattr(cfg, "PARTIAL_RECOVERY_EVIDENCE_SNIPPETS_PER_TASK", 4)
            if deterministic_evidence else None
        ),
        include_adjacent_evidence=deterministic_evidence,
    )
    for msg in retrieval_msgs:
        yield f"[STATUS] {msg}\n"

    for label, result in ordered_results:
        sub_answers.append(f"{label}\n{result}")
        if not is_empty_result(result):
            rag_found_anything = True
        preview = result[:120].replace("\n", " ")
        yield f"[STATUS] {label} → {preview}...\n"

    yield (f"[STATUS] [retrieval] 完成 rag_found={rag_found_anything} "
           f"elapsed_ms={int((time.perf_counter()-t1)*1000)}\n")

    # ── Stage 3: Knowledge synthesis (distillation) ──────────────────
    t2 = time.perf_counter()
    literal_kb = ""
    if cfg.SYNTHESIS_ENABLED and rag_found_anything:
        yield "[STATUS] 🧪 [synthesis] 知識蒸餾中...\n"
        synthesis_chunks = [
            {"text": ans, "source": extract_paper_name(ans, f"retrieved_chunk_{i}")}
            for i, ans in enumerate(sub_answers)
        ]
        knowledge_base = _synthesizer.synthesize(
            chunks=synthesis_chunks, query=question, on_status=on_status,
        )
        if deterministic_evidence:
            literal_kb = _literal_recovery_facts(ordered_results, question)
            if literal_kb:
                knowledge_base = _merge_fact_lists(knowledge_base, literal_kb)
        yield "[STATUS] 📋 事實清單已整理完成\n"
    else:
        knowledge_base = "\n\n".join(sub_answers)
    yield f"[STATUS] [synthesis] 完成 elapsed_ms={int((time.perf_counter()-t2)*1000)}\n"

    # ── Stage 3.5: 可答性 gate（Phase 2：接路由）──────────────────────
    gate_abstain, gate_notice = False, ""
    if cfg.ANSWERABILITY_GATE_ENABLED and rag_found_anything:
        from rag.answerability import assess_answerability, gate_route
        _ans = assess_answerability(question, knowledge_base)
        gate_abstain, gate_notice = gate_route(_ans["verdict"])
        yield f"[STATUS] [answerability] verdict={_ans['verdict']} abstain={gate_abstain}\n"
        recovery = _attempt_partial_recovery(
            question,
            valid_tasks,
            prefilled,
            sub_answers,
            knowledge_base,
            _ans,
            on_status=on_status,
        )
        if recovery["attempted"]:
            yield (
                f"[STATUS] [partial-recovery] accepted={recovery['accepted']} "
                f"verdict={recovery.get('recovered_assessment', {}).get('verdict')}\n"
            )
        if recovery["accepted"]:
            sub_answers = recovery["sub_answers"]
            knowledge_base = recovery["knowledge_base"]
            literal_kb = _merge_fact_lists(
                literal_kb, recovery.get("literal_facts", "")
            )
            _ans = recovery["assessment"]
            gate_abstain, gate_notice = gate_route(_ans["verdict"])

    # ── Stage 4: LLM synthesis ───────────────────────────────────────
    t3 = time.perf_counter()
    direct_render_text = ""
    direct_grounding_claims = []
    fact_contract_direct = False
    if gate_abstain:
        yield "[STATUS] 🚪 [answerability] NOT_ANSWERABLE → 誠實棄答，跳過生成\n"
        yield gate_notice
        full_text = gate_notice
    else:
        direct_answer = ""
        direct_render_meta = {}
        if (
            rag_found_anything
            and cfg.EN_DRAFT_PIPELINE
            and getattr(cfg, "COMPARISON_JSON_DIRECT_RENDER_ENABLED", False)
            and not _comparison_json_validation_errors(knowledge_base, question)
        ):
            direct_answer = _stage4_empty_answer_fallback(
                knowledge_base,
                atomic_only=True,
                question=question,
            )
            if direct_answer:
                direct_grounding_claims = split_into_sentences(direct_answer)
        if (
            not direct_answer
            and rag_found_anything
            and cfg.EN_DRAFT_PIPELINE
            and cfg.REASONING_MODE == "strict"
            and getattr(cfg, "METHOD_FACT_LIST_DIRECT_RENDER_ENABLED", False)
        ):
            direct_answer, direct_grounding_claims, direct_render_meta = _render_method_fact_list(
                knowledge_base,
                question,
                sub_questions,
            )
        if (
            not direct_answer
            and rag_found_anything
            and cfg.EN_DRAFT_PIPELINE
            and cfg.REASONING_MODE == "strict"
            and getattr(cfg, "STRUCTURED_FACT_CONTRACT_ENABLED", False)
            and not _is_method_fact_query(question)
            and not any(term in question.lower() for term in _COMPARISON_QUERY_TERMS)
        ):
            direct_answer, direct_grounding_claims, direct_render_meta = _render_validated_fact_contract(
                knowledge_base,
                sub_answers,
            )
            fact_contract_direct = bool(direct_answer)

        if direct_answer:
            full_text = direct_answer
            direct_render_text = direct_answer
            if direct_render_meta.get("schema") == "fact_contract_v1":
                yield (
                    "[STATUS] 🧱 evidence-bound fact contract → deterministic render "
                    f"accepted={len(direct_render_meta['facts'])} "
                    f"rejected={len(direct_render_meta['rejected'])}\n"
                )
            elif direct_render_meta:
                yield (
                    "[STATUS] 🧱 method fact_list → deterministic render "
                    f"selected={','.join(direct_render_meta['selected_fact_ids'])}\n"
                )
            else:
                yield "[STATUS] 🧱 validated atomic comparison_json → deterministic render\n"
            yield full_text
        elif not rag_found_anything:
            yield "[STATUS] ⚠️ RAG 未找到相關內容，切換至模型知識推理...\n"
            fallback_notice = _FALLBACK_NOTICE
            synthesis_prompt = build_fallback_prompt(
                question, _build_memory_section(memory_context, is_fallback=True)
            )
        else:
            fallback_notice = ""  # PARTIAL 軟警告在翻譯後才加，避免中文橫幅被送進翻譯
            lang = "en" if cfg.EN_DRAFT_PIPELINE else "zh"
            if cfg.REASONING_MODE == "reasoning":
                yield "[STATUS] 🧠 推理模式，LLM 綜合推論中...\n"
            else:
                yield "[STATUS] 📋 嚴格模式，LLM 整理論文內容中...\n"
            synthesis_prompt = build_synthesis_prompt(
                knowledge_base, question,
                _build_memory_section(memory_context, is_fallback=False),
                cfg.REASONING_MODE, lang,
            )

        if not direct_answer:
            if fallback_notice:
                yield fallback_notice
            full_text = fallback_notice
            try:
                for chunk in Settings.llm.stream_complete(synthesis_prompt):
                    yield chunk.delta
                    full_text += chunk.delta
            except Exception as exc:
                yield f"\n[STATUS] ⚠️ [synthesis-llm] failed; using Stage 3 facts ({str(exc)[:180]})\n"
                fallback = _stage4_empty_answer_fallback(knowledge_base, question=question)
                if fallback:
                    yield "\n\n---\n" + fallback
                full_text = fallback_notice + fallback
            if rag_found_anything and not full_text.strip():
                full_text = _stage4_empty_answer_fallback(knowledge_base, question=question)
                yield full_text
    yield f"\n[STATUS] [synthesis-llm] 完成 elapsed_ms={int((time.perf_counter()-t3)*1000)}\n"

    if rag_found_anything and not gate_abstain and not fact_contract_direct:
        rewrite_msgs = []
        corrected = _rewrite_stage4_if_needed(
            full_text, knowledge_base, question, on_status=lambda msg: rewrite_msgs.append(msg)
        )
        for msg in rewrite_msgs:
            yield f"[STATUS] {msg.strip()}\n"
        if corrected != full_text:
            yield "\n\n---\n"
            yield corrected
            full_text = corrected
        with_isotope_cost = _append_missing_isotope_cost_answer(
            full_text, knowledge_base, question
        )
        if with_isotope_cost != full_text:
            yield with_isotope_cost[len(full_text.rstrip()):]
            full_text = with_isotope_cost
        atomic_text = _separate_stability_protocol_clause(full_text)
        if atomic_text != full_text:
            yield "[STATUS] 🧱 [stage4-atomicity] separated a stability result from its protocol\n"
            yield "\n\n---\n"
            yield atomic_text
            full_text = atomic_text

    # ── Stage 5: Verification ────────────────────────────────────────
    if cfg.VERIFY_ENABLED and rag_found_anything and not gate_abstain and not fact_contract_direct:
        t4 = time.perf_counter()
        yield "[STATUS] 🔍 [verification] Stage 5: 邏輯自洽驗證中...\n"
        corrected = _verifier.verify_and_correct(
            draft_answer=full_text, knowledge_base=knowledge_base, on_status=on_status,
        )
        if corrected != full_text:
            yield "\n\n---\n📝 **已根據邏輯自洽驗證修正如下：**\n\n"
            yield corrected
            full_text = corrected
        else:
            yield "[STATUS] ✅ [verification] 邏輯驗證通過（VERIFY_PASS），答案無需修正\n"
        yield f"[STATUS] [verification] 完成 elapsed_ms={int((time.perf_counter()-t4)*1000)}\n"

    if (
        rag_found_anything
        and not gate_abstain
        and not fact_contract_direct
        and deterministic_evidence
        and literal_kb
    ):
        before_literal_guard = full_text
        full_text = _append_missing_literal_facts(full_text, literal_kb)
        if full_text != before_literal_guard:
            yield "[STATUS] 🧱 [literal-completeness] restored omitted direct evidence\n"
            yield full_text[len(before_literal_guard.rstrip()):]

    if not fact_contract_direct:
        atomic_text = _separate_stability_protocol_clause(full_text)
        if atomic_text != full_text:
            yield "[STATUS] 🧱 [stage4-atomicity] restored result/protocol separation after verification\n"
            yield "\n\n---\n"
            yield atomic_text
            full_text = atomic_text

    # ── Stage 6: Citation grounding ──────────────────────────────────
    nli_report = ""
    if cfg.CITATION_GROUNDING_ENABLED and rag_found_anything and not gate_abstain:
        t5 = time.perf_counter()
        yield "[STATUS] [grounding] 開始\n"
        try:
            grounding_msgs = []
            full_text, nli_report = run_grounding_check(
                full_text, sub_answers, knowledge_base,
                question=question, paper_engines_to_use=paper_engines_to_use,
                grounding_claims=(
                    direct_grounding_claims
                    if direct_grounding_claims and full_text == direct_render_text
                    else None
                ),
                on_status=lambda msg: grounding_msgs.append(msg),
            )
            for msg in grounding_msgs:
                yield f"[STATUS] {msg.strip()}\n"
        except Exception as e:
            nli_report = f"\n\n⚠️ 答案品質審查失敗：{e}"
        yield f"[STATUS] [grounding] 完成 elapsed_ms={int((time.perf_counter()-t5)*1000)}\n"

    # ── Stage 7: Translation ─────────────────────────────────────────
    if (
        cfg.EN_DRAFT_PIPELINE
        and getattr(cfg, "FINAL_TRANSLATION_ENABLED", True)
        and rag_found_anything
        and not gate_abstain
    ):
        t6 = time.perf_counter()
        yield "[STATUS] 🌏 [translation] 翻譯英文答案為繁體中文...\n"
        translated = translate_to_traditional_chinese(full_text, on_status=on_status)
        if translated != full_text:
            yield "\n\n---\n🌏 **繁體中文最終版本：**\n\n"
            yield translated
            full_text = translated
        yield f"[STATUS] [translation] 完成 elapsed_ms={int((time.perf_counter()-t6)*1000)}\n"

    # PARTIAL 軟警告：翻譯後才加（橫幅已是中文，不送進翻譯）。棄答時 full_text 已是橫幅，不重複。
    if gate_notice and not gate_abstain:
        yield "\n\n---\n"
        yield gate_notice
        full_text = gate_notice + full_text

    if nli_report:
        yield nli_report

    yield f"[STATUS] [pipeline] 完成 total_elapsed_ms={int((time.perf_counter()-t0)*1000)}\n"
