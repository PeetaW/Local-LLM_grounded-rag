"""Deterministic local DepMap tool router; NOT an LLM autonomous tool caller."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Callable

GENES = ("SLC7A5", "SLC3A2", "HIF1A")
MODELS = ("SCC-25", "U87MG", "FaDu", "SAS")
ASSAYS = {"expression":"rnaseq", "rna":"rnaseq", "rna-seq":"rnaseq", "表達":"rnaseq",
          "crispr":"crispr_ko", "dependency":"crispr_ko", "依賴":"crispr_ko", "敲除":"crispr_ko"}

@dataclass(frozen=True)
class Decision:
    tool: str | None
    args: dict
    reason: str

def route(question: str) -> Decision:
    if not isinstance(question, str) or not question.strip():
        return Decision(None, {}, "empty_question")
    q = question.casefold()
    if not any(k in q for k in ("depmap","sanger","cell line","細胞株","基因表達","crispr","dependency","lat1","slc7a5","slc3a2","hif1a")):
        return Decision(None, {}, "out_of_scope")
    genes = [g for g in GENES if re.search(r"(?<![a-z0-9])"+g.casefold()+r"(?![a-z0-9])",q)]
    models = [m for m in MODELS if re.search(r"(?<![a-z0-9])"+m.casefold()+r"(?![a-z0-9])",q)]
    if "lat1" in q and "SLC7A5" not in genes:
        genes.append("SLC7A5")
    assay = next((v for k,v in ASSAYS.items() if k in q), None)
    if genes and assay:
        if len(genes) != 1:
            return Decision(None, {}, "multi_gene_requires_planning")
        return Decision("depmap_gene_assay", {"gene_id":genes[0],"assay":assay}, "gene_assay_intent")
    if models and not genes:
        if len(models) != 1:
            return Decision(None, {}, "multi_model_requires_planning")
        return Decision("depmap_search_models", {"query":models[0]}, "model_lookup_intent")
    return Decision(None, {}, "ambiguous_or_unsupported_query")

def dispatch(decision: Decision, tools: dict[str, Callable]) -> dict:
    if decision.tool is None:
        return {"status":"needs_planning","reason":decision.reason}
    fn = tools.get(decision.tool)
    if fn is None:
        return {"status":"unavailable","reason":"tool_not_registered","tool":decision.tool}
    try:
        result = fn(**decision.args)
    except Exception as exc:
        return {"status":"error","tool":decision.tool,"error_type":type(exc).__name__}
    return {"status":"ok","tool":decision.tool,"arguments":decision.args,"result":result}
