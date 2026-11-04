"""Minimal read-only Sanger DepMap client with explicit provenance."""
from __future__ import annotations
import json
from urllib.parse import quote
from urllib.request import Request, urlopen

BASE_URL = "https://api.cellmodelpassports.sanger.ac.uk"

class DepMapError(RuntimeError):
    pass

def _get(path: str, timeout: int = 20) -> dict:
    request = Request(BASE_URL + path, headers={"Accept": "application/vnd.api+json", "User-Agent": "academic-depmap-research/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except Exception as exc:
        raise DepMapError(f"DepMap request failed for {path}: {exc}") from exc

def search_models(query: str) -> dict:
    if not query.strip():
        raise ValueError("query must not be empty")
    return {"source": "Sanger Cell Model Passports", "endpoint": "/search", "query": query,
            "data": _get("/search/" + quote(query.strip(), safe=""))}

def gene_dataset(gene_id: str, assay: str) -> dict:
    allowed = {"rnaseq", "crispr_ko", "proteomics", "genecnv"}
    if assay not in allowed:
        raise ValueError(f"assay must be one of {sorted(allowed)}")
    if not gene_id.strip():
        raise ValueError("gene_id must not be empty")
    path = "/genes/" + quote(gene_id.strip(), safe="") + "/datasets/" + assay
    return {"source": "Sanger Cell Model Passports", "endpoint": path,
            "gene_id": gene_id, "assay": assay, "data": _get(path)}
