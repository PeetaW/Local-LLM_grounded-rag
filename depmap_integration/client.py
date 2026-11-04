"""Minimal read-only Sanger DepMap client with explicit provenance."""
from __future__ import annotations
import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

BASE_URL = "https://api.cellmodelpassports.sanger.ac.uk"

class DepMapError(RuntimeError):
    pass

def _get(path: str, timeout: int = 15, retries: int = 2) -> dict:
    if not path.startswith("/") or "://" in path or "?" in path:
        raise ValueError("Invalid API path")
    request = Request(BASE_URL + path, headers={"Accept": "application/vnd.api+json", "User-Agent": "academic-depmap-research/0.2"})
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if not isinstance(payload, dict) or "data" not in payload:
                raise DepMapError("Invalid JSON:API response (missing data)")
            return payload
        except HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(min(2 ** attempt, 4))
                continue
            raise DepMapError(f"Upstream HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            if attempt < retries:
                time.sleep(min(2 ** attempt, 4))
                continue
            raise DepMapError(f"Upstream transport error: {type(exc).__name__}") from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise DepMapError("Upstream malformed JSON") from exc

def search_models(query: str) -> dict:
    if not isinstance(query, str) or not query.strip() or len(query) > 128:
        raise ValueError("query must be a nonempty string <=128 characters")
    return {"source": "Sanger Cell Model Passports", "endpoint": "/search", "query": query,
            "data": _get("/search/" + quote(query.strip(), safe=""))}

def gene_dataset(gene_id: str, assay: str) -> dict:
    allowed = {"rnaseq", "crispr_ko", "proteomics", "genecnv"}
    if assay not in allowed:
        raise ValueError(f"assay must be one of {sorted(allowed)}")
    if not isinstance(gene_id, str) or not gene_id.strip() or len(gene_id) > 128:
        raise ValueError("gene_id must be a nonempty string <=128 characters")
    path = "/genes/" + quote(gene_id.strip(), safe="") + "/datasets/" + assay
    return {"source": "Sanger Cell Model Passports", "endpoint": path,
            "gene_id": gene_id, "assay": assay, "data": _get(path)}
