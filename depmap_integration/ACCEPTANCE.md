# DepMap integration acceptance contract (independent gates)

Status: NOT ACCEPTED. Passing mocked unit tests alone is insufficient.

## Legal gate
- Sanger allows individual non-commercial academic API access without registration.
- Third-party application integration requires prior permission: depmap@sanger.ac.uk.
- No public service deployment or automatic ChatGPT plugin enablement until permission is documented.
- Source: https://depmap.sanger.ac.uk/documentation/api/

## Independent acceptance runner
A separate test suite, not the implementation itself, must execute the following gates and produce machine-readable JSON with case ID, observed result, expected result, pass/fail, evidence URL, timestamp, and data release.

1. Live API: query /search/FaDu, /search/SCC-25, /search/SAS, /search/U87MG; inspect raw JSON and record actual matches, not assumed coverage.
2. Identity: map aliases and Cellosaurus, Sanger and Broad IDs; reject ambiguous names, derivatives, and mismatched IDs.
3. Data: independently retrieve SLC7A5 and SLC3A2 expression plus CRISPR KO when available; compare parsed values with upstream raw records.
4. Provenance: source, release, assay, units, model ID, retrieval timestamp and original endpoint are mandatory for each numeric claim.
5. Missing data: missing model/assay must return 'unavailable', never zero or inferred values.
6. Failure simulation: timeout, HTTP 429, HTTP 500, malformed JSON, ambiguous ID, stale cache, empty dataset, and partial pagination must not silently succeed.
7. MCP: test tool discovery, schemas, parameter validation, successful call, invalid call, and safe failure response.
8. Security: authentication, least privilege, TLS, rate limiting, audit logging, no credential leakage, dependency vulnerability scan, SSRF/path traversal rejection.
9. ChatGPT: real installed/authorized plugin, observed automatic tool selection, observed tool response, source-bound citation; a mocked LLM call does not count.
10. Local RAG: real model-router tool selection, observed MCP call, source-bound output, no regression to existing paper pipeline.
11. Performance: record cold/warm latency, cache hit rate, and request count; define thresholds before judging success.
12. Reproducibility: pinned versions, repeatable run command, environment metadata, and signed-off test report.

**Final pass rule:** every mandatory gate passes with independent evidence; no untested item may be marked passed. A public HTTPS deployment is not a zero-risk guarantee.
