# DepMap integration (experimental)

Read-only prototype for the Sanger Cell Model Passports API. No experimental values are bundled or fabricated.

## Usage

```python
from depmap_integration.client import search_models, gene_dataset
print(search_models("FaDu"))
# Resolve the official gene identifier before using gene_dataset:
# print(gene_dataset("SIDG...", "rnaseq"))
```

Sanger API docs: https://depmap.sanger.ac.uk/documentation/api/endpoints/

**Important:** The Sanger API permits individual non-commercial use without login; third-party applications should contact depmap@sanger.ac.uk before direct API integration. Check terms before deploying this component as a hosted service.

## Next steps

- Confirm official gene identifiers and cell-model IDs, including aliases.
- Add bounded pagination, retry/backoff, cache, and version metadata.
- Implement local Broad release matrix ingestion with checksum verification.
- Expose validated MCP tools only after integration and contract tests.
- Benchmark SAS, FaDu, SCC-25, U87MG coverage; report missing models instead of imputing values.
