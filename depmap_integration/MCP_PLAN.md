# DepMap MCP integration — work in progress, NOT production-ready

## Separation
- Local Ollama RAG: local stdio MCP, launched on demand, no inbound ports.
- ChatGPT: separate authenticated HTTPS Streamable HTTP service, not the local stdio process.
- Both: shared validated client, normalized identifiers, dataset provenance, error handling and independent tests.
- Sanger third-party application integration requires permission before deployment: https://depmap.sanger.ac.uk/documentation/api/

## Implemented prototype
- Local read-only MCP adapter: `python -m depmap_integration.mcp_server`
- Tools: `depmap_search_models`, `depmap_gene_assay`
- The adapter delegates to existing `client.py`. It has not been end-to-end tested and is not connected to the Ollama router.

## Remaining mandatory work
1. Verify Sanger live responses and resolve SAS, FaDu, SCC-25 and U87MG with unique IDs.
2. Add independent integration tests for endpoint correctness, payload schemas, pagination, units and dataset release.
3. Implement SQLite/DuckDB cache with TTL, request timeout, 429 backoff, and strict provenance.
4. Connect local MCP client and tool selection into actual RAG router; record model tool calls.
5. Audit all existing local tool interfaces and split MCP servers by permission and runtime: literature, analytical chemistry, spectra, artifacts.
6. Build separate authenticated HTTPS MCP gateway, with no filesystem or private RAG access.
7. Confirm Sanger permission, then deploy; verify TLS, auth, rate limits, and real ChatGPT connection.
8. Run the independent ACCEPTANCE.md gates; record failures, not just successes.

## Security
The local stdio server is intentionally not a network listener. Do not expose it using a reverse proxy. Do not expose arbitrary file paths, Python execution, secrets or write access via remote MCP.
