"""Read-only local DepMap MCP adapter. Requires mcp>=1.28,<2.

Run: python -m depmap_integration.mcp_server
Transport: stdio. Do not expose this module directly to the public internet.
"""
from __future__ import annotations
from mcp.server.fastmcp import FastMCP
from .client import search_models, gene_dataset

mcp = FastMCP("DepMap Sanger (research, read-only)")

@mcp.tool()
def depmap_search_models(query: str) -> dict:
    """Find cell models by name or identifier. Verify IDs before downstream analysis."""
    return search_models(query)

@mcp.tool()
def depmap_gene_assay(gene_id: str, assay: str = "rnaseq") -> dict:
    """Retrieve one gene's dataset. assay: rnaseq, crispr_ko, proteomics, genecnv.

    Return raw source payload with provenance. Missing data must not be imputed.
    """
    return gene_dataset(gene_id, assay)

if __name__ == "__main__":
    mcp.run(transport="stdio")
