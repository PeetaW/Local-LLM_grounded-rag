"""Official MCP v2 in-process client and real stdio protocol tests."""
import asyncio
import tempfile
import unittest
from pathlib import Path

from rag.research_agent.mcp_server import create_server
from rag.research_agent.mcp_client import call_inprocess, call_private_stdio
from rag.research_agent.mcp_backend import MCPToolBackend


class MCPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "sample.asc").write_text(
            "100,1\n101,8\n102,120\n103,8\n104,1", encoding="utf-8"
        )

    def test_inprocess_client_spectra(self):
        async def check():
            from mcp import Client
            server = create_server(self.root)
            async with Client(server, raise_exceptions=True) as client:
                listed = await client.list_tools()
                names = {t.name for t in listed.tools}
                self.assertEqual(
                    names, {"spectroscopy_report", "presentation_from_discussion"}
                )
            result = await call_inprocess(server, "spectroscopy_report", {
                "measurements": [{"path": "sample.asc", "sample_id": "S1",
                                  "technique": "raman", "min_prominence": 15}],
                "output_dir": "report", "create_pptx": False
            })
            self.assertTrue(Path(result["json"]).is_file())
        asyncio.run(check())

    def test_mcp_block_escape(self):
        async def check():
            server = create_server(self.root)
            from mcp import Client
            async with Client(server, raise_exceptions=True) as client:
                result = await client.call_tool("spectroscopy_report", {
                    "measurements": [{"path": "../secret.asc", "sample_id": "S1",
                                      "technique": "raman", "min_prominence": 12}],
                    "output_dir": "report",
                })
                self.assertTrue(result.is_error)
        asyncio.run(check())

    def test_tool_registry_rejects_unknown_method(self):
        server = create_server(self.root)
        async def check():
            with self.assertRaises(ValueError):
                await call_inprocess(server, "execute_python", {})
        asyncio.run(check())

    def test_stdio_roundtrip(self):
        async def check():
            result = await call_private_stdio(self.root, "spectroscopy_report", {
                "measurements": [{"path": "sample.asc", "sample_id": "S1",
                                  "technique": "raman", "min_prominence": 15}],
                "output_dir": "stdio_report", "create_pptx": False
            })
            self.assertTrue(Path(result["json"]).is_file())
        asyncio.run(check())

    def test_sync_adapter(self):
        backend = MCPToolBackend(self.root)
        result = backend.run("spectroscopy.report", {
            "measurements": [{"path": "sample.asc", "sample_id": "S1",
                              "technique": "raman", "min_prominence": 15}],
            "output_dir": "backend_report", "create_pptx": False
        })
        self.assertTrue(Path(result["json"]).is_file())


if __name__ == "__main__":
    unittest.main()
