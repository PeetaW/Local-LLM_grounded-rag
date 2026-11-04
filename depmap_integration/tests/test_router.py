import unittest
from depmap_integration.router import route, dispatch

class RouterTests(unittest.TestCase):
    def test_gene_expression(self):
        d=route("查詢 DepMap SCC-25 的 SLC7A5 表達量")
        self.assertEqual((d.tool,d.args),("depmap_gene_assay",{"gene_id":"SLC7A5","assay":"rnaseq"}))
    def test_lat1_alias(self):
        self.assertEqual(route("Sanger FaDu LAT1 RNA expression").args["gene_id"],"SLC7A5")
    def test_crispr(self):
        self.assertEqual(route("DepMap SLC3A2 CRISPR dependency").args["assay"],"crispr_ko")
    def test_model(self):
        self.assertEqual(route("DepMap FaDu 細胞株").tool,"depmap_search_models")
    def test_no_tool_for_nonresearch(self):
        self.assertIsNone(route("今天午餐吃什麼").tool)
    def test_multi_gene_abstains(self):
        self.assertEqual(route("DepMap SLC7A5 SLC3A2 RNA expression").reason,"multi_gene_requires_planning")
    def test_multi_model_abstains(self):
        self.assertEqual(route("DepMap FaDu SAS 細胞株").reason,"multi_model_requires_planning")
    def test_dispatch_mock(self):
        d=route("DepMap FaDu 細胞株")
        r=dispatch(d,{"depmap_search_models":lambda query:{"data":[{"id":"fixture"}]}})
        self.assertEqual(r["result"]["data"][0]["id"],"fixture")
    def test_dispatch_failure(self):
        def fail(**kwargs): raise TimeoutError("secret")
        r=dispatch(route("DepMap FaDu 細胞株"),{"depmap_search_models":fail})
        self.assertEqual(r["error_type"],"TimeoutError")
        self.assertNotIn("secret",str(r))
    def test_unregistered(self):
        self.assertEqual(dispatch(route("DepMap FaDu 細胞株"),{})["status"],"unavailable")

if __name__=="__main__": unittest.main()
