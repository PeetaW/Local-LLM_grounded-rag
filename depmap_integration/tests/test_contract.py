"""Independent client contract tests. No live API assertions in this suite."""
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from depmap_integration import client

class ContractTests(unittest.TestCase):
    def test_search_path(self):
        with patch.object(client, "_get", return_value={"data": []}) as get:
            self.assertEqual(client.search_models("SCC-25")["endpoint"], "/search")
            get.assert_called_once_with("/search/SCC-25")
    def test_gene_path(self):
        with patch.object(client, "_get", return_value={"data": []}) as get:
            client.gene_dataset("SLC7A5", "rnaseq")
            get.assert_called_once_with("/genes/SLC7A5/datasets/rnaseq")
    def test_invalid_assay(self):
        with self.assertRaises(ValueError): client.gene_dataset("SLC7A5", "invalid")
    def test_empty_query(self):
        with self.assertRaises(ValueError): client.search_models(" ")
    def test_long_query(self):
        with self.assertRaises(ValueError): client.search_models("x"*129)
    def test_escape_path(self):
        with patch.object(client, "_get", return_value={"data":[]}) as get:
            client.search_models("a/b")
            get.assert_called_once_with("/search/a%2Fb")
    def test_reject_url(self):
        with self.assertRaises(ValueError): client._get("https://evil.example/")
    def test_jsonapi_success(self):
        with patch.object(client, "urlopen", return_value=_Response(b'{"data":[]}')):
            self.assertEqual(client._get("/models", retries=0), {"data":[]})
    def test_invalid_payload(self):
        with patch.object(client, "urlopen", return_value=_Response(b'{"bad":[]}')):
            with self.assertRaises(client.DepMapError): client._get("/models", retries=0)
    def test_404(self):
        with patch.object(client, "urlopen", side_effect=HTTPError("u",404,"missing",{},None)):
            with self.assertRaisesRegex(client.DepMapError,"404"): client._get("/models", retries=0)
    def test_429_retry(self):
        with patch.object(client,"urlopen",side_effect=[HTTPError("u",429,"rate",{},None),_Response(b'{"data":[]}')]) as req, patch.object(client.time,"sleep"):
            self.assertEqual(client._get("/models", retries=1), {"data":[]})
            self.assertEqual(req.call_count, 2)
    def test_network_failure(self):
        with patch.object(client,"urlopen",side_effect=URLError("offline")):
            with self.assertRaisesRegex(client.DepMapError,"transport"): client._get("/models", retries=0)

class _Response:
    def __init__(self, body): self.body=body
    def __enter__(self): return self
    def __exit__(self,*args): return None
    def read(self,*args): return self.body

if __name__=="__main__": unittest.main()
