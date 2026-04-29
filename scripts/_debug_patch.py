import sys, types, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import MagicMock as _MM, patch, MagicMock

# Exactly mimic test_nli_extensions.py setup
if "requests" not in sys.modules:
    _req = types.ModuleType("requests")
    class _Timeout(OSError): pass
    _req.exceptions = types.SimpleNamespace(Timeout=_Timeout)
    _req.post = _MM()
    sys.modules["requests"] = _req

import config as cfg
cfg.EN_DRAFT_PIPELINE = True
cfg.NLI_CONTRADICTION_ENABLED = True
cfg.NLI_DECOMPOSE_ENABLED = True
cfg.NLI_JOINT_VERIFY_ENABLED = True
cfg.NLI_TRANSLATE_TO_EN = False

from rag.citation_grounding import decompose_and_verify

print("requests module:", sys.modules.get("requests"))
print("requests.post:", getattr(sys.modules.get("requests"), "post", "MISSING"))
print("requests.__dict__ keys:", list(sys.modules.get("requests").__dict__.keys()))

mock_resp = MagicMock()
mock_resp.ok = True
mock_resp.json.return_value = {"response": '["claim1", "claim2"]'}
mock_resp.raise_for_status = MagicMock()

try:
    with patch("requests.post", return_value=mock_resp):
        print("patch entered successfully")
except AttributeError as e:
    print("AttributeError on patch enter:", e)
    import traceback
    traceback.print_exc()
