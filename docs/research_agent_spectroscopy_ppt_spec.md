# Research Agent + Adaptive Iteration + Spectroscopy + PPT — development specification

Status: **working opt-in experimental Research Agent CLI, standalone local API and MCP v2 tooling**; not enabled for the original production `api.py` or existing RAG endpoints. Deterministic mocks and CI are not a substitute for real research correctness/RTX GPU benchmark.

## Non-regression gate

- Preserve the existing seven-stage RAG pipeline and current `config.py` defaults.
- Run the official V12.1 12-question product regression on the user's Ollama/RTX 3090
  before enabling the new agent in `api.py` or merging behavior changes.
- No repeated LLM calls unless an evaluator identifies a specific, actionable
  `FailureKind`. Stop on missing input, unsupported claims, repeated failure
  fingerprints, iteration limits or budget limits. Tokens are observed only if
  the executor supplies real usage metadata; execution limits are checked
  **between** synchronous calls, not preemptively.
- A passing model self-evaluation is insufficient evidence of scientific accuracy.

## Modules delivered in initial branch

- `rag/research_agent/controller.py`: opt-in callback-driven controller with
  FAST / ADAPTIVE / DEEP modes, whitelisted repairs, structured trace.
- `rag/research_tools/spectroscopy.py`: read text spectra, preserve sample IDs,
  screen extrema, export plots and comparison metadata.
- `rag/research_tools/presentation.py`: editable text PPTX export, optional
  actual user-owned master/template file to preserve its theme/layout.
- `rag/research_tools/slide_planner.py`: evidence-aware structured findings
  and Current questions / future work slide assembly.
- `rag/research_tools/report_pipeline.py`: deterministic data -> JSON + PNG +
  PPTX tool workflow.
- `rag/research_tools/registry.py`: allowlisted tool implementations; `rag/research_agent/tool_router.py` validates private workspace and filenames before invocation.
- `rag/research_agent/outcome_planner.py` / `intent_router.py`: route verified user inputs into a bounded dependent task graph, optional Ollama JSON planning.
- `rag/research_agent/workflow.py` / `rag_adapter.py` / `ollama_adapter.py`: opt-in local end-to-end execution, original RAG bridge and model planning.
- `rag/research_agent/mcp_server.py` / `mcp_client.py` / `mcp_backend.py`: official MCP v2 stdio server/client and in-process integration tests.
- `rag/research_agent/http_app.py`: separate token-authenticated local-only API; `api.py` remains unmodified.
- `rag/research_agent/evaluator.py` / `expert_profiles.py`: structural artifact checks and advisory specialist review, **not a scientific accuracy oracle**.
- `eval/research_agent_eval.py`: 8 synthetic offline scenarios with explicit expected stops. See `docs/RESEARCH_AGENT_BENCHMARK_WINDOWS.md` for real GPU/hardware acceptance.
- Offline tests under `tests/`.

## Scientific constraints

The MVP implements first-pass numerical peak screening, not verified phase
identification. For every measurement, record technique, sample_id, spot,
x-axis units, detector/signal mode and original file path. A user must declare
whether FTIR is transmittance or absorbance. Distinguish distinct samples
from different measurement spots; never call spots independent synthesis
replicates. Never infer Fe3O4/Fe2O3 quantitative phase ratios from peak heights.
XRD refinement requires instrument wavelength, standard patterns, full-pattern
fitting and uncertainty; Raman requires laser parameters, reference spectra
and tests for laser-induced transformation; FTIR requires measurement mode.
Quantitative mineral fractions and absolute purity are out of scope.

CSV/TSV/ASC/TXT/DAT with two numeric columns are implemented; binary SIF must
first be converted using trusted vendor/export tools. Spectral metadata,
background removal, smoothing, fitted FWHM, peak deconvolution, uncertainty,
XRD Rietveld refinement and reference database matching are *future* work.
The current peak finder is deterministic but simple; it may miss overlapped,
broad or noisy peaks. It is inappropriate for independent phase assignments.

## Presentation constraints

Use `rag/research_tools/STYLE_PROFILE.md` derived from the owner's saved PPT
style profile. Default fallback is neutral white, Times New Roman and the
recorded 16:9 coordinates. For high fidelity, specify a local existing
`Shuo_Hung_Research_Presentation_Template.pptx` via `template_path`.
Original files must stay local and must **not** be committed to this public
repository. Preserve master/theme rather than constructing a completely
unrelated modern template. Body text and citations are editable; graph PNGs
are currently non-editable images. Editable PowerPoint-native charts and
visual regression testing are deferred.

Structured discussion input can fill slides without inventing observations:
`ResearchFinding(topic, observation, interpretation, evidence_refs, status)`
where status is observed/tentative/validated. Only validated items enter the
conclusion automatically. Source/provenance fields should refer to real files
or traceable publications. Text fitting and multi-page overflow need manual QA.

## Local usage (optional packages)

```bash
python -m pip install matplotlib python-pptx
python -m unittest discover -s tests -p 'test_*research_agent.py'
python -m unittest discover -s tests -p 'test_spectroscopy.py'
python -m unittest discover -s tests -p 'test_presentation.py'
python -m unittest discover -s tests -p 'test_slide_planner.py'
```

```python
from rag.research_tools.report_pipeline import analyze_spectra_to_deck

result = analyze_spectra_to_deck(
    measurements=[
        {"path": "samples/sample1.csv", "technique": "xrd",
         "sample_id": "S1", "min_prominence": 25},
        {"path": "samples/sample2.asc", "technique": "raman",
         "sample_id": "S2", "min_prominence": 5},
        {"path": "samples/sample3.csv", "technique": "ftir",
         "sample_id": "S2", "signal_mode": "transmittance",
         "min_prominence": 4}
    ],
    output_dir="output/report",
    deck_title="Fe3O4 spectroscopy data",
    template_path="private/Shuo_Hung_Research_Presentation_Template.pptx",
)
print(result)
```

`min_prominence` is in *raw signal units* and must be chosen from the
particular instrument's noise; values above are placeholders, not validated
threshold recommendations.

## Rollout gates

1. Offline deterministic tests + syntax checks (no GPU, no network).
2. Add golden synthetic and instrument-export fixtures; verify axis/peaks,
   sample separation, FTIR transmittance inversion and warnings.
3. Manually inspect output PPTX versus owner template on PowerPoint.
4. Instrument-specific reference matching: validated reference data and
   constrained interpretation; never force a phase label.
5. Add model-to-tool routing behind opt-in flag, with local file root
   sandboxing and resource limits; test prompt injection and malicious paths.
6. Evaluate on held-out research questions and spectra, with grounded expert
   judgments and latency/iteration metrics; then discuss mainline merge.

## Cost and licensing

Local stdlib processing is free. `matplotlib` and `python-pptx` are
open-source/free. Ollama usage is local; model license and memory requirements
depend on chosen model. There is no mandatory paid API.
