# Offline QA checkpoint — 2026-11-03

Branch: feature/research-agent-adaptive-iteration
PR: https://github.com/PeetaW/Local-LLM_grounded-rag/pull/10
Production RAG config/API/query_pipeline remained unchanged.

## Evidence used in private analysis

Real instrument exports supplied by the owner were inspected in a private
workspace, NOT checked into the public repository:

- Bruker RAW4.00 XRD exported text: two distinct samples, 3,669 XY rows each.
- Raman ASCII: 1-1.asc, 2,000 XY rows; additional sample/spot exports inspected.
- FTIR CSV: 1-1.csv, 3,601 XY rows; %T headers detected; additional sample/spot exports inspected.
- Private owner slide template: 7 sample slides, 13.3333 x 7.5 inches.

## Issues discovered and fixed

- Immediate-neighbor peak finder was noise-sensitive and missed XRD's major
  broad peak; replace with SciPy prominence on Savitzky-Golay smoothed signal.
  On two actual XRD exports, *illustrative* thresholds yielded eight candidate
  peaks each with a prominent 35.7 degree peak. Do not interpret counts as phases.
- Raw Raman trace produced many candidate noise extrema with old rule; improved
  thresholded smoothing and physical spacing. These thresholds are a QA
  starting point, not a validated fit or universal method.
- FTIR mode now explicitly requires transmittance or absorbance; %T peaks are
  minima in the measured trace. FTIR plots now show high wavenumber at left.
- Bruker RAW4 text header is handled separately; independent sample IDs and
  measurement spot IDs pass through the report pipeline.
- PPT master/layout inspection uncovered a major fidelity trap: the literal
  Blank layout uses the cover's large triangle background; the owner's
  content slides use the "只有標題" (Title Only) layout with top chevrons and
  dashed separator. Cover and content now select separately.
- Exclude absolute filesystem paths from visible PPTX source footers.
- Research Tool Router now restricts tool names, path roots, file sizes,
  extensions, and the number of inputs; no arbitrary shell execution.
- Initial SciPy CI failure was due to missing optional package; repaired by
  requirements-research-tools.txt. Branch CI with integration and sandbox
  tests passed as of Actions run 41673769860.

## Important caveats

- The private visual/numerical pilot was executed separately from the
  repository checkout (container internet clone unavailable); exact repository
  software was verified in GitHub Actions using synthetic fixtures.
- The generated PPTX visual comparison was manually inspected using the
  owner's private template. A perfect match on every original slide style
  is NOT yet guaranteed.
- No Rietveld / reference matching / quantitative phase estimation /
  absolute purity determination is implemented.
- No local Ollama RTX GPU full benchmark has been executed.
- No fully autonomous LLM-to-research-tool routing has been enabled.
- All new code remains isolated from the existing seven-stage production RAG.

## Next gate requiring the owner's own PC

Follow docs/LOCAL_TEST_WINDOWS.md to run:
1. Independent worktree + offline tests under Windows.
2. Local real instrument exports and owner template smoke test.
3. V12.1 unchanged 12-question Ollama GPU regression, capture result artifacts.
4. Compare with prior baseline if present and inspect SCIENTIFIC correctness
   and negative controls, not just average scores.

Do not merge this Draft PR until these gates and an explicit review pass.
