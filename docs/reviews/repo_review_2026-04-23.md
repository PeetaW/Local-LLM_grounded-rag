# Repo Analysis & Code Review (2026-04-23)

## Scope
- Reviewed: architecture docs, config, API surface, initialization path, core RAG pipeline modules.
- Did not run end-to-end inference (local Ollama models and paper corpora are environment-dependent).

## Overall Assessment
This repository is already at a **strong prototype / pre-production** level for a local academic RAG stack:
- clear 5-stage pipeline design,
- explicit verification and correction loop,
- practical integration with Open WebUI via OpenAI-compatible endpoints,
- and thoughtful engineering notes in docs.

Main gaps are around **startup side effects**, **operational hardening**, **resilience under malformed LLM outputs**, and **test coverage automation**.

---

## Strengths

### 1) Architecture and product direction are clear
- README and spec describe a coherent 5-stage process with model role separation and rationale.
- ADR records key design tradeoff (status callback vs stdout interception), showing good engineering maturity.

### 2) Config centralization is good
- Most important runtime knobs are consolidated in `config.py`, including retrieval, verification, and NLI toggles.
- Explicit comments explain what changes require index rebuild.

### 3) Practical indexing and VL workflow
- Automatic VL backfill/re-run and orphan-index cleanup are good operational touches.
- Index build path includes chunk summarization before vectorization, which is a thoughtful retrieval-quality optimization.

### 4) API compatibility and UX
- OpenAI-compatible `/v1/chat/completions` and model listing endpoint support easy Open WebUI integration.
- Prompt-injection pattern checks and session-id validation are useful guardrails.

---

## Key Risks / Code Review Findings

### High Priority
1. **Import-time heavy initialization in `main.py`**
   - `main.py` performs model/index initialization at module import time.
   - `api.py` imports from `main`, so service startup implicitly triggers all expensive side effects.
   - Risk: slow/unreliable startup, hard testing, fragile dependency order.
   - Recommendation: move to explicit app startup lifecycle (`FastAPI` startup event or factory pattern).

2. **Sensitive / deployment-specific defaults in committed config**
   - `ACTIVE_PROJECT` is hard-coded to one project.
   - This can cause accidental cross-project behavior in multi-user deployments.
   - Recommendation: support env-var override (`ACTIVE_PROJECT`, model names, URLs), keep safe defaults in code.

3. **Verifier fail-open behavior may hide real failures**
   - In verifier single-batch execution, exceptions are treated as pass.
   - This improves availability but can silently degrade correctness.
   - Recommendation: return explicit degraded-state marker and surface in final answer metadata.

### Medium Priority
4. **Session memory store is process-local in-memory dict**
   - Good for prototype, but no TTL cleanup task / persistence / concurrency-safe external store.
   - Recommendation: move session store to Redis or persistent backend when serving real users.

5. **Prompt-injection defense is regex-only**
   - Basic protection exists but is bypassable by paraphrase/multilingual obfuscation.
   - Recommendation: add structured instruction hierarchy and model-side refusal policy checks.

6. **Index config mismatch handling exits process**
   - Current behavior uses `sys.exit(1)` when index config differs.
   - For server mode, hard process termination is harsh and operator-unfriendly.
   - Recommendation: raise typed exception and return actionable health/status endpoint details.

### Low Priority
7. **Verbose prints over structured logging**
   - Many modules use `print` instead of logger.
   - Recommendation: standardize on logging with request/session correlation ids.

8. **Codebase still contains archive scripts and mixed historical docs**
   - Useful for local experiments, but increases maintenance surface.
   - Recommendation: move legacy scripts behind a clear `legacy/` policy or separate branch.

---

## Suggested Roadmap (Practical)

### Milestone A (stability)
- Refactor initialization into explicit startup hook.
- Replace fail-open verifier fallback with `VERIFY_UNKNOWN` status and user-visible note.
- Convert critical `print` to logger with consistent format.

### Milestone B (operability)
- Env-var driven config layer + runtime config dump endpoint.
- Replace in-memory `session_store` with Redis (or SQLite with pruning for single-node).
- Add health checks for model availability and index integrity.

### Milestone C (quality)
- Add unit tests for:
  - session id resolution,
  - prompt injection detector,
  - subquery task flattening,
  - grounding-score parsing.
- Add one integration smoke test with mocked Ollama HTTP responses.

---

## Code Review Verdict
- **Direction**: Excellent and differentiated (academic-grounded local RAG, with verification loop).
- **Current quality**: Strong prototype with advanced features.
- **Before broader usage**: prioritize startup refactor + observability + explicit degraded-mode handling.

If these are addressed, this repo can move from "power-user research tool" to a much more production-ready service.
