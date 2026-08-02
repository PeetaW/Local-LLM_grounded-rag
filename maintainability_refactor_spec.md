# Maintainability Refactor Spec

> Status: planned, audited 2026-08-02. Start only after the full 12-question
> `baseline_v11_structured_contract_full` report is frozen.

## Goal

Reduce the maintenance cost of the active codebase without changing retrieval,
generation, grounding, API behavior, or evaluation semantics. This is an
equivalence refactor: move and simplify proven code first; add no framework,
dependency, interface, or speculative extension point.

## Audited Scope

The snapshot includes tracked active Python under the repository root, `rag/`,
`eval/`, and `scripts/`. Historical Python under `archive/` and generated data,
indexes, logs, and eval artifacts are excluded.

| Area | Files | Lines | Notes |
|---|---:|---:|---|
| Root entry points (`main.py`, `api.py`, `config.py`) | 3 | 883 | Runtime/bootstrap/config |
| `rag/` runtime | 29 | 9,855 | Core product pipeline |
| `eval/` runtime | 4 | 2,075 | Evaluation and judges |
| `scripts/` tests and tools | 22 | 10,091 | Unit tests, probes, historical utilities |
| **Total active Python** | **58** | **22,904** | 2026-08-02 snapshot |

Line count is only a triage signal. A large file is a refactor target only when
it also mixes responsibilities, contains long routines, duplicates flow, or has
high coupling.

## Hotspots

### Product and evaluation runtime

| File | Lines | Composition / observed pressure | Priority |
|---|---:|---|---|
| `rag/query_pipeline.py` | 1,854 | Orchestration plus fact/method/comparison rendering, recovery, Stage 4 validation; non-stream and stream entry points are 360/341 lines, comparison renderer is 330 lines | P0 |
| `eval/judge.py` | 1,195 | Correctness contract judge and translation fidelity judge in one module | P1 |
| `rag/citation_grounding.py` | 1,062 | NLI lifecycle/inference, lexical support, citation scoring/reporting, self-correction, decomposition and joint verification; circular dependency with `query_grounding_flow.py` | P1 |
| `rag/fact_contract.py` | 969 | Requirement detection/scoring, evidence catalog cleanup, schema, validation, completion, deduplication and rendering | P2 |
| `rag/knowledge_synthesizer.py` | 879 | Comparison normalization/schema/repair plus model generation and synthesis orchestration; `synthesize()` is 229 lines | P2 |
| `eval/run_eval.py` | 645 | Runner, debug artifacts, Markdown reporting, rejudge and compare CLI | P2 |
| `rag/query_retrieval.py` | 607 | Retrieval execution and evidence-window construction; currently cohesive enough | Monitor |
| `rag/comparison_json_validator.py` | 601 | Requirement extraction plus a 256-line validator | P2 |
| `api.py` | 475 | One 246-line OpenAI chat endpoint plus schemas/session/guardrails/runtime state | Existing API spec |

### Tests and tools

| File | Lines | Action |
|---|---:|---|
| `scripts/test_refactor.py` | 3,101 | Split by production module/behavior; keep assertions unchanged |
| `scripts/test_knowledge_synthesizer_prompt.py` | 1,573 | Split fact-contract, comparison-contract, and synthesizer tests |
| `scripts/test_judge.py` | 972 | Split correctness and translation judge tests |
| `scripts/test_nli_extensions.py` | 687 | Keep together until grounding modules are split |
| `scripts/preprocessing/vl_quality_test-1.py` | 612 | Keep as the canonical VL CLI, then rename cleanly |
| `scripts/preprocessing/vl_quality_test-1 backup.py` | 411 | Delete; Git history is the backup |
| `scripts/preprocessing/vl_quality_test.py` | 347 | Remove after confirming the `-1` script supersedes it; rename the canonical script to this path |
| `scripts/test_ab_retrieval.py` | 453 | Retire if `eval/run_eval.py --run/--compare` covers its remaining use; it targets the older `zvi` benchmark |
| `scripts/_debug_patch.py` | 37 | Delete one-off requests patch probe |
| `scripts/rebuild_q07_ligfix.py` | 100 | Remove from active scripts after preserving the command/result in eval history |
| `scripts/test_memory_check.py` | 26 | Rename to `inspect_memory.py`; it is an environment inspection tool, not a unit test |

Confirmed low-value cleanup can remove roughly 800 lines immediately, or up to
about 1,350 lines if the legacy retrieval A/B script is fully superseded. No
dependency removal is required.

## Refactor Plan

### M0 - Hygiene and test layout

Do first after the v11 baseline is frozen. These changes should not require an
AI pipeline run.

1. Delete the tracked VL backup and one-off debug patch.
2. Consolidate the three VL preprocessing variants into one documented CLI.
3. Confirm and retire the old `zvi` retrieval A/B runner if the eval harness has
   feature parity.
4. Move pure unit tests from `scripts/` into a discoverable `tests/` tree,
   mirroring production modules. Keep environment/model probes under `scripts/`.
5. Split the three oversized test files without rewriting fixtures.
6. Fix the invalid escape warnings in `scripts/test_stage5.py`.
7. Either use `PipelineContext` in the shared pipeline stages described below,
   or delete the currently unused dataclass and its tests.

Acceptance:

- identical test count before/after the move
- one standard-library discovery command runs all offline unit tests
- no generated files or model calls during unit discovery
- README commands and project tree match the new paths

### M1 - Query pipeline boundary

This is the highest-value runtime refactor because `query_pipeline.py` has the
highest coupling and changes frequently during quality work.

1. Extract pure fact/method/comparison rendering and Stage 4 contract checks to
   one focused `rag/structured_rendering.py` module.
2. Keep recovery logic separate from rendering; extract it only if the remaining
   pipeline still mixes recovery policy with orchestration.
3. Replace duplicated stage bodies in `execute_structured_query()` and
   `execute_structured_query_stream()` with explicit shared helpers for planning,
   retrieval, synthesis/rendering, grounding, and translation.
4. Preserve both public entry points and all status/artifact names. Do not build
   a generic event framework or plugin system.
5. Reuse the existing `PipelineContext` only if it measurably removes duplicated
   argument/state plumbing; otherwise delete it.

Targets are guidance, not arbitrary CI limits: `query_pipeline.py` below about
1,000 lines, no orchestration routine above about 180 lines, and no change to
streaming output order.

Acceptance:

- existing query/refactor tests pass unchanged
- saved Q01/Q04/Q07/Q08/Q09 artifacts replay to byte-equivalent English drafts
  apart from explicitly documented formatting
- API streaming/non-streaming parity is manually smoke-tested
- focused eval runs happen only after offline equivalence checks pass

### M2 - Judge and grounding boundaries

1. Split `eval/judge.py` into correctness and translation implementations while
   keeping `eval.judge` as the small compatibility facade used by `run_eval.py`.
2. Move shared sentence/citation parsing out of `query_grounding_flow.py` into a
   small dependency-neutral text helper so `citation_grounding.py` no longer
   imports its orchestrator.
3. If `citation_grounding.py` remains above the review threshold, extract only
   NLI model lifecycle/batch inference; keep scoring and decision policy together.
4. Split corresponding tests at the same time so each module has a clear owner.

Acceptance:

- correctness and translation schemas/results remain unchanged on saved reports
- NLI device, batching, lexical support, timing counters, and release behavior
  remain unchanged
- no new circular imports

### M3 - Structured contract internals

These modules are quality-critical, so refactor them only after M1/M2 and only
with saved-artifact replay.

1. Move fact requirement detection/scoring from `fact_contract.py` into a focused
   requirement module; keep catalog validation/completion/rendering together.
2. Split the 256-line comparison validator into private validators by existing
   schema section before considering another file.
3. Move comparison normalization/schema/repair out of
   `KnowledgeSynthesizer.synthesize()` only when one existing contract module can
   own the behavior cleanly.
4. Extract small internal methods from `synthesize()`; do not introduce a class
   hierarchy, strategy objects, or factories.

Acceptance:

- comparison JSON and fact-contract serialized artifacts remain compatible
- Q01/Q04/Q07/Q08/Q09 focused regression has no correctness, grounding, or
  translation regression
- repair count and Stage 3 latency do not increase

### M4 - Eval runner and API

1. Move Markdown/debug-artifact writing out of `eval/run_eval.py` only after the
   test layout is stable; keep one CLI and current flags.
2. Execute the existing `api-refactor-spec.md` independently. Do not combine API
   lifecycle work with query-pipeline or indexing refactors.

## Size Review Policy

Use these as review triggers, not hard style rules:

- runtime file above 1,200 lines: refactor before adding another responsibility
- runtime file between 800 and 1,200 lines: require a single clear owner/purpose
- function above 200 lines: split before adding another branch
- function above 120 lines: document why the flow must remain contiguous
- unit-test file above 1,200 lines: split by production ownership
- tracked files named `backup`, `old`, or `copy`: reject; Git is the history

Small adapters such as `retriever.py` and `reranker.py` are not merge targets just
because they are short; they have clear construction boundaries and active callers.

## Roadmap Placement

1. Finish and freeze the full v11 quality baseline.
2. Run M0 hygiene/test-layout work.
3. Perform Stage 3 latency work and M1 query-pipeline extraction in small,
   independently benchmarked commits.
4. Start pipeline v4 staged indexing, then ingestion-health Phase 2.
5. Apply M2/M3 opportunistically before adding behavior to the affected modules.
6. Implement memory redesign.
7. Perform the API refactor last unless API lifecycle becomes an operational
   blocker.

Plan-and-Execute and agentic retrieval remain deferred; they should not add more
branches to the current pipeline before the core flow is smaller and the full
baseline is stable.

## Definition of Done

- active code has no tracked backup/one-off debug scripts
- pure unit tests are discoverable and grouped by module ownership
- `query_pipeline.py` is primarily orchestration
- judges and grounding no longer combine unrelated policy/runtime concerns
- public API, artifact names, config defaults, and saved schema formats remain
  backward compatible
- every refactor is verified offline first and by the smallest relevant eval set
  before a full regression
