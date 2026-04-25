# Query Engine Refactor Spec

## Summary

This spec defines a maintainable refactor plan for [rag/query_engine.py](/C:/Users/User/.codex/worktrees/56b1/rag_project/rag/query_engine.py:1).

The current module has grown into a core "god file" that mixes:

- query planning
- paper routing
- retrieval orchestration
- embedding guard and retry logic
- answer synthesis
- grounding fallback
- answer verification handoff
- translation
- streaming and non-streaming execution flows

The goal of this refactor is to improve maintainability, reduce debug difficulty, and prevent future file growth from turning the query pipeline into an even harder-to-change bottleneck.

## Background

Current issues observed in `rag/query_engine.py`:

1. Too many responsibilities are coupled in a single module.
2. Streaming and non-streaming flows appear to duplicate substantial orchestration logic.
3. Debug helpers and infrastructure workarounds live next to business logic.
4. Stage boundaries are not explicit, so failures are harder to localize.
5. The file is already large enough that safe iteration is becoming harder.

This is a good point to refactor before additional retrieval, grounding, and latency optimizations are added.

## Goals

1. Keep the current external behavior stable while improving internal structure.
2. Split the query pipeline into stage-oriented modules.
3. Make streaming and non-streaming paths share one core orchestration flow.
4. Improve observability so failures can be mapped to a specific stage.
5. Prevent any single script from growing into an unmaintainable giant file again.

## Non-Goals

This refactor should not expand into a full behavior rewrite.

Do not treat the following as part of the initial scope:

- redesigning prompt strategy
- replacing grounding or NLI algorithms
- introducing job queues
- changing models or inference providers
- doing a full performance optimization pass
- changing the public API contract unless strictly necessary

## Design Principles

1. Refactor structure first, optimize logic second.
2. Prefer small modules with explicit responsibilities.
3. Shared orchestration must not be duplicated across stream and non-stream modes.
4. Pipeline stages should exchange well-defined data structures instead of loose tuples everywhere.
5. Keep the public entrypoints stable while moving the implementation inward.

## File Size Policy

To reduce long-term maintenance cost, use the following policy:

- Target size per file: 300 to 600 lines
- Soft limit: 700 lines
- Hard limit: 900 lines
- If a file exceeds 700 lines, note why in the commit/PR description
- If a file exceeds 900 lines, it must be split further

This policy applies to all new modules created in this refactor.

## Proposed Module Layout

```text
rag/
  query_pipeline.py
  query_types.py
  query_planning.py
  query_retrieval.py
  query_embedding_guard.py
  query_grounding_flow.py
  query_translation.py
  query_prompts.py
  query_formatting.py
```

If a smaller first step is preferred, `query_formatting.py` can be deferred and folded into `query_pipeline.py` temporarily.

## Module Responsibilities

### `query_pipeline.py`

Primary responsibilities:

- expose the public pipeline entrypoints
- coordinate stage execution
- route between streaming and non-streaming output modes
- call shared orchestration logic

Should not contain:

- low-level embedding retry code
- planning prompt details
- grounding section parsing details
- translation prompt implementation details

### `query_types.py`

Primary responsibilities:

- define dataclasses or typed payloads shared between stages
- make stage inputs and outputs explicit
- reduce fragile tuple-based wiring

Suggested examples:

- `PlannedQuery`
- `SubqueryTask`
- `SubqueryResult`
- `PipelineContext`
- `GroundingBundle`
- `PipelineArtifacts`

### `query_planning.py`

Primary responsibilities:

- detect target paper
- select relevant papers
- plan sub-questions

Should not contain:

- retrieval execution
- answer generation
- translation

### `query_retrieval.py`

Primary responsibilities:

- prepare retrieval query text
- build subquery tasks
- run retrieval in parallel
- generate paper-level raw answers from nodes

Should not contain:

- final answer verification
- final translation
- grounding fallback section formatting

### `query_embedding_guard.py`

Primary responsibilities:

- clean text before embedding
- test embedding calls
- retry / truncate / fallback logic for embedding edge cases
- embed-specific debug diagnostics

Should not contain:

- general query planning logic
- full retrieval orchestration

### `query_grounding_flow.py`

Primary responsibilities:

- parse direct/inference/speculation sections
- partition citation results by section
- orchestrate grounding fallback behavior

Should not contain:

- paper routing
- translation
- endpoint or transport concerns

### `query_translation.py`

Primary responsibilities:

- translate final English draft to Traditional Chinese
- map section headers consistently
- preserve labels, units, formulas, and verification tags

Should not contain:

- retrieval logic
- grounding orchestration

### `query_prompts.py`

Primary responsibilities:

- build the four synthesis prompt variants: reasoning EN, reasoning ZH, strict EN, strict ZH
- accept well-defined parameters (`knowledge_base`, `question`, `memory_section`) and return a prompt string
- serve as the single place to modify prompt strategy without touching pipeline logic

Should not contain:

- LLM calls
- retrieval logic
- any pipeline orchestration

Background: the current `query_engine.py` embeds these four prompt variants twice — once in the non-streaming path and once in the streaming path. Extracting them into named builder functions eliminates the duplication and makes future prompt tuning a single-file change.

### `query_formatting.py`

Primary responsibilities:

- final answer assembly helpers
- optional rendering-oriented formatting helpers
- small text post-formatting utilities used by the pipeline

Should not contain:

- LLM calls
- retrieval logic

## Recommended Refactor Order

Use the following order to reduce risk:

1. Create spec and branch baseline
2. Introduce `query_types.py`
3. Extract `query_planning.py`
4. Extract `query_embedding_guard.py`
5. Extract `query_retrieval.py`
6. Extract `query_translation.py`
7. Extract `query_grounding_flow.py`
8. Introduce a shared internal orchestrator for stream and non-stream
9. Shrink the final entry module into `query_pipeline.py`
10. Add stage-level logging and regression checklist

Reasoning:

- planning and embedding guard are relatively separable
- retrieval is high-value and should be isolated before deeper cleanup
- grounding flow is more tightly coupled and should be extracted after the structure is clearer
- stream/non-stream unification should happen after stage modules exist

## Task Table

| ID | Task | Description | Deliverable | Acceptance Criteria |
|---|---|---|---|---|
| QE-01 | Create refactor baseline | Freeze scope and document behavior-preserving intent | branch + spec | scope, goals, and non-goals are written down |
| QE-02 | Introduce query types | Define shared typed stage payloads | `query_types.py` | stage interfaces use clearer structures |
| QE-03 | Extract planning module | Move paper routing and sub-question planning | `query_planning.py` | planning behavior remains stable |
| QE-04 | Extract embedding guard module | Move embedding cleanup/retry/debug logic | `query_embedding_guard.py` | retrieval preprocessing remains stable |
| QE-05 | Extract retrieval module | Move subquery task building and retrieval flow | `query_retrieval.py` | retrieval results do not regress |
| QE-06 | Extract grounding flow module | Move section parsing and grounding fallback logic | `query_grounding_flow.py` | grounding output shape remains stable |
| QE-07 | Extract translation module | Move final translation helper | `query_translation.py` | translation stage behavior remains stable |
| QE-07b | Extract prompt builder module | Move four synthesis prompt variants into named builder functions | `query_prompts.py` | prompt output is identical; each variant callable independently |
| QE-08 | Build shared orchestrator | Remove duplicated stream/non-stream core flow via `_run_pipeline_core` | internal runner | major duplicated orchestration is removed |
| QE-09 | Shrink public entry module | Keep only thin pipeline entrypoints; update all callers to import from `query_pipeline` | `query_pipeline.py` | public entrypoints remain stable; `query_engine.py` moved to `archive/` |
| QE-10 | Add stage logs | Add stage start/end/error/timing logs | structured log points | failures can be mapped to pipeline stage |
| QE-11 | Create regression checklist | Define representative question set | test checklist | before/after behavior can be compared |
| QE-12 | Update docs | Document module boundaries and extension rules | docs update | future changes are easier to reason about |

## Public Entry Points

The following entrypoints should stay stable unless there is a compelling reason to change them:

- `execute_structured_query(...)`
- `execute_structured_query_stream(...)`

If they must change, any affected callers must be updated in the same refactor branch.

## Stage Model

The pipeline should be made explicit as a sequence of stages:

1. planning
2. retrieval
3. synthesis
4. grounding
5. verification
6. translation
7. finalize

Each stage should ideally:

- accept a well-defined input object
- return a well-defined output object
- emit clear debug or status logs
- avoid hidden mutation where possible

### Stream / Non-stream Unification

The current `execute_structured_query` and `execute_structured_query_stream` duplicate all seven pipeline stages. The two functions differ only in two points:

- **Status reporting**: the non-streaming path calls an `on_status` callback; the streaming path yields `[STATUS] ...` strings.
- **LLM output**: the non-streaming path accumulates the full text and returns it; the streaming path yields each chunk as it arrives.

The shared orchestrator (`_run_pipeline_core`) should execute stages 1–6 and carry intermediate state via `PipelineContext`. The two public entrypoints then diverge only at the final output step:

```
_run_pipeline_core(ctx)          ← planning → retrieval → synthesis → grounding → verification → translation
  ↑                   ↑
execute_structured_query     execute_structured_query_stream
(accumulate → return str)    (yield status + yield chunks)
```

This means any change to planning, retrieval, grounding, or translation logic is made once and applies to both modes automatically.

## Logging / Debugging Spec

This refactor is partly motivated by debug difficulty, so stage-level observability is required.

Each stage should log at least:

- `stage_name`
- `question_id` or request id
- `status=ok|fallback|error`
- `elapsed_ms`
- `paper_count` when relevant
- `subquery_count` when relevant
- `error_type` when relevant

Recommended decision flags:

- `target_paper_detected`
- `grounding_enabled`
- `verification_enabled`
- `translation_applied`
- `streaming_mode`

Recommended stages to log:

- planning
- retrieval
- synthesis
- grounding
- verification
- translation
- finalize

## Regression Checklist

At minimum, validate these scenarios before and after the refactor:

1. single-paper direct query
2. multi-paper comparison query
3. no-answer / insufficient-evidence query
4. grounding success path
5. grounding fallback path
6. English draft pipeline plus Traditional Chinese translation
7. streaming output path
8. non-streaming output path
9. query with memory context present
10. query that triggers embedding cleanup or retry logic

Prefer keeping a fixed smoke-test question set for quick manual comparisons.

## Definition of Done

This refactor is complete when:

1. Public query entrypoints still work.
2. Streaming and non-streaming modes share one core orchestration flow.
3. Planning, retrieval, grounding, translation, embedding guard, and prompt building logic are separated into dedicated modules.
4. No resulting file exceeds 900 lines.
5. The public entry module (`query_pipeline.py`) is significantly smaller than the original `query_engine.py`.
6. A regression checklist exists and has been run.
7. Logs can identify the stage at which a failure occurred.
8. `query_engine.py` is moved to `archive/` as a historical reference; all active callers (`api.py`, `main.py`, `scripts/test_query.py`) import from `rag.query_pipeline`.

## Implementation Notes

Follow these working rules during the refactor:

1. Prefer move-only or minimal-change commits at first.
2. Do not optimize algorithm behavior while extracting modules unless required for correctness.
3. Keep old and new names stable where it reduces migration risk.
4. Refactor duplicated flow before tuning behavior.
5. Add thin wrapper helpers rather than spreading direct imports everywhere.

## Suggested Milestones

### Milestone 1: Structural extraction

- create `query_types.py`
- create `query_planning.py`
- create `query_embedding_guard.py`
- create `query_retrieval.py`

### Milestone 2: Orchestration cleanup

- create `query_grounding_flow.py`
- create `query_translation.py`
- create `query_prompts.py`
- unify stream/non-stream orchestration via `_run_pipeline_core`

### Milestone 3: Maintainability hardening

- add stage logging
- create regression checklist
- document extension rules

## Future Extensions After This Refactor

These should be easier after the module boundaries are cleaned up:

- paper-level routing improvements
- top-k grounding restrictions
- early-exit verification
- per-stage metrics collection
- background query analysis tooling
- more targeted unit tests

## Final Note

This refactor should be treated as a maintainability and operability milestone, not as wasted cleanup work.

The system has already reached the point where additional features will become riskier unless the query pipeline is decomposed into smaller, stage-oriented modules. Doing this now should make later optimization work faster, safer, and easier to debug.
