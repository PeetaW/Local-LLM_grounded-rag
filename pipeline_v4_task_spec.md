## Summary

Current indexing is doing several high-value but expensive tasks in one synchronous path:

- PDF text extraction
- image extraction
- VL description generation
- chunking / embeddings
- chunk summaries
- repair / rerun handling

This gives good answer quality, but it also couples together:

- time-to-first-usable-index
- image/VL failure handling
- rebuild safety
- overall system latency and maintainability

The next iteration should split the pipeline into clear stages so a paper can become **searchable quickly**, then be **enriched incrementally**.

## Goals

1. Reduce time-to-first-searchable-paper.
2. Make VL/image failures non-blocking.
3. Make rerun / repair flows local and safe.
4. Lower average query latency by reducing unnecessary heavy LLM/NLI work.
5. Improve maintainability with explicit pipeline states.

## Proposed Pipeline

### Fast path
Used to get a paper into a searchable state as quickly as possible.

- compute paper fingerprint (`filename + hash + mtime`)
- extract PDF text
- trim references / normalize metadata
- chunk text
- create embeddings
- build base index

Result: the paper is searchable even before VL or summary enrichment finishes.

### Slow path
Used for expensive enrichments that should not block indexing.

- select image candidates worth processing
- run VL on high-value images only
- convert successful image descriptions into retrievable nodes
- incrementally update the paper index

### Enrichment path
Optional quality improvements after the paper is already usable.

- chunk summaries
- paper-level summary / routing metadata
- retrieval optimization metadata

### Repair path
Used only for failures and partial reruns.

- track `needs_review`, `skipped`, `timeout`, `last_error`
- rerun only failed or selected images
- rebuild only the affected paper incrementally
- avoid global startup side effects during repair commands

## Architectural Notes

### 1. Split "base index" from "enriched index"
A paper should not wait for VL and summary generation before it becomes searchable.

### 2. Add per-paper state tracking
Suggested fields:

- `text_index_ready`
- `vl_pending`
- `vl_partial`
- `summary_ready`
- `needs_review_count`
- `last_successful_build`
- `paper_fingerprint`
- `vl_prompt_version`
- `summary_prompt_version`

### 3. Make rebuilds transactional
Current repair/reindex flow should avoid "delete old index first, then try rebuilding".

Suggested approach:

- build into a temp directory
- validate success
- atomically replace old index

### 4. Isolate repair CLI from full app startup
Repair commands like `--rerun-vl` should not initialize the whole app or scan/clean unrelated papers before starting work.

### 5. Add image pre-filtering before VL
Skip low-value images such as:

- logos
- decorative graphics
- tiny images
- duplicate/repeated images
- non-informative assets

Prioritize:

- charts
- tables
- figures with captions
- schematics
- experimental diagrams

### 6. Introduce paper-level routing
Before chunk retrieval across the full corpus, first select candidate papers using paper-level metadata/summary vectors.

### 7. Add top-k gating for NLI / grounding
Run expensive grounding only on a very small reranked candidate set.

### 8. Add early-exit query paths
If evidence confidence is already high, skip deeper verification layers.

### 9. Support incremental builds
If PDF text, images, prompts, and model choices have not changed, do not recompute downstream artifacts.

### 10. Add instrumentation
Track at least:

- text extraction time
- image count
- VL success rate
- average VL latency per image
- chunk count
- summary generation time
- rebuild success/failure reason
- query latency by stage

## Prioritized Action List

### High impact / do first
1. Split indexing into base-index and enrichment stages.
2. Isolate `--rerun-vl` and other repair flows from full startup.
3. Make single-paper reindex transactional.
4. Add fingerprint-based incremental rebuild checks.

### Medium-term
5. Add image candidate filtering before VL.
6. Add paper-level routing index.
7. Restrict NLI/grounding to tiny top-k candidate sets.
8. Add early-exit logic for high-confidence queries.

### Observability / long-term stability
9. Add per-paper pipeline state files.
10. Add timing and failure metrics for indexing and query stages.

## Acceptance Criteria

- A newly added paper becomes searchable without waiting for VL enrichment.
- VL failures no longer block base indexing.
- `--rerun-vl` only affects the target paper.
- Failed rebuilds do not destroy the last good index.
- Re-running indexing on unchanged papers skips redundant work.
- Query latency improves for common cases without lowering grounding quality.

## Why this matters

This project’s core idea is strong: ingest full papers as multi-modal knowledge rather than plain PDF text only. The next big step is not adding more heavy stages, but separating **must-have ingestion**, **quality enrichment**, and **repair workflows** so the system becomes faster, safer, and easier to evolve.