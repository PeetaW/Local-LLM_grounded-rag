# API Refactor Spec

## Summary

Refactor `api.py` into a thinner HTTP transport layer with clearer service boundaries, safer startup behavior, and lower maintenance cost.

The current `api.py` is only about 400 lines, but it already mixes multiple responsibilities:

- FastAPI app creation and route registration
- Pydantic request/response schemas
- Session ID validation and in-memory session state
- Prompt-injection guardrails
- Memory recall and preference persistence
- Query pipeline orchestration
- OpenAI-compatible streaming response formatting
- Import-time dependency on `main.py` global state

The main problem is not just file length. It is that transport concerns, runtime lifecycle, and business orchestration are tightly coupled in one file.

---

## Background

Current risks observed in the project:

1. `api.py` imports `paper_engines` and memory collections directly from `main.py`.
2. Starting the API can therefore trigger full app initialization, index loading, and cleanup side effects.
3. Streaming requests create a per-request thread pool with no explicit lifecycle management.
4. Session handling, preference writes, and query execution are mixed directly inside route handlers.
5. OpenAI-compatible transport formatting is coupled to query orchestration logic.

These issues make the API harder to test, slower to reason about, and riskier to operate as the project grows.

---

## Goals

1. Make `api.py` a thin entry layer.
2. Remove API import-time dependence on `main.py` global state.
3. Separate HTTP transport logic from query/memory orchestration.
4. Centralize startup and runtime dependency loading behind an explicit app state container.
5. Isolate streaming formatting and executor management.
6. Keep Open WebUI / OpenAI-compatible endpoints behaviorally stable.
7. Keep each API-related file small enough for practical maintenance.

---

## Non-Goals

This refactor should not try to solve everything at once.

- Do not redesign the whole query pipeline in this task.
- Do not replace FastAPI.
- Do not introduce a database-backed session system yet.
- Do not rewrite memory logic internals unless required for decoupling.
- Do not change endpoint contracts unless compatibility forces it.
- Do not combine this work with the indexing/rebuild safety refactor.

---

## Design Principles

1. Explicit dependencies over import-time globals.
2. Thin routes, thicker services.
3. Single-purpose modules.
4. Shared lifecycle for long-lived resources.
5. Clear separation between transport, orchestration, and infrastructure helpers.
6. Keep OpenAI-compatible response formatting isolated from business logic.

---

## Proposed Module Layout

```text
rag_project/
  api.py
  api_app_state.py
  api_models.py
  api_sessions.py
  api_guardrails.py
  api_services.py
  api_streaming.py
  api_routes.py
```

### Module Responsibilities

#### `api.py`

Keep only:

- FastAPI app creation
- app startup/shutdown hook registration
- router registration

Should not contain:

- query orchestration
- session mutation logic
- streaming chunk assembly
- `main.py` global imports

#### `api_app_state.py`

Responsible for:

- explicit runtime dependency container
- loading and exposing:
  - `paper_engines`
  - `episodic_collection`
  - `preference_collection`
- startup initialization entrypoints
- optional shared executor lifecycle

Suggested shape:

- `ApiRuntimeState` dataclass or simple class
- `build_runtime_state()`
- `get_runtime_state()`

#### `api_models.py`

Move all request/response models here:

- `QueryRequest`
- `QueryResponse`
- `ChatMessage`
- `ChatCompletionRequest`

Optional:

- helper response chunk schemas for readability

#### `api_sessions.py`

Move session logic here:

- `SESSION_MAX_TURNS`
- `SESSION_MAX_COUNT`
- `session_store`
- `_validate_session_id()`
- `_resolve_session_id()`
- helper functions for session read/write and trimming

Optional helpers:

- `build_short_term_context()`
- `store_turn()`
- `trim_session_store()`

#### `api_guardrails.py`

Move input validation and safety checks here:

- prompt injection regex patterns
- `_check_prompt_injection()`
- future request validation helpers

#### `api_services.py`

This is the main orchestration layer.

Responsible for:

- taking app state + request data
- collecting short-term memory
- recalling episodic / preference memory
- preference detection and persistence
- invoking query pipeline
- post-processing answer
- returning a structured service result

This layer should know about:

- query engine / query pipeline
- memory system
- answer post-processing

This layer should not know about:

- FastAPI request objects
- SSE chunk formatting details

#### `api_streaming.py`

Responsible for:

- shared executor strategy
- streaming generator lifecycle
- OpenAI-compatible SSE chunk packaging
- status chunk emission rules
- stream finalization / error framing

This layer should not do:

- memory recall
- session persistence policy
- prompt injection checks

#### `api_routes.py`

Optional but recommended.

Use this if you want to keep route handlers grouped away from app construction.

Responsible for:

- `/health`
- `/query`
- `/v1/models`
- `/v1/chat/completions`

Each route should be a thin adapter:

- parse request
- call service
- shape HTTP response

---

## File Size Guideline

To avoid repeating the `query_engine.py` situation:

- target size per file: `150-400` lines
- soft limit: `500` lines
- hard limit: `700` lines

If any API-related file grows beyond `500` lines, split by responsibility before adding more logic.

---

## Key Refactor Tasks

| ID | Task | Description | Output | Acceptance Criteria |
|---|---|---|---|---|
| API-01 | Establish runtime state boundary | Create explicit app runtime state instead of importing from `main.py` | `api_app_state.py` | API no longer imports `paper_engines` or memory collections from `main.py` |
| API-02 | Extract API models | Move all Pydantic models into a dedicated module | `api_models.py` | `api.py` no longer defines request/response schemas inline |
| API-03 | Extract session management | Move session validation/store logic into its own module | `api_sessions.py` | session helpers are reusable and independently testable |
| API-04 | Extract guardrails | Move prompt-injection checks out of routes | `api_guardrails.py` | route handlers no longer embed regex guard logic |
| API-05 | Create service orchestration layer | Build a service layer for query + memory orchestration | `api_services.py` | route handlers only adapt HTTP to service calls |
| API-06 | Extract streaming transport | Isolate SSE/OpenAI chunk formatting and generator lifecycle | `api_streaming.py` | streaming logic no longer lives inline in `/v1/chat/completions` |
| API-07 | Fix streaming executor lifecycle | Replace per-request unmanaged executors with shared or properly closed executor handling | shared executor or managed lifecycle | repeated streaming requests do not leak worker threads |
| API-08 | Thin out route layer | Keep `api.py` or `api_routes.py` focused on endpoint definitions only | thin route module | main API file becomes easy to scan |
| API-09 | Add structured logging points | Add stage-level logs for request path, stream lifecycle, and service outcome | unified logging | failures are traceable by stage |
| API-10 | Add regression checklist | Define manual smoke tests for query and chat compatibility | test checklist | behavior can be checked after refactor |

---

## Recommended Refactor Order

To minimize risk, refactor in this order:

1. `API-02` Extract models
2. `API-03` Extract session logic
3. `API-04` Extract guardrails
4. `API-06` Extract streaming helpers
5. `API-05` Introduce service layer
6. `API-01` Introduce runtime state boundary
7. `API-07` Fix executor lifecycle
8. `API-08` Thin route layer
9. `API-09` Add structured logs
10. `API-10` Finalize regression checklist

Why this order:

- models, sessions, and guardrails are low-risk extractions
- streaming is self-contained enough to isolate early
- service orchestration should be extracted before lifecycle changes
- decoupling from `main.py` is the most important change, but it is safer once the surrounding boundaries already exist

---

## Runtime State Proposal

The API should depend on an explicit runtime object rather than module-level globals.

Example concept:

```python
@dataclass
class ApiRuntimeState:
    paper_engines: dict
    episodic_collection: object
    preference_collection: object
    stream_executor: ThreadPoolExecutor | None = None
```

This runtime state should be:

- created during FastAPI startup
- stored on `app.state`
- injected into route/service helpers

This prevents API boot from implicitly executing unrelated CLI startup logic.

---

## Service Layer Proposal

Define a service result object so the route handlers do not assemble everything themselves.

Suggested service methods:

- `handle_query(request, runtime_state, session_store)`
- `handle_chat_completion(request, runtime_state, session_store)`
- `build_memory_context(...)`
- `handle_preference_write(...)`

Possible service result fields:

- `answer`
- `session_id`
- `status_messages`
- `grounding_score`
- `should_stream`

The exact schema can stay simple, but routes should receive a structured result rather than orchestrating everything inline.

---

## Streaming Refactor Requirements

The current streaming path is one of the heaviest and most fragile API responsibilities.

Minimum requirements:

1. Do not create unmanaged thread pools per request.
2. Clearly define when the worker starts and ends.
3. Keep SSE chunk formatting separate from query execution.
4. Have one place for:
   - initial assistant delta
   - status chunk emission
   - content chunk emission
   - final stop chunk
   - error chunk

Recommended options:

- preferred: shared executor on app state
- acceptable: per-request executor with explicit `shutdown()`

---

## Logging and Observability Spec

Add logs that help answer:

- did the request reach the API correctly
- which path was taken: `/query`, `/v1/chat/completions`, stream or non-stream
- was prompt injection blocked
- was this a preference write
- did memory recall succeed
- did query execution complete
- did streaming complete cleanly

Suggested log fields:

- `request_id`
- `session_id`
- `route`
- `stream`
- `stage`
- `elapsed_ms`
- `status`
- `error_type`

Suggested stages:

- `request_received`
- `guardrails_checked`
- `session_resolved`
- `memory_loaded`
- `query_executed`
- `post_processed`
- `stream_started`
- `stream_completed`
- `stream_failed`

---

## Regression Checklist

After refactor, manually validate at least:

1. `/health` returns healthy status and expected counts
2. `/query` normal question path works
3. `/query` prompt injection block still works
4. preference message path still stores preference and short-circuits correctly
5. session ID reuse works across multiple turns
6. `/v1/models` remains Open WebUI compatible
7. `/v1/chat/completions` non-streaming path works
8. `/v1/chat/completions` streaming path works
9. repeated streaming requests do not steadily increase worker thread count
10. API startup no longer requires importing runtime globals from `main.py`

---

## Definition of Done

This refactor is complete when:

1. `api.py` is a thin app/route entry file
2. API no longer imports runtime state directly from `main.py`
3. session, guardrails, models, services, and streaming logic are split into dedicated modules
4. streaming executor lifecycle is explicit and safe
5. OpenAI-compatible endpoints still behave correctly
6. no API-related file exceeds the hard limit
7. route handlers are small enough to read in one pass
8. manual smoke checks pass

---

## Follow-Up Work

Once this refactor is done, the next natural follow-ups are:

1. connect API startup to a safer app bootstrap path separate from CLI startup
2. unify query pipeline import usage so API does not depend on deprecated query-engine entrypoints
3. add basic tests for session handling and streaming chunk formatting
4. consider replacing in-memory session store with a bounded pluggable backend if usage grows

---

## Notes

This spec is intentionally focused on decoupling and maintainability first.

If behavior must change, prefer:

- making dependencies explicit
- reducing side effects
- preserving endpoint compatibility

over adding new features during the same refactor.
