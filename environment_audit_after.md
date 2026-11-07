# Environment audit after repair

Audit date: 2026-11-07 Asia/Taipei

| Acceptance item | Result | Evidence |
|---|---|---|
| CMD runs `conda --version` | PASS | New CMD returned Conda 25.11.1. |
| PowerShell runs `conda --version` | PASS | New Windows PowerShell and PowerShell 7 returned Conda 25.11.1. |
| `conda activate llm_env` | PASS | All tested shells resolved Python first to `G:\Anaconda\envs\llm_env\python.exe`. |
| Python/SSL/core imports | PASS | Python 3.11.14, OpenSSL 3.0.19; NumPy, PyTorch, LlamaIndex, Ollama client, HTTPX, Sentence Transformers, ChromaDB, PyMuPDF, FastAPI, Uvicorn, SciPy, and Pandas imported. |
| `pip check` | PASS | Original `llm_env` now reports `No broken requirements found.` after a targeted dependency repair. |
| PyTorch/CUDA | PASS | PyTorch 2.10.0+cu128 reported CUDA available on RTX 3090. |
| llm_env console launchers | PASS | 89 relocated entry-point launchers plus pip, pip3, and wheel were regenerated offline and backed up. Uvicorn 0.41.0, pytest 9.0.2, pip 26.0.1, and wheel 0.46.3 launched. |
| Ollama CLI | PASS | Ollama 0.40.15; `ollama list` returned all 11 existing models. |
| Ollama API | PASS | `/api/tags` returned 11 models from `G:\HuggingFace\ollama_models`. |
| Existing model inference | PASS | `qwen2.5:14b` returned `LOCAL_AI_OK`; no model download occurred. |
| Post-repair LlamaIndex inference | PASS | The production `OpenAILike` client called `qwen2.5:14b` and returned `LLAMA_INDEX_OK`. |
| `ollama pull` download | PASS | After confirming the existing `qwen2.5:14b` tag (9.0 GB) and 1395.14 GB free on G:, `ollama pull qwen2.5:14b` verified all blobs, SHA-256 digest, and wrote the manifest successfully. |
| GPU/model runtime | PASS | The local model smoke test completed; `ollama ps` was callable. |
| Git repository/worktrees | PASS | Repository, branch, remote, and both worktree records were readable. |
| Private PDFs/indexes | PASS | Files remained local and ignored by Git; no PDF or index was deleted. |
| RAG health scan | PASS | `python main.py --health` completed against 31 PDFs. |
| RAG API startup | PASS | 30 canonical indexes loaded; `/health` and `/v1/models` returned HTTP 200. |
| RAG end-to-end smoke | PASS | Q01 returned HTTP 200 in 424.3 seconds; grounding score 1.000; no repair/fallback. This was not a benchmark result. |
| RAG memory side effect | WARN | The normal `/query` path added one episodic-memory entry, as designed; it was not deleted. |
| Research Agent environment | PASS | Independent Python 3.11 environment created at `G:\Anaconda\envs\research_agent_env`; `pip check` passed with MCP 2.3.0 and Matplotlib 3.11.2. |
| Research Agent unit tests | PASS | All 33 Agent-specific tests passed. The full 63-test run completed OK; one optional Windows symlink test was skipped because symlink creation was unavailable. |
| Research Agent benchmark | PASS | Deterministic offline benchmark passed 8/8 with zero false successes; scientific correctness was outside this synthetic benchmark. |
| Research Agent API startup | PASS | `start_research_agent.cmd` started the local service and `/health` returned `ready`. |
| Restart persistence | PASS | Fresh CMD and PowerShell sessions both found Conda and Ollama, activated `llm_env`, and resolved Python to `G:\Anaconda\envs\llm_env\python.exe`. This satisfies the handoff's reboot-or-new-terminal criterion. |
| V12.1 Baseline readiness | PASS | Every environment acceptance item in the handoff is now satisfied. The V12.1 Baseline itself has not been run and remains a separate evaluation stage. |

Current important paths:

- Conda base: `G:\Anaconda`
- Production environment: `G:\Anaconda\envs\llm_env`
- Pre-repair environment clone: `G:\Anaconda\envs\llm_env_pre_repair_20261107`
- Research Agent environment: `G:\Anaconda\envs\research_agent_env`
- Repository: `G:\Projects\rag_project`
- Ollama executable: `C:\Users\User\AppData\Local\Programs\Ollama\ollama.exe`
- Ollama models: `G:\HuggingFace\ollama_models`
- Active project: `projects/boron_bnct`
- RAG API: `http://127.0.0.1:8000`
- Research Agent API after separate setup: `http://127.0.0.1:8765`
