# Windows local setup and daily use

## Daily RAG start

Open a new CMD or PowerShell window, then run:

```powershell
cd G:\Projects\rag_project
.\check_environment.ps1
.\start_rag.cmd
```

The API listens on `http://127.0.0.1:8000`. Check it with:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/v1/models
```

Stop the service with `Ctrl+C`. The launcher forces UTF-8 because the project prints characters that fail under the default CP950 code page.

## Ollama

Run the read-only check:

```powershell
.\check_ollama.ps1
ollama list
ollama ps
```

Models are stored in `G:\HuggingFace\ollama_models`. To download another model, first confirm the exact official tag, download size, and free G-drive space. Only then run, with explicit approval:

```powershell
ollama pull <official-model-tag>
```

Do not delete the existing `blobs` or `manifests` directories to troubleshoot a pull.

The Ollama App also stores a **Model location** setting. It must match `OLLAMA_MODELS`. If `ollama list` becomes empty after a reboot while the G-drive blobs still exist, open Ollama **Settings**, set **Model location** to `G:\HuggingFace\ollama_models`, then fully quit and restart the App.

## Research Agent isolation

Research Agent dependencies are installed in the existing `research_agent_env`; do not merge them into `llm_env`. To recreate that environment later, use:

```powershell
conda create -n research_agent_env python=3.11
conda activate research_agent_env
python -m pip install -r requirements-research-agent.txt -r requirements-research-tools.txt
```

Before starting the local API, define a private workspace and a secret token of at least 24 characters:

```powershell
$env:RESEARCH_AGENT_WORKSPACE = "G:\PrivateResearch"
$env:RESEARCH_AGENT_TOKEN = "replace-with-a-long-random-local-token"
.\start_research_agent.cmd
```

Optional local model planning:

```powershell
$env:RESEARCH_AGENT_ENABLE_OLLAMA = "1"
$env:RESEARCH_AGENT_MODEL = "qwen3.5:27b"
```

The Research Agent API binds only to `http://127.0.0.1:8765`.

## Offline Research Agent validation

```powershell
$env:TEMP = "G:\Projects\rag_project\.tmp\research_agent_tests"
$env:TMP = $env:TEMP
conda activate research_agent_env
python -m unittest discover -s tests -p "test_research_agent*.py" -v
python -m eval.research_agent_eval --output eval\results\agent_offline_benchmark.json
```

## Updating safely

- Do not upgrade all packages in `llm_env`.
- Save `python -m pip check`, `conda list`, and a fresh RAG smoke result before changing dependencies.
- Do not delete PDFs, indexes, MemoryDB, evaluation results, or Ollama models.
- Do not run a full V12.1 Benchmark until `environment_audit_after.md` has no blocking items.

## Rollback information

Pre-repair files are under `.local_recovery_backup/20261107-2034/`, and a complete environment clone is at `G:\Anaconda\envs\llm_env_pre_repair_20261107`. Restoring registry, profiles, Conda launchers, or environments changes system behavior; review `repair_report.md` before restoring anything.
