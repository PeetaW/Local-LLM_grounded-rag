# Local AI repair report

Date: 2026-11-07 Asia/Taipei

## Outcome

The Conda shell, original production `llm_env`, Ollama model path and pull workflow, GPU inference, Git checkout, original RAG, and the isolated Research Agent environment were recovered and verified after a full Windows reboot. Both environments pass `pip check`; both APIs start successfully; fresh CMD and PowerShell sessions preserve the repaired settings. Every environment acceptance item in the handoff is now satisfied. This clears the environment prerequisite for a new V12.1 Baseline run, but does not mean that benchmark has passed.

## Changes made

- Backed up all changed user, Conda, registry, and launcher files under `.local_recovery_backup/20261107-2034/`.
- Replaced stale user Conda package/environment paths with `G:\Anaconda\pkgs` and `G:\Anaconda\envs`.
- Appended `G:\Anaconda\condabin`, `G:\Git\cmd`, and `G:\nodejs` to the user PATH without overwriting existing PATH entries.
- Updated `OLLAMA_MODELS` to `G:\HuggingFace\ollama_models` and `CLAUDE_CODE_GIT_BASH_PATH` to `G:\Git\bin\bash.exe`.
- Corrected the Ollama App's own Settings > Model location from the stale E-drive path to `G:\HuggingFace\ollama_models`; the App setting had overridden the repaired user environment variable after reboot.
- Rebuilt the broken Conda 25.11.1 launcher from its installed entry point; no Conda package was reinstalled or upgraded.
- Ran official Conda initialization for CMD and PowerShell.
- Set Windows PowerShell CurrentUser execution policy to `RemoteSigned`; system policy was not changed.
- Rebuilt 92 broken `llm_env` console launchers from installed package metadata; package versions were unchanged.
- Cloned the complete pre-repair environment to `G:\Anaconda\envs\llm_env_pre_repair_20261107` before changing production packages.
- Repaired the original `llm_env` with targeted compatible versions of LlamaIndex OpenAI support, Pandas, PyArrow, and Setuptools.
- Removed stale OpenAI-agent add-ons plus unused `open-webui` and `memsearch` packages from production after confirming that the repository does not import them. They remain recoverable in the pre-repair clone.
- Created `G:\Anaconda\envs\research_agent_env` and installed the Research Agent and research-tools requirements there.
- Restarted Ollama after verifying the exact process and model directory.
- Changed orphan-index cleanup to retain and warn instead of deleting index directories.
- Added health checks, launchers, setup instructions, and before/after audit records.
- Updated README startup examples to use `start_rag.cmd` instead of an E-drive path.

## Verification summary

- Conda: PASS in new CMD, Windows PowerShell, and PowerShell 7.
- llm_env: PASS for Python, SSL, core imports, CUDA, application execution, and dependency consistency.
- Ollama: PASS after reboot for CLI, API, 11-model inventory, 50/50 manifest references, and GPU inference returning `POST_REBOOT_OLLAMA_OK`.
- Ollama pull: PASS for `qwen2.5:14b`; its 9.0 GB blobs and SHA-256 digest were verified, the manifest was written, and a post-pull run returned `OLLAMA_PULL_OK`.
- RAG: PASS after reboot for corpus health, 30 canonical index loads, API health, model list, and Q01 HTTP 200 in 391.7 seconds with grounding 1.000 and no repair/fallback. The smoke answer omitted two requested comparison values, so this is not a quality benchmark result.
- Research Agent: PASS after reboot in its isolated environment; `pip check` passed, the full 63-test run completed OK with one optional symlink test skipped, the 8/8 deterministic benchmark passed with zero false successes, and its local `/health` endpoint returned `ready`.
- Full V12.1 Benchmark: NOT RUN.

## Known limitations

1. Historical evaluation JSON and example-only documents still contain E-drive paths. They are records/examples, not active runtime configuration, and were intentionally not mass-rewritten.
2. The pre- and post-reboot smoke queries used the normal API and therefore added two episodic-memory records in total.
3. The post-reboot Q01 smoke answer did not include the requested 99.2 and 34.2 nM comparison values despite grounding 1.000; environment acceptance passes, but answer quality still requires the separate V12.1 evaluation.
4. The V12.1 Baseline was not run as part of environment recovery and still requires its own fresh evaluation and release-gate review.
