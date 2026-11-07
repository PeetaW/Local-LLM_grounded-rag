# Environment audit before repair

Audit time: 2026-11-07 20:34 Asia/Taipei

| Area | Status before repair | Evidence |
|---|---|---|
| Windows | PASS | Windows build `10.0.26300.9711`; edition query was blocked by the host. |
| PowerShell | WARN | PowerShell 7.6.9 was available; Windows PowerShell used the default Restricted policy and could not load a profile. |
| G drive | PASS | `G:` had about 1402.50 GB free. |
| Conda base | FAIL | `G:\Anaconda` existed and Base Python 3.13.11 worked, but `G:\Anaconda\condabin\conda.bat --version` returned `The system cannot execute the specified program.` |
| Conda launcher | FAIL | `G:\Anaconda\Scripts\conda.exe` was a relocated launcher; `python -m conda` worked and reported Conda 25.11.1 at `G:\Anaconda`. |
| Conda configuration | FAIL | `C:\Users\User\.condarc` still used `E:\Anaconda\pkgs` and `E:\Anaconda\envs`; `E:\Anaconda` did not exist. |
| Shell initialization | FAIL | Conda was absent from PATH, CMD AutoRun was absent, and both user PowerShell profiles were absent. |
| llm_env Python | PASS | `G:\Anaconda\envs\llm_env\python.exe` ran Python 3.11.14 with OpenSSL 3.0.19. |
| llm_env launchers | FAIL | 89 of 95 console launchers still embedded `E:\Anaconda\envs\llm_env`; `pip.exe` also failed to start. |
| Python dependencies | WARN | Core imports worked, but `pip check` reported 11 version conflicts across datasets/pyarrow and mixed LlamaIndex/OpenAI/Pandas/Setuptools versions. |
| GPU | PASS | RTX 3090, 24 GB VRAM; driver 595.79; driver CUDA 13.2. |
| Ollama executable/API | WARN | Ollama 0.40.15 and the API were reachable, but `ollama list` was empty. |
| Ollama model path | FAIL | User `OLLAMA_MODELS` pointed to `E:\HuggingFace\ollama_models`; complete data was at `G:\HuggingFace\ollama_models`. |
| Ollama data | PASS | G contained 11 manifests, 44 blobs, about 167.07 GB; all 50 manifest-to-blob references existed. |
| Git | PASS | Repository and worktrees were readable; active branch was `feature/research-agent-adaptive-iteration`; the worktree was initially clean. |
| RAG data | PASS | 31 boron BNCT PDFs, 6 ZVI PDFs, both index trees, 14 MemoryDB files, and 5309 evaluation files were present. |
| RAG startup safety | FAIL | Startup would delete the incomplete index directory for a duplicate PDF through automatic orphan cleanup. |
| Research Agent | FAIL | Production `llm_env` lacked matplotlib and had MCP 1.26.0, while the branch requires MCP 2.x in a separate environment. |

Backups were created before mutation under `.local_recovery_backup/20261107-2034/`, including user Conda config, HKCU environment registry, Conda initialization files, the original Conda launcher, llm_env launchers, and Conda history.
