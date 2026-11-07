@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

set "CONDA_BASE="
for /f "delims=" %%I in ('conda info --base 2^>nul') do set "CONDA_BASE=%%I"
if not defined CONDA_BASE (
  echo [FAIL] Conda is not available. Close this window and open a new terminal.
  exit /b 1
)

call "%CONDA_BASE%\Scripts\activate.bat" research_agent_env
if errorlevel 1 (
  echo [FAIL] research_agent_env is not installed. See README_LOCAL_SETUP.md.
  exit /b 1
)

if not defined RESEARCH_AGENT_WORKSPACE (
  echo [FAIL] Set RESEARCH_AGENT_WORKSPACE to a private local folder.
  exit /b 1
)
if not exist "%RESEARCH_AGENT_WORKSPACE%\" (
  echo [FAIL] RESEARCH_AGENT_WORKSPACE does not exist.
  exit /b 1
)
if not defined RESEARCH_AGENT_TOKEN (
  echo [FAIL] Set RESEARCH_AGENT_TOKEN to a secret value with at least 24 characters.
  exit /b 1
)

python -c "import fastapi, matplotlib, mcp, pptx"
if errorlevel 1 (
  echo [FAIL] Research Agent dependencies are incomplete.
  exit /b 1
)

echo Starting Research Agent at http://127.0.0.1:8765
python -m rag.research_agent.http_app
