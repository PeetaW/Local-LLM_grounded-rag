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

call "%CONDA_BASE%\Scripts\activate.bat" llm_env
if errorlevel 1 (
  echo [FAIL] Could not activate llm_env.
  exit /b 1
)

powershell.exe -NoLogo -ExecutionPolicy Bypass -File "%~dp0check_environment.ps1"
if errorlevel 1 (
  echo [FAIL] Environment preflight failed. RAG was not started.
  exit /b 1
)

echo Starting RAG at http://127.0.0.1:8000
python -m uvicorn api:app --host 127.0.0.1 --port 8000
