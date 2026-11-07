param()

$ErrorActionPreference = "Continue"
$failures = 0

function Pass([string]$Message) { Write-Host "[PASS] $Message" }
function Warn([string]$Message) { Write-Host "[WARN] $Message" }
function Fail([string]$Message) { $script:failures++; Write-Host "[FAIL] $Message" }

Push-Location $PSScriptRoot
try {
    $condaCommand = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $condaCommand) {
        Fail "Conda is not available in this PowerShell session. Open a new terminal."
        exit $failures
    }

    $condaVersion = (conda --version 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -eq 0) { Pass $condaVersion } else { Fail $condaVersion }

    $envInfo = conda info --envs --json 2>$null | ConvertFrom-Json
    $llmEnv = @($envInfo.envs | Where-Object { (Split-Path $_ -Leaf) -eq "llm_env" })[0]
    if (-not $llmEnv) {
        Fail "llm_env was not found."
        exit $failures
    }

    $python = Join-Path $llmEnv "python.exe"
    if (-not (Test-Path -LiteralPath $python)) {
        Fail "llm_env Python is missing: $python"
        exit $failures
    }

    $pythonInfo = (& $python -c "import ssl,sys; print(sys.version.split()[0]+' | '+sys.executable+' | '+ssl.OPENSSL_VERSION)" 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -eq 0) { Pass $pythonInfo } else { Fail $pythonInfo }

    $cudaRaw = & $python -c "import json,torch; print(json.dumps({'available':torch.cuda.is_available(),'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))" 2>&1
    if ($LASTEXITCODE -eq 0) {
        $cuda = $cudaRaw | ConvertFrom-Json
        if ($cuda.available) { Pass "PyTorch $($cuda.torch), CUDA $($cuda.cuda), $($cuda.device)" } else { Fail "PyTorch cannot use CUDA." }
    } else {
        Fail ($cudaRaw | Out-String).Trim()
    }

    $ollamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
    $tags = $null
    if (-not $ollamaCommand) {
        Fail "Ollama is not on PATH."
    } else {
        $ollamaVersion = (& ollama --version 2>&1 | Out-String).Trim()
        if ($LASTEXITCODE -eq 0) { Pass $ollamaVersion } else { Fail $ollamaVersion }
        try {
            $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 10
            Pass "Ollama API returned $(@($tags.models).Count) models."
        } catch {
            Fail "Ollama API is unavailable at 127.0.0.1:11434."
        }
    }

    $modelRoot = [Environment]::GetEnvironmentVariable("OLLAMA_MODELS", "User")
    if (-not $modelRoot) { $modelRoot = $env:OLLAMA_MODELS }
    if ($modelRoot -and (Test-Path -LiteralPath (Join-Path $modelRoot "blobs")) -and (Test-Path -LiteralPath (Join-Path $modelRoot "manifests"))) {
        Pass "Ollama model storage: $modelRoot"
    } else {
        Fail "OLLAMA_MODELS does not point to a complete model directory."
    }

    $preflightCode = @'
import json
from pathlib import Path
import config as cfg
from rag.corpus_health import duplicate_skip_set

papers = sorted(Path(cfg.PAPERS_DIR).glob("*.pdf"))
skipped = duplicate_skip_set([p.name for p in papers])
canonical = [p for p in papers if p.name not in skipped]
missing = [p.name for p in canonical if not (Path(cfg.INDEX_BASE_DIR) / p.stem / "docstore.json").is_file()]
config_path = Path(cfg.INDEX_CONFIG_PATH)
stored = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None
print(json.dumps({
    "active_project": cfg.ACTIVE_PROJECT,
    "papers": len(papers),
    "canonical_papers": len(canonical),
    "duplicates_skipped": len(skipped),
    "missing_indexes": missing,
    "index_config_matches": stored == cfg.INDEX_BUILD_CONFIG,
    "required_models": [cfg.LLM_MODEL, cfg.PLANNING_LLM_MODEL, cfg.SYNTHESIS_MODEL, cfg.VERIFY_MODEL, cfg.VL_MODEL, cfg.EMBED_MODEL],
}))
'@
    $preflightRaw = $preflightCode | & $python - 2>&1
    if ($LASTEXITCODE -ne 0) {
        Fail ($preflightRaw | Out-String).Trim()
    } else {
        $preflight = $preflightRaw | ConvertFrom-Json
        if ($preflight.index_config_matches -and @($preflight.missing_indexes).Count -eq 0) {
            Pass "RAG project '$($preflight.active_project)': $($preflight.canonical_papers) indexes ready; $($preflight.duplicates_skipped) duplicate skipped."
        } else {
            Fail "RAG indexes are incomplete or the index configuration changed."
        }
        if ($tags) {
            $installed = @($tags.models | ForEach-Object { $_.name })
            $missingModels = @($preflight.required_models | Sort-Object -Unique | Where-Object {
                $_ -notin $installed -and ("{0}:latest" -f $_) -notin $installed
            })
            if ($missingModels.Count -eq 0) { Pass "All configured Ollama models are installed." } else { Fail "Missing Ollama models: $($missingModels -join ', ')" }
        }
    }

    if (Get-Command git -ErrorAction SilentlyContinue) {
        $gitStatus = (& git status --short --branch 2>&1 | Out-String).Trim()
        Pass "Git repository is accessible."
        if (($gitStatus -split "`r?`n").Count -gt 1) { Warn "The worktree has local changes; do not overwrite them." }
    } else {
        Fail "Git is not on PATH."
    }

    $pipCheck = & $python -m pip check 2>&1
    if ($LASTEXITCODE -eq 0) {
        Pass "pip check found no dependency conflicts."
    } else {
        Fail "pip check reports dependency conflicts."
        $pipCheck | ForEach-Object { Write-Host "       $_" }
    }
} finally {
    Pop-Location
}

if ($failures -eq 0) { Write-Host "Environment check completed with no blocking failures." }
exit $failures
