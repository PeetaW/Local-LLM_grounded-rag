param()

$ErrorActionPreference = "Continue"
$failures = 0
function Pass([string]$Message) { Write-Host "[PASS] $Message" }
function Warn([string]$Message) { Write-Host "[WARN] $Message" }
function Fail([string]$Message) { $script:failures++; Write-Host "[FAIL] $Message" }

$command = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $command) {
    Fail "Ollama is not on PATH."
    exit $failures
}

Pass "Ollama executable: $($command.Source)"
$version = (& ollama --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -eq 0) { Pass $version } else { Fail $version }

$modelRoot = [Environment]::GetEnvironmentVariable("OLLAMA_MODELS", "User")
if (-not $modelRoot) { $modelRoot = $env:OLLAMA_MODELS }
if ($modelRoot -and (Test-Path -LiteralPath (Join-Path $modelRoot "blobs")) -and (Test-Path -LiteralPath (Join-Path $modelRoot "manifests"))) {
    $blobFiles = @(Get-ChildItem -LiteralPath (Join-Path $modelRoot "blobs") -File -ErrorAction SilentlyContinue)
    $size = [math]::Round((($blobFiles | Measure-Object Length -Sum).Sum / 1GB), 2)
    Pass "Model storage: $modelRoot ($($blobFiles.Count) blobs, $size GB)"
} else {
    Fail "OLLAMA_MODELS is missing or incomplete: $modelRoot"
}

try {
    $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 10
    $names = @($tags.models | ForEach-Object { $_.name })
    if ($names.Count) { Pass "API model list: $($names -join ', ')" } else { Fail "Ollama API returned an empty model list." }
} catch {
    Fail "Ollama API is unavailable at 127.0.0.1:11434."
}

$running = (& ollama ps 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -eq 0) { Pass "ollama ps completed.`n$running" } else { Warn $running }

$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
    $gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -eq 0) { Pass "GPU: $gpu" } else { Warn $gpu }
} else {
    Warn "nvidia-smi is not available."
}

exit $failures
