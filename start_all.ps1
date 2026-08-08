# Starts the llama.cpp server for the local Qwen model and prints next steps.
# Replaces the old Script.bat (which only activated a conda env and never
# actually started anything).
#
# This does NOT register itself as a Windows Scheduled Task / Startup item —
# run it manually, or register it yourself, e.g.:
#   schtasks /Create /TN "BrainAnalyzerLLM" /TR "powershell.exe -File `"$PSScriptRoot\start_all.ps1`"" /SC ONLOGON

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not installed or not on PATH. Install it from https://docs.astral.sh/uv/"
    exit 1
}

Write-Host "Syncing Python environment with uv..." -ForegroundColor Cyan
uv sync

Write-Host "Starting llama-server (Qwen) in the background..." -ForegroundColor Cyan
$job = Start-Job -ScriptBlock {
    Set-Location $using:PSScriptRoot
    uv run python llm_runner.py
}

$envFile = Join-Path $PSScriptRoot ".env"
$port = "8081"
if (Test-Path $envFile) {
    $match = Select-String -Path $envFile -Pattern "^LLAMA_SERVER_PORT=(.+)$"
    if ($match) { $port = $match.Matches[0].Groups[1].Value.Trim() }
}

Write-Host "Waiting for llama-server to become ready on port $port..." -ForegroundColor Cyan
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
    Start-Sleep -Seconds 2
}

if (-not $ready) {
    Write-Warning "llama-server did not report healthy within the timeout. Check job output with: Receive-Job -Id $($job.Id)"
} else {
    Write-Host "llama-server is up at http://127.0.0.1:$port" -ForegroundColor Green
}

Write-Host ""
Write-Host "llama-server is running as background job #$($job.Id) (Receive-Job / Stop-Job $($job.Id) to manage it)." -ForegroundColor Yellow
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  Scenario 1 (plain LLM chat):        uv run python chat_plain.py"
Write-Host "  Scenario 2 (LLM + ML as MCP tool):  uv run python chat_with_tools.py"
Write-Host "  (MCP server itself is spawned automatically by chat_with_tools.py; no separate step needed.)"
