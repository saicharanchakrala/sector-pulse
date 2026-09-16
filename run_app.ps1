# Start the app pointed at the containerised feed.
#
#   powershell -ExecutionPolicy Bypass -File run_app.ps1
#
# WHY A SCRIPT AND NOT THREE VARIABLES. Forgetting them does not fail
# loudly. The app falls back to local disk, finds whatever a previous
# local feed run left there, and renders it: on 2026-09-15 that was 5,131
# bars across 2,501 instruments, all of it 9,935 seconds old, under a
# status panel that said the feed was present. The scan itself refuses
# bars that stale and downloads instead, so the numbers stay honest - but
# you would spend the morning wondering why the feed "was not working"
# while it streamed perfectly well into a bucket nothing was reading.
#
# Pass -Local to run against local disk deliberately, which is the right
# thing when the container is stopped and you are working on history.

param([switch]$Local)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

if ($Local) {
    . .\deploy\env.ps1 -Off
} else {
    . .\deploy\env.ps1

    # Say what the feed is actually doing before Streamlit takes over the
    # terminal, so "no live bars" is diagnosed here rather than guessed at
    # from the UI.
    $state = aws ecs describe-services --cluster avsp-cluster --services zone-pulse `
        --profile $env:AWS_PROFILE --query "services[0].[desiredCount,runningCount]" --output text 2>$null
    if ($LASTEXITCODE -eq 0 -and $state) {
        $parts = $state -split "\s+"
        if ($parts[1] -eq "0") {
            Write-Host "WARNING: the zone-pulse task is NOT running (desired $($parts[0]), running $($parts[1]))." -ForegroundColor Yellow
            Write-Host "         Run deploy\push_token.ps1 to start it, or the app will" -ForegroundColor Yellow
            Write-Host "         find no live bars and download instead." -ForegroundColor Yellow
        } else {
            Write-Host "feed task running ($($parts[1])/$($parts[0]))" -ForegroundColor Green
        }
    }
}

.\.venv\Scripts\streamlit.exe run app.py
