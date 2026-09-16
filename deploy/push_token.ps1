# Push this morning's Kite access token to SSM, then start the feed.
#
# WHY THIS EXISTS. A Kite access token dies around 06:00 IST every day and
# a new one needs an interactive login with your password and 2FA. There is
# no browser on Fargate, so the login happens where it always has - on this
# machine, in the app - and only the resulting token travels.
#
# Run it after you have logged in through the app (which writes
# .kite_session.json), any time before the session you care about:
#   powershell -ExecutionPolicy Bypass -File deploy\push_token.ps1
#
# The token is read from the session file and handed to SSM through a
# temporary JSON file, not through --value. That is not fussiness: an
# argument is visible in the process list for the duration of the call,
# to every other user on the machine. PowerShell history would not have
# recorded it either way, so the earlier comment here claiming history as
# the reason was describing the wrong mechanism.

$ErrorActionPreference = "Stop"

$Profile_ = "innomesh-dev"
$Cluster  = "avsp-cluster"
$Name     = "zone-pulse"
$Session  = Join-Path (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)) ".kite_session.json"

if (-not (Test-Path $Session)) {
    throw "No .kite_session.json. Log in through the app first, then re-run this."
}

$payload = Get-Content $Session -Raw | ConvertFrom-Json
$token   = $payload.access_token
$key     = $payload.api_key
if ([string]::IsNullOrWhiteSpace($token)) { throw "The session file holds no access_token." }

# Kite stamps the session when it was created. A token from yesterday will
# be refused by every call the feed makes, and the failure would look like
# a network problem rather than an expired login, so it is checked here.
if ($payload.created_at) {
    $age = (Get-Date) - [datetime]::Parse($payload.created_at)
    if ($age.TotalHours -gt 12) {
        Write-Host "WARNING: this token was created $([int]$age.TotalHours) hours ago and has probably expired." -ForegroundColor Yellow
        Write-Host "Log in through the app again before relying on it." -ForegroundColor Yellow
    }
}

Write-Host "[1] Storing the access token" -ForegroundColor Cyan

# Through a temp file so the token never appears in this process's argv.
# Deleted in a finally, so an exception mid-call does not leave a live
# broking token sitting in the temp directory.
#
# No KeyId: it encrypts under the account's default aws/ssm key. That is
# a deliberate choice and it has a consequence worth knowing - the shared
# Innomesh-ecs-task-execution-role carries AmazonSSMReadOnlyAccess on "*",
# so anything in this account running under that role can read this
# parameter back. The API key alone cannot trade; this token can.
$tmp = [System.IO.Path]::GetTempFileName()
try {
    $body = @{
        Name      = "/$Name/kite-access-token"
        Type      = "SecureString"
        Value     = $token
        Overwrite = $true
    } | ConvertTo-Json
    # NO BYTE ORDER MARK. Set-Content -Encoding utf8 on Windows PowerShell
    # 5.1 writes UTF-8 WITH a BOM, and the AWS CLI rejects that with
    # "Error parsing parameter 'cli-input-json': Invalid JSON received",
    # which says nothing about the cause.
    [System.IO.File]::WriteAllText($tmp, $body, (New-Object System.Text.UTF8Encoding($false)))
    aws ssm put-parameter --cli-input-json "file://$tmp" --profile $Profile_ | Out-Null
} finally {
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
}
# A failed native command does not throw, so an unchecked exit code would
# leave the feed starting with yesterday's token and no warning.
if ($LASTEXITCODE -ne 0) { throw "put-parameter failed (exit $LASTEXITCODE)." }
Write-Host "    stored /$Name/kite-access-token" -ForegroundColor Green

# The API key rides in the task definition as a plain environment variable
# rather than here: it travels in Kite's own login URL and can do nothing
# without this token. setup.ps1 inlines it from .kite_session.json, so if
# you ever change API keys, re-run setup.ps1 rather than this script.
if (-not [string]::IsNullOrWhiteSpace($key)) {
    Write-Host "    (API key is in the task definition, not SSM)" -ForegroundColor DarkGray
}

Write-Host "[2] Starting the feed" -ForegroundColor Cyan
# Secrets are read once, when the container starts, so an already-running
# task would go on using yesterday's token. Scaling to 1 from 0 is what
# makes it pick this one up; if it is already running, force a new
# deployment instead.
$running = aws ecs describe-services --cluster $Cluster --services $Name `
    --profile $Profile_ --query "services[0].desiredCount" --output text
if ($running -eq "0") {
    aws ecs update-service --cluster $Cluster --service $Name --desired-count 1 --profile $Profile_ | Out-Null
    Write-Host "    scaled to 1" -ForegroundColor Green
} else {
    aws ecs update-service --cluster $Cluster --service $Name --force-new-deployment --profile $Profile_ | Out-Null
    Write-Host "    already running, forced a restart so it reads the new token" -ForegroundColor Green
}

Write-Host ""
Write-Host "Give it 15-20 minutes before the open: it prewarms prior sessions" -ForegroundColor Yellow
Write-Host "from Kite at 3 requests a second, and a container starts with none." -ForegroundColor Yellow
Write-Host ""
Write-Host "Watch it with:"
Write-Host "  aws logs tail /ecs/$Name --follow --profile $Profile_"
