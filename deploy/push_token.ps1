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
# Needed for the ECR lookup at the end. The ecs calls resolve it from the
# profile; describe-images does not, and passing an empty --region fails
# with a message about credentials rather than about the region.
$Region   = "ap-southeast-2"
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
# ALWAYS A FORCED DEPLOYMENT, even from zero, and that is the whole point
# of this block rather than a plain scale-up.
#
# Two things have to happen here: the container must re-read the secret,
# and it must run the CURRENT image. Scaling 0 -> 1 does the first and
# NOT the second. ECS resolves the :latest tag to a digest when a
# DEPLOYMENT is created, and a scale-up starts tasks from the existing
# deployment - so it launches whatever digest was current when that
# deployment was made, however many images have been pushed since.
#
# Observed 2026-09-21: an image pushed at 07:48 was ignored by a scale-up
# at 07:55, which started Friday's code - no store hydrate, no scan
# logging, none of that morning's work - and looked entirely healthy. It
# took comparing the running task's digest against the one on :latest to
# see it at all.
#
# --force-new-deployment re-resolves the tag, so the task that comes up is
# built from the image actually in ECR. From zero it also raises the
# desired count, so this single call covers both cases.
$desired = aws ecs describe-services --cluster $Cluster --services $Name `
    --profile $Profile_ --query "services[0].desiredCount" --output text
if ($LASTEXITCODE -ne 0) { throw "Could not read the service (exit $LASTEXITCODE)." }

if ($desired -eq "0") {
    aws ecs update-service --cluster $Cluster --service $Name `
        --desired-count 1 --force-new-deployment --profile $Profile_ | Out-Null
} else {
    aws ecs update-service --cluster $Cluster --service $Name `
        --force-new-deployment --profile $Profile_ | Out-Null
}
if ($LASTEXITCODE -ne 0) { throw "Could not update the service (exit $LASTEXITCODE)." }
Write-Host "    forced a new deployment, so it reads the new token AND the current image" -ForegroundColor Green

# WHICH IMAGE IT WILL ACTUALLY RUN, printed rather than assumed. The
# failure above was invisible precisely because nothing ever said which
# digest was running, and a wrong one behaves like a right one until you
# look for a feature that is missing.
$latest = aws ecr describe-images --repository-name $Name --region $Region `
    --image-ids imageTag=latest --profile $Profile_ `
    --query "imageDetails[0].imageDigest" --output text 2>$null
if ($LASTEXITCODE -eq 0 -and $latest) {
    Write-Host "    :latest is $latest" -ForegroundColor DarkGray
    Write-Host "    confirm the running task matches it once it is up:" -ForegroundColor DarkGray
    Write-Host "      aws ecs describe-tasks --cluster $Cluster --tasks (aws ecs list-tasks --cluster $Cluster --service-name $Name --query taskArns[0] --output text) --query tasks[0].containers[0].imageDigest --output text --profile $Profile_" -ForegroundColor DarkGray
}

Write-Host ""
# The old wording here said 15-20 minutes, which was true when a cold
# start refetched 17 days for ~2,500 symbols from Kite - measured
# 2026-09-18 at 836 seconds. The nightly job now publishes the folded
# 3-minute store and the feed hydrates it: 93.2 MB in about a second on
# 2026-09-21. What remains is whatever the store could not prove, so the
# time depends on how complete last night's refresh was.
Write-Host "Prewarm hydrates the published 3-minute store, then fetches only" -ForegroundColor Yellow
Write-Host "what it does not cover - minutes, not the old 15-20. Watch for" -ForegroundColor Yellow
Write-Host "'hydrated the 3minute store' followed by 'N/M symbols have prior bars'." -ForegroundColor Yellow
Write-Host "If the hydrate line is missing, last night's job did not publish." -ForegroundColor Yellow
Write-Host ""
Write-Host "Watch it with:"
Write-Host "  aws logs tail /ecs/$Name --follow --profile $Profile_"
