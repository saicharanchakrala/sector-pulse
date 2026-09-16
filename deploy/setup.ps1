# Registers the task definition and creates the ECS service.
#
# Safe to re-run: it re-registers a revision and points the service at it.
#
# NOT HERE ANY MORE, because both are one-time and already done:
#   * the bucket policy granting Innomesh-ecs-task-execution-role
#     s3:GetObject on s3://original-image-test/zone-pulse/* - the one
#     permission that role was missing, granted on the BUCKET so nothing
#     shared had to change. Re-create with:
#       aws s3api put-bucket-policy --bucket original-image-test `
#         --policy file://deploy/bucket-policy.json --profile innomesh-dev
#   * the /ecs/zone-pulse log group, 30-day retention.
#
# ROLES. Both the task role and the execution role are the account's
# existing Innomesh-ecs-task-execution-role, which already carries ECR
# pull, CloudWatch logs, AmazonSSMReadOnlyAccess and s3:PutObject on "*".
#
# Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File deploy\setup.ps1

$ErrorActionPreference = "Stop"

$Profile_   = "innomesh-dev"
$Cluster    = "avsp-cluster"
$Name       = "zone-pulse"
$Here       = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root       = Split-Path -Parent $Here

# Taken from the cluster's existing service so the feed lands in the same
# subnets and security group. Outbound only: no load balancer, no inbound.
$Subnets    = "subnet-0b047ea0ce0a2a359,subnet-0be642dbf321f8d1a,subnet-024a818f420dd330d"
$SecGroup   = "sg-0cb4c355ce5f895f3"

function Step($n, $text) { Write-Host ""; Write-Host "[$n] $text" -ForegroundColor Cyan }
function Ok($text)       { Write-Host "    $text" -ForegroundColor Green }
function Skip($text)     { Write-Host "    $text" -ForegroundColor DarkGray }

# --- 1. Task definition --------------------------------------------------

Step 1 "Task definition"

# THE API KEY IS AN IDENTIFIER, NOT A CREDENTIAL. It travels in Kite's own
# login URL and in the websocket URL, and on its own it can do nothing -
# every call needs the access token too. So it rides as a plain
# environment variable. The ACCESS TOKEN does not: that one places orders,
# and stays a SecureString that push_token.ps1 writes.
#
# It is substituted here from .kite_session.json rather than committed, so
# the repository never carries it and there is no manual step to forget.
$session = Join-Path $Root ".kite_session.json"
if (-not (Test-Path $session)) {
    throw "No .kite_session.json - log in through the app once, then re-run."
}
$apiKey = (Get-Content $session -Raw | ConvertFrom-Json).api_key
if ([string]::IsNullOrWhiteSpace($apiKey)) {
    throw "The session file holds no api_key."
}

$tmp = [System.IO.Path]::GetTempFileName()
try {
    $json = (Get-Content "$Here/task-definition.json" -Raw).Replace("__KITE_API_KEY__", $apiKey)

    # NO BYTE ORDER MARK. Set-Content -Encoding utf8 on Windows PowerShell
    # 5.1 writes UTF-8 WITH a BOM, and the AWS CLI rejects that with
    # "Error parsing parameter 'cli-input-json': Invalid JSON received",
    # which says nothing about the cause. WriteAllText with an explicit
    # UTF8Encoding($false) is the only reliable way to get a bare file.
    [System.IO.File]::WriteAllText($tmp, $json, (New-Object System.Text.UTF8Encoding($false)))

    $revision = aws ecs register-task-definition --cli-input-json "file://$tmp" `
        --profile $Profile_ --query "taskDefinition.revision" --output text
} finally {
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
}

# A FAILED NATIVE COMMAND DOES NOT THROW, even under ErrorActionPreference
# Stop. Without this the script printed "registered zone-pulse: with the
# API key inlined" over an empty revision and went on to create a service
# pointing at a task definition that did not exist.
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($revision)) {
    throw "register-task-definition failed (exit $LASTEXITCODE). Nothing was created."
}
Ok "registered ${Name}:$revision with the API key inlined"

# --- 2. Service ----------------------------------------------------------

Step 2 "ECS service"

# STOP BEFORE STARTING. The default is maximumPercent 200 and
# minimumHealthyPercent 100, which brings the replacement task up BEFORE
# taking the old one down. For a web service that is the point; here it
# would briefly run two feeds, holding two Kite sockets, building two
# different bar sets and racing each other on the same S3 key.
#
# That matters more than it looks. live_feed skips its file lock on
# Fargate on the grounds that desiredCount 1 is a stronger singleton
# guarantee - and with the default deployment configuration, it is not one
# at all. This is what makes that trade honest.
$Deployment = "maximumPercent=100,minimumHealthyPercent=0"

$svc = aws ecs describe-services --cluster $Cluster --services $Name `
    --profile $Profile_ --query "services[?status=='ACTIVE'].serviceName" --output text
if ($svc -eq $Name) {
    Skip "service $Name already exists, updating to the new revision"
    aws ecs update-service --cluster $Cluster --service $Name `
        --task-definition "${Name}:$revision" `
        --deployment-configuration $Deployment `
        --profile $Profile_ | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "update-service failed (exit $LASTEXITCODE)." }
    Ok "service points at ${Name}:$revision, stop-before-start"
} else {
    # desiredCount 0 ON PURPOSE. Nothing should start until the access
    # token exists, and you scale this yourself around the session anyway.
    aws ecs create-service --cluster $Cluster --service-name $Name `
        --task-definition "${Name}:$revision" --desired-count 0 --launch-type FARGATE `
        --deployment-configuration $Deployment `
        --network-configuration "awsvpcConfiguration={subnets=[$Subnets],securityGroups=[$SecGroup],assignPublicIp=ENABLED}" `
        --profile $Profile_ | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "create-service failed (exit $LASTEXITCODE)." }
    Ok "created service $Name at desiredCount 0, stop-before-start"
}

Write-Host ""
Write-Host "Ready. Every trading morning, after logging in through the app:" -ForegroundColor Yellow
Write-Host "  powershell -ExecutionPolicy Bypass -File deploy\push_token.ps1"
Write-Host ""
Write-Host "Nightly (run_nightly.cmd does it): publish_turnover.py, or the" -ForegroundColor Yellow
Write-Host "feed streams ~216 F&O names instead of ~2,500." -ForegroundColor Yellow
Write-Host ""
