# The three variables that point this machine at the containerised feed.
#
# DOT-SOURCE IT, so the variables land in your own session rather than in
# a child process that exits immediately:
#
#   . deploy\env.ps1
#   .venv\Scripts\python publish_turnover.py
#   .venv\Scripts\streamlit run app.py
#
# Exists because setting three variables inline is the kind of thing that
# works once and then gets mistyped, and because getting it wrong is not
# loud: publish_turnover refuses outright, but the APP just carries on
# reading local disk and shows you a feed that stopped at whatever the
# last local run wrote.
#
# Unset them again with deploy\env.ps1 -Off to go back to local disk.

param([switch]$Off)

if ($Off) {
    Remove-Item Env:SECTOR_PULSE_S3_BUCKET -ErrorAction SilentlyContinue
    Remove-Item Env:SECTOR_PULSE_S3_PREFIX -ErrorAction SilentlyContinue
    Remove-Item Env:SECTOR_PULSE_S3_REGION -ErrorAction SilentlyContinue
    Write-Host "Back to local disk." -ForegroundColor Yellow
    return
}

$env:SECTOR_PULSE_S3_BUCKET = "original-image-test"
$env:SECTOR_PULSE_S3_PREFIX = "zone-pulse"
$env:SECTOR_PULSE_S3_REGION = "ap-southeast-2"

# AWS_PROFILE too, because boto3 on this machine has no default profile and
# the failure - NoCredentialsError - surfaces as a StorageError that reads
# like a broken bucket rather than a missing profile.
if (-not $env:AWS_PROFILE) { $env:AWS_PROFILE = "innomesh-dev" }

Write-Host "Pointing at s3://$env:SECTOR_PULSE_S3_BUCKET/$env:SECTOR_PULSE_S3_PREFIX (profile $env:AWS_PROFILE)" -ForegroundColor Green
