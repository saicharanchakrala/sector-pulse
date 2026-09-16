# Build the feed image and push it to ECR.
#
# Docker Desktop must be running. Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File deploy\push_image.ps1
#
# The build context is source only - see .dockerignore, which keeps the
# 754 MB bar_cache, the 559 MB forecast_cache and every personal file out
# of both the context and the image.

$ErrorActionPreference = "Stop"

$Profile_ = "innomesh-dev"
$Account  = "322535271012"
$Region   = "ap-southeast-2"
$Name     = "zone-pulse"
$Registry = "$Account.dkr.ecr.$Region.amazonaws.com"
$Image    = "$Registry/$Name"

# A tag that says what was built and when. :latest moves too, because the
# task definition pins to :latest and a dated tag alone would never deploy.
$Tag = Get-Date -Format "yyyyMMdd-HHmm"

Write-Host "[1] Checking Docker" -ForegroundColor Cyan
docker version --format "{{.Server.Version}}"
if (-not $?) { throw "Docker is not running. Start Docker Desktop and retry." }

Write-Host "[2] Building $Name`:$Tag" -ForegroundColor Cyan
docker build -t "${Name}:$Tag" .
if (-not $?) { throw "Build failed" }

Write-Host "[3] Signing in to ECR" -ForegroundColor Cyan
# THROUGH CMD, NOT A POWERSHELL PIPE. PowerShell's pipeline hands native
# programs decoded text with its own line endings, and docker login reads
# the result as a malformed token - it fails with a bare
# "400 Bad Request" that says nothing about encoding. The token itself is
# fine either way (2,128 characters, retrieved without error); only the
# pipe is at fault. cmd's pipe is byte-clean, so this works.
cmd /c "aws ecr get-login-password --region $Region --profile $Profile_ | docker login --username AWS --password-stdin $Registry"
if ($LASTEXITCODE -ne 0) { throw "ECR login failed (exit $LASTEXITCODE)" }

Write-Host "[4] Pushing" -ForegroundColor Cyan
docker tag "${Name}:$Tag" "${Image}:$Tag"
docker tag "${Name}:$Tag" "${Image}:latest"
docker push "${Image}:$Tag"
docker push "${Image}:latest"

Write-Host ""
Write-Host "Pushed ${Image}:$Tag and :latest" -ForegroundColor Green
Write-Host "The service picks it up on its next start, or force one now with:"
Write-Host "  aws ecs update-service --cluster avsp-cluster --service $Name --force-new-deployment --profile $Profile_"
