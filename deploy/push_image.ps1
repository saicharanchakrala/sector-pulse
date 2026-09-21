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
Write-Host ""
# THIS USED TO SAY "the service picks it up on its next start", WHICH IS
# WRONG and cost a morning. ECS resolves :latest to a digest when a
# DEPLOYMENT is created, not when a task starts, so scaling 0 -> 1 runs
# whatever digest that deployment resolved - however many images have been
# pushed since. Observed 2026-09-21: this image was pushed at 07:48 and a
# scale-up at 07:55 started the one from three days earlier, looking
# entirely healthy while missing every feature the push was for.
Write-Host "A SCALE-UP WILL NOT PICK THIS UP." -ForegroundColor Yellow
Write-Host "ECS resolves :latest when a DEPLOYMENT is created, so starting" -ForegroundColor Yellow
Write-Host "the service from zero reuses the digest of the last deployment." -ForegroundColor Yellow
Write-Host "Force one - deploy\push_token.ps1 now does this for you, or:" -ForegroundColor Yellow
Write-Host "  aws ecs update-service --cluster avsp-cluster --service $Name --force-new-deployment --profile $Profile_"
Write-Host ""
Write-Host "Then confirm the running task matches what you just pushed:" -ForegroundColor DarkGray
Write-Host "  aws ecr describe-images --repository-name $Name --region $Region --image-ids imageTag=latest --query imageDetails[0].imageDigest --output text --profile $Profile_"
