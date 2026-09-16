# The variables that point this machine at the containerised feed.
#
# DOT-SOURCE IT, so they land in your own session rather than in a child
# process that exits immediately:
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
# THE VALUES LIVE IN env.vars, not here. run_nightly.cmd needs the same
# four, and cmd cannot dot-source a PowerShell script - so both read one
# plain KEY=VALUE file instead of each carrying its own copy. Two copies
# is how the nightly job came to run with the bucket unset, silently
# skipping the two steps that publish to it.
#
# Unset them again with deploy\env.ps1 -Off to go back to local disk.

param([switch]$Off)

$varsFile = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "env.vars"
if (-not (Test-Path $varsFile)) {
    Write-Host "Missing $varsFile - cannot configure the object store." -ForegroundColor Red
    return
}

$pairs = @{}
foreach ($line in Get-Content $varsFile) {
    $text = $line.Trim()
    if (-not $text -or $text.StartsWith("#")) { continue }
    $split = $text.IndexOf("=")
    if ($split -lt 1) { continue }
    $pairs[$text.Substring(0, $split).Trim()] = $text.Substring($split + 1).Trim()
}

if ($Off) {
    foreach ($name in $pairs.Keys) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
    Write-Host "Back to local disk." -ForegroundColor Yellow
    return
}

foreach ($name in $pairs.Keys) {
    # AWS_PROFILE is set only if nothing already chose one, so a shell
    # deliberately pointed at another account is not silently overridden.
    if ($name -eq "AWS_PROFILE" -and (Test-Path "Env:AWS_PROFILE")) { continue }
    Set-Item -Path "Env:$name" -Value $pairs[$name]
}

Write-Host ("Pointing at s3://{0}/{1} (profile {2})" -f `
    $env:SECTOR_PULSE_S3_BUCKET, $env:SECTOR_PULSE_S3_PREFIX,
    $env:AWS_PROFILE) -ForegroundColor Green
