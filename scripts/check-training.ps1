[CmdletBinding()]
param(
    [ValidateRange(0, 60)]
    [int]$SampleSeconds = 10
)

$ErrorActionPreference = 'Stop'
$cfExecutable = 'C:\Program Files\CloudFolder\cf.exe'
if (-not (Test-Path -LiteralPath $cfExecutable)) {
    throw "CloudFolder executable not found: $cfExecutable"
}
$repositoryRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repositoryRoot
try {
    & $cfExecutable run -- /home/dell/anaconda3/bin/conda run --no-capture-output -n wty_LTL env PYTHONNOUSERSITE=1 python scripts/check_paper_campaign.py --sample-seconds $SampleSeconds
    $remoteExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $remoteExitCode
