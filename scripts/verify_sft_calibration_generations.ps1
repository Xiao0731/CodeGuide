param(
    [string]$OutputDir = "outputs/sft/calibration_eval",
    [string]$ContainerImage = "python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    python scripts/evaluate_sft_adapter.py --stage verify `
        --output-dir $OutputDir `
        --container-image $ContainerImage
    if ($LASTEXITCODE -ne 0) {
        throw "SFT calibration verification failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
