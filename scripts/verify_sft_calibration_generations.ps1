param(
    [string]$OutputDir,
    [string]$ContainerImage = "python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $outputDirWasExplicit = -not [string]::IsNullOrWhiteSpace($OutputDir)
    if (-not $outputDirWasExplicit) {
        $OutputDir = "outputs/sft/calibration_eval"
    }
    $selectionPath = Join-Path $OutputDir "selection.json"
    if (-not (Test-Path -LiteralPath $selectionPath)) {
        if ($outputDirWasExplicit) {
            throw "selection.json not found under explicitly requested OutputDir: $OutputDir"
        }
        $candidates = @(Get-ChildItem -Path $repoRoot -Recurse -Filter "selection.json" -File |
            Where-Object { $_.Directory.Name -eq "calibration_eval" })
        if ($candidates.Count -eq 1) {
            $OutputDir = Resolve-Path -LiteralPath $candidates[0].Directory.FullName -Relative
            Write-Host "Auto-detected calibration generation directory: $OutputDir"
        }
        elseif ($candidates.Count -eq 0) {
            throw "selection.json not found under $OutputDir or elsewhere in the repository"
        }
        else {
            $paths = ($candidates | ForEach-Object { $_.FullName }) -join ", "
            throw "multiple calibration generation directories found; pass -OutputDir explicitly: $paths"
        }
    }
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
