param(
    [string]$Python = "",
    [string]$Output = "data/sft_train_smoke_ref_label_call_based.jsonl",
    [string]$ReferenceCache = "data/cache/taco_reference_verification_train_full.jsonl",
    [string]$LogPath = "data/cache/ref_label_call_smoke.log",
    [int]$SampleSize = 5,
    [int]$SampleSeed = 42,
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $PSScriptRoot "run_ref_label_smoke.ps1"

$argsList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $scriptPath,
    "-Output", $Output,
    "-ReferenceCache", $ReferenceCache,
    "-LogPath", $LogPath,
    "-NoDifficultyStratification",
    "-IoModeFilter", "call_based",
    "-SampleSize", "$SampleSize",
    "-SampleSeed", "$SampleSeed"
)

if ($Python) {
    $argsList += @("-Python", $Python)
}

if ($KeepExisting) {
    $argsList += "-KeepExisting"
}

& powershell @argsList

if ($LASTEXITCODE -ne 0) {
    throw "run_ref_label_smoke.ps1 failed with exit code $LASTEXITCODE"
}
