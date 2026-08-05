param(
    [string]$RunRoot = "outputs/eval/evalplus_code_capability_v1",
    [string]$Image = "ganler/evalplus:v0.3.1",
    [int]$Parallel = 4,
    [switch]$PullImage,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path ".").Path
$RunRootHost = Join-Path $ProjectRoot $RunRoot

if (-not (Test-Path $RunRootHost)) { throw "Run root does not exist: $RunRootHost" }
if ($Parallel -le 0) { throw "Parallel must be positive" }

docker info | Out-Null
if ($PullImage) {
    docker pull $Image
    if ($LASTEXITCODE -ne 0) { throw "Failed to pull $Image" }
}

$variants = @("base", "mixed_lr1e4_step020", "mixed_lr1e4_step200")
$datasets = @("humaneval", "mbpp")
$expected = @{ "humaneval" = 164; "mbpp" = 378 }
$logDir = Join-Path $RunRootHost "evaluation_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

foreach ($dataset in $datasets) {
    foreach ($variant in $variants) {
        $relativeSample = "$RunRoot/samples/$dataset/$variant.jsonl"
        $hostSample = Join-Path $ProjectRoot $relativeSample
        if (-not (Test-Path $hostSample)) { throw "Missing sample file: $hostSample" }
        $actualRows = (Get-Content $hostSample -Encoding UTF8 | Where-Object { $_.Trim().Length -gt 0 }).Count
        if ($actualRows -ne $expected[$dataset]) {
            throw "Sample count mismatch for $dataset/$variant: expected=$($expected[$dataset]) actual=$actualRows"
        }
        if ($Force) {
            Get-ChildItem -Path (Split-Path $hostSample) -Filter "$variant*eval_results*" -ErrorAction SilentlyContinue | Remove-Item -Force
        }
        $containerSample = "/app/" + ($relativeSample -replace "\\", "/")
        $logPath = Join-Path $logDir "${dataset}_${variant}.log"
        Write-Host "=== EvalPlus $dataset / $variant ==="
        $dockerArgs = @("run", "--rm", "-v", "${ProjectRoot}:/app", "-w", "/app", $Image, "evalplus.evaluate", "--dataset", $dataset, "--samples", $containerSample, "--parallel", "$Parallel")
        if ($Force) { $dockerArgs += "--i-just-wanna-run" }
        & docker @dockerArgs 2>&1 | Tee-Object -FilePath $logPath
        if ($LASTEXITCODE -ne 0) { throw "EvalPlus failed for $dataset/$variant" }
    }
}

python scripts/summarize_evalplus_code_capability.py --run-root $RunRoot
if ($LASTEXITCODE -ne 0) { throw "Summary generation failed" }
Write-Host "[success] EvalPlus code-capability evaluation complete"
Write-Host "$RunRoot/reports/evalplus_code_summary_wide.csv"
