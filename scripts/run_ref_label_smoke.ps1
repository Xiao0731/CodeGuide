param(
    [string]$Python = "",
    [string]$Output = "data/sft_train_smoke_ref_label.jsonl",
    [string]$ReferenceCache = "data/cache/taco_reference_verification_train_full.jsonl",
    [string]$LogPath = "data/cache/ref_label_smoke.log",
    [string[]]$Difficulties = @("easy", "medium", "medium_hard", "hard", "very_hard"),
    [int]$PerDifficulty = 1,
    [string]$IoModeFilter = "any",
    [int]$SampleSize = 0,
    [int]$SampleSeed = 42,
    [switch]$NoDifficultyStratification,
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"

function Resolve-Python {
    param([string]$RequestedPython)
    if ($RequestedPython) {
        return $RequestedPython
    }
    $venvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }
    return "python"
}

function Require-Env {
    param([string]$Name)
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "Missing environment variable: $Name"
    }
}

function Quote-CmdArg {
    param([string]$Value)
    if ($null -eq $Value) {
        return '""'
    }
    return '"' + ($Value -replace '"', '\"') + '"'
}

$pythonExe = Resolve-Python $Python

New-Item -ItemType Directory -Force -Path (Split-Path $Output) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null

$skipBuild = $KeepExisting -and (Test-Path $Output)

if (!$skipBuild) {
    Require-Env "DISTILL_API_KEY"
    Require-Env "DISTILL_BASE_URL"
    Require-Env "DISTILL_MODEL"

    if (!(Test-Path $ReferenceCache)) {
        throw "Reference cache not found: $ReferenceCache"
    }
}

if (!$KeepExisting -and (Test-Path $Output)) {
    Remove-Item -LiteralPath $Output -Force
}

Write-Host "Running reference_guided_label smoke..."
Write-Host "Output: $Output"
Write-Host "Reference cache: $ReferenceCache"
Write-Host "Log: $LogPath"

$buildArgs = @(
    "scripts/build_sft_dataset.py",
    "--source", "taco",
    "--max_items", "26000",
    "--distill-mode", "reference_guided_label",
    "--reference-cache", $ReferenceCache,
    "--require-verified-reference",
    "--min-reference-pass-rate", "1.0",
    "--run_code",
    "--verification-timeout", "60",
    "--max-output-tokens", "8192",
    "--thinking-mode", "off",
    "--concurrency", "1",
    "--out", $Output
)

if ($Difficulties -and $Difficulties.Count -gt 0) {
    if ($NoDifficultyStratification) {
        $Difficulties = @()
    }
}

if ($Difficulties -and $Difficulties.Count -gt 0) {
    $buildArgs += @("--stratified-difficulties")
    $buildArgs += $Difficulties
    $buildArgs += @("--per-difficulty", "$PerDifficulty", "--stratified-seed", "42")
}

if ($IoModeFilter -and $IoModeFilter -ne "any") {
    $buildArgs += @("--io-mode-filter", $IoModeFilter)
}

if ($SampleSize -gt 0) {
    $buildArgs += @("--sample-size", "$SampleSize", "--sample-seed", "$SampleSeed")
}

if ($skipBuild) {
    Write-Host "KeepExisting enabled and output exists; skipping build/API call and running report only."
}
else {
    $quoted = @((Quote-CmdArg $pythonExe))
    $quoted += $buildArgs | ForEach-Object { Quote-CmdArg $_ }
    $cmdLine = ($quoted -join " ") + " 2>&1"
    cmd.exe /d /c $cmdLine | Tee-Object -FilePath $LogPath
    $buildExitCode = $LASTEXITCODE
    if ($buildExitCode -ne 0) {
        throw "build_sft_dataset.py failed with exit code $buildExitCode"
    }
}

Write-Host ""
Write-Host "Smoke report"
Write-Host "============"

$reportScript = @'
# -*- coding: utf-8 -*-
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.data.code_validator import extract_code, validate_syntax
from src.reward.execution import verify_code

output = Path(sys.argv[1])
log_path = Path(sys.argv[2])

records = []
if output.exists():
    with output.open("r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
log_truncated = ("max_tokens" in log_text and "length" in log_text) or ("finish_reason == \"length\"" in log_text)

def metadata_for_reward(record):
    meta = record.get("metadata", {})
    return {
        "io_mode": meta.get("io_mode"),
        "fn_name": meta.get("fn_name"),
        "starter_code": meta.get("starter_code"),
        "test_cases": meta.get("test_cases") or [],
        "tags": meta.get("tags") or [],
        "skill_types": meta.get("skill_types") or [],
        "raw_tags": meta.get("raw_tags") or [],
    }

def has_reference_leak(record):
    user = next((m.get("content", "") for m in record.get("messages", []) if m.get("role") == "user"), "")
    leak_markers = [
        "reference_solution",
        "candidate_results",
        "selected_raw_solution_index",
        "attempted_candidates",
    ]
    return any(marker in user for marker in leak_markers)

def looks_interface_mismatch(result):
    text = f"{result.error or ''} {result.first_failure or ''}".lower()
    markers = [
        "no top-level function",
        "solution has no callable method",
        "no callable method",
        "fn_name",
        "callable",
    ]
    return any(marker in text for marker in markers)

def code_exposes_fn_name(code, fn_name):
    if not fn_name or not code:
        return None
    pattern = r"\bdef\s+" + re.escape(fn_name) + r"\s*\("
    return re.search(pattern, code) is not None

def starter_contract_ok(code, starter_code, fn_name):
    if not code:
        return False
    if not starter_code:
        return code_exposes_fn_name(code, fn_name) if fn_name else True
    if "class Solution" in starter_code:
        return ("class Solution" in code) and bool(code_exposes_fn_name(code, fn_name))
    if fn_name:
        return bool(code_exposes_fn_name(code, fn_name))
    return True

print(f"generated_records: {len(records)}")
if not records:
    print("no records generated; inspect the log file.")
    sys.exit(0)

for idx, record in enumerate(records, 1):
    meta = record.get("metadata", {})
    assistant = next((m.get("content", "") for m in record.get("messages", []) if m.get("role") == "assistant"), "")
    code = extract_code(assistant)
    code_block_ok = code is not None
    syntax_ok = False
    syntax_error = None
    pass_rate = meta.get("pass_rate")
    exec_error = None
    first_failure = None
    interface_mismatch = False
    if code_block_ok:
        syntax_ok, syntax_error = validate_syntax(code)
        if syntax_ok:
            result = verify_code(code, metadata_for_reward(record), timeout=60.0)
            pass_rate = result.pass_rate
            exec_error = result.error
            first_failure = result.first_failure
            interface_mismatch = looks_interface_mismatch(result)
    fn_name_declared = code_exposes_fn_name(code or "", meta.get("fn_name"))
    starter_ok = starter_contract_ok(code or "", meta.get("starter_code") or "", meta.get("fn_name"))

    obvious_truncation = log_truncated or not code_block_ok
    if assistant.count("```") % 2 == 1:
        obvious_truncation = True

    print(f"- #{idx} id={record.get('id')}")
    print(f"  difficulty: {meta.get('difficulty')}")
    print(f"  io_mode: {meta.get('io_mode')}")
    print(f"  fn_name: {meta.get('fn_name')}")
    print(f"  verified_reference: {meta.get('reference_verified')} pass_rate={meta.get('reference_pass_rate')} selected={meta.get('selected_reference_index')}")
    print(f"  user_reference_leak: {has_reference_leak(record)}")
    print(f"  code_block: {code_block_ok}")
    print(f"  syntax_ok: {syntax_ok}")
    print(f"  assistant_code_pass_rate: {pass_rate}")
    print(f"  function_name_mismatch: {interface_mismatch}")
    print(f"  fn_name_declared: {fn_name_declared}")
    print(f"  starter_contract_ok: {starter_ok}")
    print(f"  obvious_truncation: {obvious_truncation}")
    print(f"  syntax_error: {syntax_error}")
    print(f"  execution_error: {exec_error}")
    print(f"  first_failure: {first_failure}")
'@

$tmpReport = New-TemporaryFile
[System.IO.File]::WriteAllText(
    $tmpReport,
    $reportScript,
    [System.Text.UTF8Encoding]::new($false)
)
try {
    & $pythonExe $tmpReport $Output $LogPath
    if ($LASTEXITCODE -ne 0) {
        throw "smoke report failed with exit code $LASTEXITCODE"
    }
}
finally {
    Remove-Item -LiteralPath $tmpReport -Force -ErrorAction SilentlyContinue
}
