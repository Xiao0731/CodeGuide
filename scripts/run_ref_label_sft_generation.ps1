param(
    [string]$Python = "",
    [string]$ReferenceCache = "data/cache/taco_reference_verification_train_full.jsonl",
    [string]$PilotAccepted = "data/sft_train_pilot50_accepted.jsonl",
    [string]$PilotRejected = "data/sft_train_pilot50_rejected.jsonl",
    [string]$PilotLog = "data/cache/ref_label_pilot50.log",
    [string]$FullAccepted = "data/sft_train_ref_label_accepted.jsonl",
    [string]$FullRejected = "data/sft_train_ref_label_rejected.jsonl",
    [string]$FullLog = "data/cache/ref_label_full_generation.log",
    [int]$PilotPerBucket = 5,
    [int]$Concurrency = 3,
    [switch]$PilotOnly,
    [switch]$StartFullOnly
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

function Load-LocalEnv {
    param([string]$EnvPath)
    if (!(Test-Path $EnvPath)) {
        return
    }
    Get-Content -LiteralPath $EnvPath | ForEach-Object {
        $line = $_.Trim()
        if (!$line -or $line.StartsWith("#")) {
            return
        }
        if ($line -match '^\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"(.*)"\s*$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
        }
        elseif ($line -match '^\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*''(.*)''\s*$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
        }
    }
    Write-Host "Loaded environment variables from $EnvPath"
}

function Require-Env {
    param([string]$Name)
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "Missing environment variable: $Name"
    }
    if ($value -match "YOUR_|your_|placeholder|PLACEHOLDER") {
        throw "Environment variable ${Name} still looks like a placeholder; please fill it in .env"
    }
}

function Count-Jsonl {
    param([string]$Path)
    if (!(Test-Path $Path)) {
        return 0
    }
    return (Get-Content -LiteralPath $Path | Where-Object { $_.Trim() }).Count
}

function Quote-PSArg {
    param([string]$Value)
    return "'" + ($Value -replace "'", "''") + "'"
}

function Quote-CmdArg {
    param([string]$Value)
    if ($null -eq $Value) {
        return '""'
    }
    return '"' + ($Value -replace '"', '\"') + '"'
}

function Invoke-LoggedPython {
    param(
        [string]$PythonExe,
        [string[]]$ArgsList,
        [string]$LogPath
    )
    $quoted = @((Quote-CmdArg $PythonExe))
    $quoted += $ArgsList | ForEach-Object { Quote-CmdArg $_ }
    $cmdLine = ($quoted -join " ") + " 2>&1"
    cmd.exe /d /c $cmdLine | Tee-Object -FilePath $LogPath -Append
    $script:LastLoggedPythonExitCode = $LASTEXITCODE
}

function Summarize-Rejected {
    param([string]$Path)
    if (!(Test-Path $Path)) {
        return @{}
    }
    $counts = @{}
    Get-Content -LiteralPath $Path | Where-Object { $_.Trim() } | ForEach-Object {
        try {
            $record = $_ | ConvertFrom-Json
            $key = [string]$record.failure_type
            if (!$key) { $key = "unknown" }
            if (!$counts.ContainsKey($key)) { $counts[$key] = 0 }
            $counts[$key] += 1
        }
        catch {}
    }
    return $counts
}

function Has-Systemic-Failure {
    param([hashtable]$RejectedCounts, [int]$Total)
    if ($Total -le 0) {
        return $true
    }
    foreach ($key in @("llm_failed", "no_code_block", "syntax_error", "docker_unsupported", "unsupported", "interface_mismatch")) {
        $count = 0
        if ($RejectedCounts.ContainsKey($key)) {
            $count = [int]$RejectedCounts[$key]
        }
        if ($count -ge [Math]::Max(5, [Math]::Ceiling($Total * 0.30))) {
            Write-Host "Systemic failure candidate: $key = $count / $Total"
            return $true
        }
    }
    return $false
}

$repoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $repoRoot
Load-LocalEnv (Join-Path $repoRoot ".env")

Require-Env "DISTILL_API_KEY"
Require-Env "DISTILL_BASE_URL"
Require-Env "DISTILL_MODEL"
Require-Env "CODEGUIDE_EXECUTION_IMAGE"

if (!(Test-Path $ReferenceCache)) {
    throw "Reference cache not found: $ReferenceCache"
}

$pythonExe = Resolve-Python $Python
New-Item -ItemType Directory -Force -Path (Split-Path $PilotAccepted) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $PilotRejected) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $PilotLog) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $FullAccepted) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $FullRejected) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $FullLog) | Out-Null

$commonArgs = @(
    "scripts/build_sft_dataset.py",
    "--source", "taco",
    "--max_items", "26000",
    "--distill-mode", "reference_guided_label",
    "--reference-cache", $ReferenceCache,
    "--require-verified-reference",
    "--min-reference-pass-rate", "1.0",
    "--run-code",
    "--execution-backend", "docker",
    "--container-image", ([Environment]::GetEnvironmentVariable("CODEGUIDE_EXECUTION_IMAGE")),
    "--verification-timeout", "60",
    "--max-output-tokens", "8192",
    "--distill-retries", "1",
    "--thinking-mode", "off",
    "--concurrency", "$Concurrency"
)

if (!$StartFullOnly) {
    Write-Host "Starting 50-problem reference_guided_label pilot..."
    $pilotArgs = @(
        "--out", $PilotAccepted,
        "--rejected-out", $PilotRejected,
        "--stratified-io-modes", "call_based", "standard_input",
        "--stratified-difficulties", "easy", "medium", "medium_hard", "hard", "very_hard",
        "--per-io-difficulty", "$PilotPerBucket",
        "--stratified-seed", "20260728"
    )
    Invoke-LoggedPython $pythonExe ($commonArgs + $pilotArgs) $PilotLog
    $pilotExitCode = $script:LastLoggedPythonExitCode
    if ($pilotExitCode -ne 0) {
        throw "pilot generation failed with exit code $pilotExitCode"
    }

    $accepted = Count-Jsonl $PilotAccepted
    $rejected = Count-Jsonl $PilotRejected
    $total = $accepted + $rejected
    $rejectedCounts = Summarize-Rejected $PilotRejected
    Write-Host "Pilot summary: accepted=$accepted rejected=$rejected total=$total"
    Write-Host "Pilot rejected failure types:"
    $rejectedCounts.GetEnumerator() | Sort-Object Name | ForEach-Object {
        Write-Host "  $($_.Name): $($_.Value)"
    }

    if ($PilotOnly) {
        Write-Host "PilotOnly enabled; not starting full generation."
        exit 0
    }

    if (Has-Systemic-Failure $rejectedCounts $total) {
        Write-Host "Pilot found concentrated engineering failures; full generation not started."
        exit 2
    }
}

Write-Host "Starting full reference_guided_label generation in background..."
$fullArgs = @(
    "--out", $FullAccepted,
    "--rejected-out", $FullRejected
)
$stderrLog = "$FullLog.err"
$proc = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList ($commonArgs + $fullArgs) `
    -RedirectStandardOutput $FullLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru
Write-Host "Full generation started. PID=$($proc.Id)"
Write-Host "Accepted: $FullAccepted"
Write-Host "Rejected: $FullRejected"
Write-Host "Log: $FullLog"
Write-Host "Error log: $stderrLog"
