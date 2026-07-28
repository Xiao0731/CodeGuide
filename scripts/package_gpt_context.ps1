param(
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $projectRoot "artifacts\codeguide_gpt_context_$timestamp.zip"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $projectRoot $OutputPath
}

$outputFullPath = [System.IO.Path]::GetFullPath($OutputPath)
$outputDirectory = Split-Path -Parent $outputFullPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$rootFiles = @(
    ".gitignore",
    "requirements.txt",
    "colab_setup.ipynb"
)

$includeDirectories = @(
    "configs",
    "docs",
    "evals",
    "scripts",
    "src",
    "tests",
    "data\seeds"
)

$sampleFiles = @(
    "data\sft_train_smoke4.jsonl",
    "data\sft_train_smoke_ref_label.jsonl",
    "data\sft_train_smoke_ref_label_call_based.jsonl",
    "data\cache\taco_reference_verification_multi_smoke4.jsonl",
    "data\cache\taco_reference_verification_call_smoke.jsonl",
    "data\raw\TACO\README.md",
    "data\raw\TACO\TACO.py"
)

$files = New-Object System.Collections.Generic.List[System.IO.FileInfo]

$rootMarkdownFiles = Get-ChildItem -LiteralPath $projectRoot -File -Filter "*.md"
foreach ($markdownFile in $rootMarkdownFiles) {
    $files.Add($markdownFile)
}

foreach ($relativePath in $rootFiles + $sampleFiles) {
    $fullPath = Join-Path $projectRoot $relativePath
    if (Test-Path -LiteralPath $fullPath -PathType Leaf) {
        $files.Add((Get-Item -LiteralPath $fullPath))
    }
}

foreach ($relativeDirectory in $includeDirectories) {
    $fullDirectory = Join-Path $projectRoot $relativeDirectory
    if (-not (Test-Path -LiteralPath $fullDirectory -PathType Container)) {
        continue
    }

    Get-ChildItem -LiteralPath $fullDirectory -Recurse -File | Where-Object {
        $relative = $_.FullName.Substring($projectRoot.Length).TrimStart("\", "/")
        $segments = $relative -split "[\\/]"

        "__pycache__" -notin $segments -and
        ".pytest_cache" -notin $segments -and
        $_.Extension -notin @(".pyc", ".pyo") -and
        $_.Name -notlike ".env*" -and
        $_.Extension -ne ".log"
    } | ForEach-Object {
        $files.Add($_)
    }
}

$uniqueFiles = $files |
    Sort-Object FullName -Unique |
    Where-Object { $_.FullName -ne $outputFullPath }

if (Test-Path -LiteralPath $outputFullPath) {
    Remove-Item -LiteralPath $outputFullPath -Force
}

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$stream = [System.IO.File]::Open(
    $outputFullPath,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::ReadWrite,
    [System.IO.FileShare]::None
)

try {
    $archive = New-Object System.IO.Compression.ZipArchive(
        $stream,
        [System.IO.Compression.ZipArchiveMode]::Create,
        $false
    )

    try {
        foreach ($file in $uniqueFiles) {
            $entryName = $file.FullName.Substring($projectRoot.Length).TrimStart("\", "/")
            $entryName = $entryName.Replace("\", "/")

            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive,
                $file.FullName,
                $entryName,
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }
    finally {
        $archive.Dispose()
    }
}
finally {
    $stream.Dispose()
}

$archiveInfo = Get-Item -LiteralPath $outputFullPath
$archiveHash = (Get-FileHash -LiteralPath $outputFullPath -Algorithm SHA256).Hash

Write-Host ""
Write-Host "GPT context package created"
Write-Host "  Path:  $($archiveInfo.FullName)"
Write-Host "  Files: $($uniqueFiles.Count)"
Write-Host "  Size:  $([math]::Round($archiveInfo.Length / 1MB, 2)) MB"
Write-Host "  SHA256: $archiveHash"
