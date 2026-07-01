param(
    [string]$Suite = "smoke",
    [string]$Mode = "speed",
    [string]$Models = "lite",
    [int]$Samples = 256,
    [int]$Runs = 1,
    [int]$Epochs = 50,
    [string]$Tag = "",
    [int]$NumWorkers = 0,
    [switch]$NoPersistentWorkers,
    [switch]$SkipExisting
)

. (Join-Path $PSScriptRoot "_python.ps1")
$PythonExe = Resolve-BenchmarkPython

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Push-Location $RepoRoot
try {
    $args = @(
        "run.py",
        "--suite", $Suite,
        "--mode", $Mode,
        "--models", $Models,
        "--n-samples", $Samples,
        "--itr", $Runs
    )
    if ($Epochs -ge 0) {
        $args += @("--train_epochs", $Epochs)
    }
    if ($Tag) {
        $args += @("--out-subdir", $Tag)
    }
    if ($NumWorkers -ge 0) {
        $args += @("--num_workers", $NumWorkers)
    }
    if ($NoPersistentWorkers) {
        $args += "--no_persistent_workers"
    }
    if ($SkipExisting) {
        $args += "--skip-existing"
    }
    & $PythonExe @args
}
finally {
    Pop-Location
}
