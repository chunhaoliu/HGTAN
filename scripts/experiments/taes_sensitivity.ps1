param(
    [string]$Mode = "gpu",
    [int]$Samples = 4000,
    [int]$Runs = 3,
    [int]$Epochs = 120,
    [int]$BatchSize = 256,
    [int]$NumWorkers = 0,
    [string]$Tag = "taes_sensitivity",
    [switch]$SkipExisting,
    [switch]$SkipCompile,
    [switch]$CompileOnly
)

. (Join-Path $PSScriptRoot "..\benchmarks\_python.ps1")
$PythonExe = Resolve-BenchmarkPython

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Push-Location $RepoRoot
try {
    $skipArg = @()
    if ($SkipExisting) { $skipArg += "--skip-existing" }
    $runtimeArg = @("--num_workers", $NumWorkers, "--batch_size", $BatchSize, "--no_amp")

    if (-not $CompileOnly) {
        & $PythonExe -u run.py --suite observed_time --mode $Mode --models seq_curve --n-samples $Samples --itr $Runs --train_epochs $Epochs --out-subdir "${Tag}_observed_time" @runtimeArg @skipArg
        & $PythonExe -u run.py --suite distance_degradation --mode $Mode --models seq_curve --n-samples $Samples --itr $Runs --train_epochs $Epochs --out-subdir "${Tag}_distance_degradation" @runtimeArg @skipArg
    }

    if (-not $SkipCompile) {
        & $PythonExe -u scripts/compile_taes_bundle.py --tag $Tag --suite-prefix $Tag
    }
}
finally {
    Pop-Location
}
