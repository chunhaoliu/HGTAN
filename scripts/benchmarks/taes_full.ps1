param(
    [string]$Mode = "gpu",
    [int]$Samples = 4000,
    [int]$Runs = 5,
    [int]$Epochs = 150,
    [int]$NumWorkers = 0,
    [string]$Tag = "taes_main",
    [switch]$SkipExisting,
    [switch]$SkipCompile,
    [switch]$CompileOnly
)

. (Join-Path $PSScriptRoot "_python.ps1")
$PythonExe = Resolve-BenchmarkPython

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Push-Location $RepoRoot
try {
    $skipArg = @()
    if ($SkipExisting) { $skipArg += "--skip-existing" }
    $runtimeArg = @("--num_workers", $NumWorkers)

    if (-not $CompileOnly) {
        & $PythonExe -u run.py --suite comparison --mode $Mode --models seq_main --n-samples $Samples --itr $Runs --train_epochs $Epochs --out-subdir "${Tag}_comparison" @runtimeArg @skipArg
        & $PythonExe -u run.py --suite ablation --mode $Mode --models seq_ablation --n-samples $Samples --itr $Runs --train_epochs $Epochs --out-subdir "${Tag}_ablation" @runtimeArg @skipArg
    }

    if (-not $SkipCompile) {
        & $PythonExe -u scripts/compile_taes_bundle.py --tag $Tag --suite-prefix $Tag
    }
}
finally {
    Pop-Location
}
