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

$Script = Join-Path $PSScriptRoot "..\benchmarks\taes_full.ps1"
& $Script -Mode $Mode -Samples $Samples -Runs $Runs -Epochs $Epochs -NumWorkers $NumWorkers -Tag $Tag -SkipExisting:$SkipExisting -SkipCompile:$SkipCompile -CompileOnly:$CompileOnly
