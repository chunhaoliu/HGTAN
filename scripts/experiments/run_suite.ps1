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

$Script = Join-Path $PSScriptRoot "..\benchmarks\run_suite.ps1"
& $Script -Suite $Suite -Mode $Mode -Models $Models -Samples $Samples -Runs $Runs -Epochs $Epochs -Tag $Tag -NumWorkers $NumWorkers -NoPersistentWorkers:$NoPersistentWorkers -SkipExisting:$SkipExisting
