param(
    [ValidateSet("Main", "Stability", "Statistics", "Assets", "Audit", "All")]
    [string]$Stage = "All",
    [string]$Python = "py"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$OutputRoot = (Resolve-Path (Join-Path $Repo "..\Outputs\atuav_assessment")).Path
$Common = @(
    "--mode", "gpu", "--n-samples", "4000", "--itr", "3",
    "--train_epochs", "100", "--patience", "25", "--batch_size", "256",
    "--num_workers", "0", "--no_persistent_workers", "--learning_rate", "0.0003",
    "--weight_decay", "0.0005", "--prior_weight_alpha", "0.1", "--dropout", "0.08",
    "--no_mixup", "--label_smoothing", "0.0", "--skip-existing"
)

function Invoke-PaperPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

Push-Location $Repo
try {
    if ($Stage -in @("Main", "All")) {
        Invoke-PaperPython (@("run.py", "--suite", "comparison", "--models", "seq_main", "--out-subdir", "r3_comparison_formal_c5_s3") + $Common)
        Invoke-PaperPython (@("run.py", "--suite", "policy_robustness", "--models", "FlatSequenceMLP,TemporalGRU,TemporalLSTM,TemporalTransformer,TemporalTCN,TemporalHGTAN", "--out-subdir", "r3_policy_formal_c5_s3") + $Common)
        Invoke-PaperPython (@("run.py", "--suite", "scenario_holdout", "--models", "FlatSequenceMLP,TemporalGRU,TemporalLSTM,TemporalTransformer,TemporalTCN,TemporalHGTAN", "--out-subdir", "r3_holdout_formal_c5_s3") + $Common)
        Invoke-PaperPython (@("run.py", "--suite", "ablation", "--models", "seq_ablation", "--out-subdir", "r3_ablation_formal_c5_s3") + $Common)
        Invoke-PaperPython (@("run.py", "--suite", "fixed_endpoint_observed_time", "--models", "seq_window", "--out-subdir", "fixed_endpoint_window_formal_s3") + $Common)
        Invoke-PaperPython (@("run.py", "--suite", "fixed_endpoint_ablation", "--models", "seq_ablation", "--out-subdir", "fixed_endpoint_ablation_obs32_formal_s3") + $Common)
        Invoke-PaperPython (@("run.py", "--suite", "distance_degradation", "--models", "seq_curve", "--out-subdir", "r3_distance_formal_c5_s3") + $Common)
        Invoke-PaperPython (@("run.py", "--suite", "missing_robustness", "--models", "seq_missing", "--out-subdir", "r3_missing_formal_c5_s3") + $Common)
    }

    if ($Stage -in @("Stability", "All")) {
        $Seeds = @("161803", "141421", "173205", "223607", "244949", "265359", "316227", "331777", "362881", "447213")
        $StabilityArgs = @(
            "run.py", "--suite", "comparison", "--mode", "gpu",
            "--models", "FlatSequenceMLP,TemporalGRU,TemporalLSTM,TemporalTransformer,TemporalTCN,TemporalHGTAN",
            "--n-samples", "4000", "--itr", "10", "--seeds"
        ) + $Seeds + @(
            "--train_epochs", "100", "--patience", "25", "--batch_size", "256",
            "--num_workers", "0", "--no_persistent_workers", "--learning_rate", "0.0003",
            "--weight_decay", "0.0005", "--prior_weight_alpha", "0.1", "--dropout", "0.08",
            "--no_mixup", "--label_smoothing", "0.0", "--skip-existing", "--out-subdir", "r3_stability_formal_c5_s10"
        )
        Invoke-PaperPython $StabilityArgs
        $Metrics = Join-Path $OutputRoot "experiments\r3_stability_formal_c5_s10\ATUAV-Core__latent_state_masked\run_metrics.csv"
        $Audit = Join-Path $OutputRoot "paper\taes_r1_c5\stability_composite_f1_audit.csv"
        Invoke-PaperPython @(
            "scripts/audit_lockbox_comparison.py", "--run-metrics", $Metrics,
            "--primary-model", "TemporalHGTAN", "--baselines", "FlatSequenceMLP", "TemporalGRU",
            "TemporalLSTM", "TemporalTransformer", "TemporalTCN", "--out", $Audit
        )
    }

    if ($Stage -in @("Statistics", "All")) {
        Invoke-PaperPython @("scripts/export_dataset_statistics.py", "--seeds", "42", "123", "456", "--n-samples", "4000")
    }

    if ($Stage -in @("Assets", "All")) {
        Invoke-PaperPython @(
            "scripts/collect_results.py", "--root", "../Outputs/atuav_assessment/experiments",
            "--out", "../Outputs/atuav_assessment/compiled", "--tag", "r3_c5_fixed_endpoint",
            "--suites", "r3_comparison_formal_c5_s3,r3_policy_formal_c5_s3,r3_holdout_formal_c5_s3,r3_ablation_formal_c5_s3,fixed_endpoint_window_formal_s3,fixed_endpoint_ablation_obs32_formal_s3,r3_distance_formal_c5_s3,r3_missing_formal_c5_s3"
        )
        Invoke-PaperPython @(
            "scripts/make_taes_tables.py", "--compiled", "../Outputs/atuav_assessment/compiled",
            "--tag", "r3_c5_fixed_endpoint", "--paper-out-dir", "../Outputs/atuav_assessment/paper/taes_r1_c5/tables"
        )
        Invoke-PaperPython @(
            "scripts/make_taes_figures.py", "--compiled", "../Outputs/atuav_assessment/compiled",
            "--experiment-root", "../Outputs/atuav_assessment/experiments",
            "--tag", "r3_c5_fixed_endpoint", "--paper-out-dir", "../Outputs/atuav_assessment/paper/taes_r1_c5/figures"
        )
    }

    if ($Stage -in @("Audit", "All")) {
        Invoke-PaperPython @("scripts/audit_taes_r1.py")
    }
}
finally {
    Pop-Location
}
