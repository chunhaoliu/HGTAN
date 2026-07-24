# Air-Target UAV Sequential Threat Assessment

This project focuses on sequential multi-feature threat assessment for air-target UAVs. The layout stays compact, closer to the clear `data / exp / models / utils / run.py` style used by Autoformer and FEDformer, but the research framing follows a threat-assessment algorithm paper rather than a dataset-standardization paper. The current codebase keeps one paper-facing experiment chain:

- `comparison`: baseline comparison on the default sequential threat-assessment protocol.
- `ablation`: TemporalHGTAN ablation under one default setting plus two matched stress settings.
- `observed_time` and `distance_degradation`: focused sensitivity axes that support the comparison experiment.
- `policy_robustness`: pre-specified reference-policy variants used to test whether model ranking depends on one frozen policy.
- `scenario_holdout`: leave-one-scenario-family-out evaluation across Probe-Surveillance, EW-Contested, Strike-Penetration, and Saturation-Overload.

The ten-seed `r3_stability_formal_c5_s10` run is the paired statistical audit
of the trainable models. Everything else has been pushed out of the formal
path or archived.

## Structure

- `data/`: synthetic air-target generation and sequential data pipeline.
- `configs/paper/`: frozen machine-readable paper model, training, seed, and result identities.
- `exp/`: experiment engine, compact suite registry, and result writer.
- `layers/`: reusable neural blocks.
- `models/`: traditional baselines, sequential baselines, TemporalHGTAN, and its paper ablations.
- `scripts/`: result collection and figure/table helper scripts.
- `scripts/paper/`: one-command PowerShell orchestration for the frozen TAES R1 evidence chain.
- `utils/`: configuration, training, metrics, and runtime helpers.
- `run.py`: the only main Python entry.

Runtime outputs are written outside this tree to `../Outputs/atuav_assessment/`.
If the OneDrive-backed output directory is not writable in a local sandbox, pass
`--out-root <writable-output-root>` and keep the same `--out-subdir` names.

## Sequential Information Boundary

The formal sequential protocol is `latent_state_masked`. Each clean scenario
contains latent mission, platform, formation, defense, environment, and asset
states. The sequence generator uses these states together with clean
engagement geometry to define frozen reference threat and urgency assessments
in `data/reference_policy.py`. The model receives only the sensor-degraded
indicator sequence: target type and the latent mission code are masked by
default. This makes the main task recovery of a fixed clean-reference policy
from incomplete observations, not direct regression of an observed-feature
score.

The sequential generator bypasses the instantaneous generator's
label-conditioned boundary/confusion injections and applies observation noise,
jitter, and missingness only after the clean reference sequence is built.

## Official Experiments

List available suites:

```powershell
py run.py --list-suites
```

Quick validation:

```powershell
py run.py --suite smoke --mode speed --models traditional --n-samples 256 --itr 1
py run.py --suite seq_smoke --mode speed --models seq_lite --n-samples 128 --itr 1 --train_epochs 1
```

Final manuscript recipe:

```powershell
$common = @('--mode','gpu','--n-samples','4000','--itr','3','--train_epochs','100','--patience','25','--batch_size','256','--num_workers','0','--no_persistent_workers','--learning_rate','0.0003','--prior_weight_alpha','0.1','--dropout','0.08','--no_mixup','--label_smoothing','0.0','--skip-existing')
py run.py --suite comparison --models seq_main --out-subdir r3_comparison_formal_c5_s3 @common
py run.py --suite policy_robustness --models FlatSequenceMLP,TemporalGRU,TemporalLSTM,TemporalTransformer,TemporalTCN,TemporalHGTAN --out-subdir r3_policy_formal_c5_s3 @common
py run.py --suite scenario_holdout --models FlatSequenceMLP,TemporalGRU,TemporalLSTM,TemporalTransformer,TemporalTCN,TemporalHGTAN --out-subdir r3_holdout_formal_c5_s3 @common
py run.py --suite ablation --models seq_ablation --out-subdir r3_ablation_formal_c5_s3 @common
py run.py --suite fixed_endpoint_observed_time --models seq_window --out-subdir fixed_endpoint_window_formal_s3 @common
py run.py --suite fixed_endpoint_ablation --models seq_ablation --out-subdir fixed_endpoint_ablation_obs32_formal_s3 @common
py run.py --suite distance_degradation --models seq_curve --out-subdir r3_distance_formal_c5_s3 @common
py run.py --suite missing_robustness --models seq_missing --out-subdir r3_missing_formal_c5_s3 @common
```

The same frozen recipe is available as a staged script:

```powershell
./scripts/paper/run_taes_r1.ps1 -Stage Main
./scripts/paper/run_taes_r1.ps1 -Stage Stability
./scripts/paper/run_taes_r1.ps1 -Stage Statistics
./scripts/paper/run_taes_r1.ps1 -Stage Assets
./scripts/paper/run_taes_r1.ps1 -Stage Audit
```

The authoritative identities and reporting convention are stored in
`configs/paper/taes_r1_c5.json`. Paper tables report the mean plus or minus the
sample standard deviation across the three main seeds. The stability audit
uses ten fixed paired seeds and the same `TemporalHGTAN` registry entry as the
three-seed comparison.

The default `ablation` suite retains the full-window module controls:

- default full observation
- far-range stress at `5000 m`

The paper-facing history sensitivity uses `fixed_endpoint_observed_time`: every
32/64/96/128-frame tail window ends at the same terminal decision point. The
separate `fixed_endpoint_ablation` suite compares adaptive fusion, mean pooling,
and last-frame selection over the final 32 frames. The distance, policy, and
scenario suites retain their previous roles; none is external field validation.

On Windows, `--num_workers 0 --no_persistent_workers` is the stable fallback when DataLoader multiprocessing is noisy.

## Model Groups

- `traditional`: TOPSIS, GRA, Fuzzy, Entropy-TOPSIS, Combined-TOPSIS, TemporalHMM
- `seq_lite`: LastFrameMLP, TemporalHGTAN
- `seq_main`: traditional baselines plus LastFrameMLP, MeanPoolMLP, FlatSequenceMLP, TemporalGRU, TemporalLSTM, TemporalTransformer, TemporalTCN, TemporalHGTAN
- `seq_curve`: TOPSIS, TemporalHMM, TemporalGRU, TemporalLSTM, TemporalHGTAN for dynamic-curve and sensitivity figures
- `seq_ablation`: TemporalHGTAN, TemporalHGTAN_LastFrame, TemporalHGTAN_MeanPool, TemporalHGTAN_NoSynergy, TemporalHGTAN_NoPrior

For the classical dual-task baselines, each MCDM family now uses its own urgency scorer on an urgency-relevant indicator subset instead of sharing one identical urgency rule across every static method.

## Manuscript Assets

The current revision manuscript is kept in one file outside this repository:

- `../R1_Major_Revision/Unmarked_Revised_Manuscript/Unmarked_Revised_Manuscript.tex`

Collect only the final c5 suites before refreshing manuscript figures:

```powershell
py scripts/collect_results.py --root ../Outputs/atuav_assessment/experiments --out ../Outputs/atuav_assessment/compiled --tag r3_c5_fixed_endpoint --suites r3_comparison_formal_c5_s3,r3_policy_formal_c5_s3,r3_holdout_formal_c5_s3,r3_ablation_formal_c5_s3,fixed_endpoint_window_formal_s3,fixed_endpoint_ablation_obs32_formal_s3,r3_distance_formal_c5_s3,r3_missing_formal_c5_s3
py scripts/make_taes_figures.py --compiled ../Outputs/atuav_assessment/compiled --experiment-root ../Outputs/atuav_assessment/experiments --tag r3_c5_fixed_endpoint
```

These commands regenerate:

- `../Outputs/atuav_assessment/compiled/r3_c5_fixed_endpoint_summary.csv`
- `../Outputs/atuav_assessment/compiled/r3_c5_fixed_endpoint_figures/`

## Outputs

Official assessment outputs:

```text
../Outputs/atuav_assessment/experiments/<tag>_comparison/
../Outputs/atuav_assessment/experiments/<tag>_ablation/
../Outputs/atuav_assessment/experiments/<tag>_observed_time/
../Outputs/atuav_assessment/experiments/<tag>_distance_degradation/
../Outputs/atuav_assessment/experiments/<tag>_policy/
../Outputs/atuav_assessment/experiments/<tag>_holdout/
```

Compiled manuscript assets:

```text
../Outputs/atuav_assessment/compiled/<tag>_summary.csv
../Outputs/atuav_assessment/compiled/<tag>_figures/
```

Frozen audit assets:

```text
../Outputs/atuav_assessment/paper/taes_r1_c5/dataset_statistics.json
../Outputs/atuav_assessment/paper/taes_r1_c5/dataset_statistics.sha256
../Outputs/atuav_assessment/paper/taes_r1_c5/stability_composite_f1_audit.csv
../Outputs/atuav_assessment/paper/taes_r1_c5/audit_report.json
```

Historical result folders under `../Outputs/atuav_bench/` are kept only for backward compatibility. Paper claims must use one named experiment chain and must not mix old and final configurations.
Archived optimization outputs under `../Archive/HGTAN_exploration_20260713/`
are provenance records only and are excluded from manuscript tables.
