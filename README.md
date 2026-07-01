# Air-Target UAV Sequential Threat Assessment

This project focuses on sequential multi-feature threat assessment for air-target UAVs. The layout stays compact, closer to the clear `data / exp / models / utils / run.py` style used by Autoformer and FEDformer, but the research framing follows a threat-assessment algorithm paper rather than a dataset-standardization paper. The current codebase keeps two official manuscript experiments:

- `comparison`: baseline comparison on the default sequential threat-assessment protocol.
- `ablation`: TemporalHGTAN ablation under one default setting plus two matched stress settings.
- `observed_time` and `distance_degradation`: focused sensitivity axes that support the comparison experiment.

Everything else has been pushed out of the formal path or removed.

## Structure

- `data/`: synthetic air-target generation and sequential data pipeline.
- `exp/`: experiment engine, compact suite registry, and result writer.
- `layers/`: reusable neural blocks.
- `models/`: traditional baselines, sequential baselines, and HGTAN variants.
- `scripts/`: result collection and figure/table helper scripts.
- `utils/`: configuration, training, metrics, and runtime helpers.
- `run.py`: the only main Python entry.

Runtime outputs are written outside this tree to `../Outputs/atuav_assessment/`.

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

Formal manuscript experiments:

```powershell
py run.py --suite comparison --mode gpu --models seq_main --n-samples 4000 --itr 3 --train_epochs 120 --out-subdir taes_main_comparison
py run.py --suite ablation --mode gpu --models seq_ablation --n-samples 4000 --itr 3 --train_epochs 120 --out-subdir taes_main_ablation
```

The `ablation` suite now bundles three settings under one formal experiment:

- default full observation
- short-history stress at `32` frames (`6.4 s`)
- far-range stress at `5000 m`

Focused sensitivity axes:

```powershell
py run.py --suite observed_time --mode gpu --models seq_curve --n-samples 4000 --itr 3 --train_epochs 120 --batch_size 256 --no_amp --out-subdir taes_sensitivity_observed_time
py run.py --suite distance_degradation --mode gpu --models seq_curve --n-samples 4000 --itr 3 --train_epochs 120 --batch_size 256 --no_amp --out-subdir taes_sensitivity_distance_degradation
```

Or use the compact campaign script:

```powershell
.\scripts\experiments\taes_full.ps1 -Samples 4000 -Runs 3 -Epochs 120 -Tag taes_main -NumWorkers 0
.\scripts\experiments\taes_sensitivity.ps1 -Samples 4000 -Runs 3 -Epochs 120 -BatchSize 256 -Tag taes_sensitivity -NumWorkers 0
```

On Windows, `--num_workers 0 --no_persistent_workers` is the stable fallback when DataLoader multiprocessing is noisy.

## Model Groups

- `traditional`: TOPSIS, GRA, Fuzzy, Entropy-TOPSIS, Combined-TOPSIS, TemporalHMM
- `seq_lite`: LastFrameMLP, TemporalHGTAN
- `seq_main`: traditional baselines plus LastFrameMLP, MeanPoolMLP, TemporalGRU, TemporalLSTM, TemporalHGTAN
- `seq_curve`: TOPSIS, TemporalHMM, TemporalLSTM, TemporalHGTAN for dynamic-curve and sensitivity figures
- `seq_ablation`: TemporalHGTAN, TemporalHGTAN_LastFrame, TemporalHGTAN_MeanPool, TemporalHGTAN_NoSynergy, TemporalHGTAN_NoPrior

For the classical dual-task baselines, each MCDM family now uses its own urgency scorer on an urgency-relevant indicator subset instead of sharing one identical urgency rule across every static method.

## Manuscript Assets

The paper-facing manuscript is kept in one file:

- `../IEEE_TAES_Manuscript/IEEEtaes_Manuscript.tex`

After the two official runs, compile the assessment summary and refresh the manuscript figure assets:

```powershell
py scripts/compile_taes_bundle.py --tag taes_main --suite-prefix taes_main
```

This default command regenerates:

- `../Outputs/atuav_assessment/compiled/<tag>_summary.csv`
- `../Outputs/atuav_assessment/compiled/<tag>_figures/`

If you explicitly need internal table/figure snippets for checking, you can still request them:

```powershell
py scripts/compile_taes_bundle.py --tag taes_main --suite-prefix taes_main --paper-out-dir ../Outputs/atuav_assessment/compiled/taes_main_paper
```

## Outputs

Official assessment outputs:

```text
../Outputs/atuav_assessment/experiments/<tag>_comparison/
../Outputs/atuav_assessment/experiments/<tag>_ablation/
../Outputs/atuav_assessment/experiments/<tag>_observed_time/
../Outputs/atuav_assessment/experiments/<tag>_distance_degradation/
```

Compiled manuscript assets:

```text
../Outputs/atuav_assessment/compiled/<tag>_summary.csv
../Outputs/atuav_assessment/compiled/<tag>_figures/
```

Historical result folders under `../Outputs/atuav_bench/` are kept only for backward compatibility with earlier runs. The current project philosophy is simple: one default sequential assessment protocol, two official experiments, two focused sensitivity axes, one core manuscript TEX, and only the strongest evidence kept in the formal paper path.
