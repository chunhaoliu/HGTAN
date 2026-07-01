# AGENTS.md

This file is the authoritative project instruction source for the clean HGTAN
code repository.

## Project Overview

Research project: **Sequential Multi-Feature Threat Assessment for Low-Altitude
UAV Targets with Temporal Heterogeneous Reasoning**.

- Primary entry point: `run.py`
- Primary task: predict `threat_label` (5 ordinal levels) and `urgency_label`
  (3 ordinal levels) from noisy sequential UAV-track observations
- Main model family: `TemporalHGTAN`, a dual-task sequential assessment model

## Core Framing

- There are 16 input features grouped into capability, intent, opportunity, and
  context.
- Target type is masked at decision time under the main sequential protocol.
- Labels come from clean scenario state, while model inputs come from degraded
  observations.

## Formal Experiment Chain

Treat these as the current formal paper-facing experiment chain:

- `comparison`
- `ablation`
- `observed_time`
- `distance_degradation`

Treat these as validation or support utilities rather than main paper evidence:

- `smoke`
- `seq_smoke`
- ad hoc `smoke_*`, `dry_*`, and `*_check` result folders
- `scripts/run_external_ground_validation.py`

## Key Commands

Run experiment commands from this repository root.

```powershell
# List experiment suites
py run.py --list-suites

# Quick validation
py run.py --suite smoke --mode speed --models traditional --n-samples 256 --itr 1
py run.py --suite seq_smoke --mode speed --models seq_lite --n-samples 128 --itr 1 --train_epochs 1

# Formal manuscript experiments
py run.py --suite comparison --mode gpu --models seq_main --n-samples 4000 --itr 3 --train_epochs 120 --out-subdir taes_main_comparison
py run.py --suite ablation --mode gpu --models seq_ablation --n-samples 4000 --itr 3 --train_epochs 120 --out-subdir taes_main_ablation
py run.py --suite observed_time --mode gpu --models seq_curve --n-samples 4000 --itr 3 --train_epochs 120 --batch_size 256 --no_amp --out-subdir taes_sensitivity_observed_time
py run.py --suite distance_degradation --mode gpu --models seq_curve --n-samples 4000 --itr 3 --train_epochs 120 --batch_size 256 --no_amp --out-subdir taes_sensitivity_distance_degradation

# Compile results
py scripts/compile_taes_bundle.py --tag taes_main --suite-prefix taes_main
```

On Windows, use `--num_workers 0 --no_persistent_workers` when DataLoader
multiprocessing is unstable.

## Architecture Snapshot

```text
.
|- run.py
|- data/
|  |- generator.py
|  |- sequence_generator.py
|  |- sequence_pipeline.py
|  |- dataset_protocol.py
|  `- data_loader.py
|- models/
|  |- hgtan.py
|  |- traditional_baselines.py
|  |- temporal_baselines.py
|  |- graph_baselines.py
|  `- model_factory.py
|- layers/
|  `- hgtan_layers.py
|- exp/
|  |- exp_main.py
|  `- registry.py
|- utils/
`- scripts/
```

## Working Rules

- Prefer reading `README.md` when choosing supported experiment flows.
- Prefer `run.py` as the single entry point instead of calling lower-level
  modules directly.
- Do not treat external validation data as a replacement for the main UAV-track
  evaluation.
- Do not overwrite or reinterpret historical result folders without checking
  timestamps and naming first.
- Runtime outputs are intentionally ignored by Git.

## Dependency Notes

- Expected Python stack: `numpy<2.0`, `torch>=2.2,<2.4`,
  `scikit-learn`, `pandas`, `matplotlib`, `seaborn`.
- The public generator creates synthetic data only; no real operational tracks
  are distributed here.
