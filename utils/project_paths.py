"""Central path conventions for the ATUAV threat-assessment workspace."""

from __future__ import annotations

import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
OUTPUTS_ROOT = PROJECT_ROOT / "Outputs"
RESULTS_DIR = OUTPUTS_ROOT / "atuav_assessment"
EXPERIMENT_ROOT = RESULTS_DIR / "experiments"
# Backward-compatible name used by older helper scripts.
BENCHMARK_ROOT = EXPERIMENT_ROOT
LEGACY_RESULTS_DIR = OUTPUTS_ROOT / "atuav_bench"
LEGACY_BENCHMARK_ROOT = LEGACY_RESULTS_DIR / "benchmark"
COMPILED_ROOT = RESULTS_DIR / "compiled"
LOGS_ROOT = RESULTS_DIR / "logs"
FIGURE_DIR = OUTPUTS_ROOT / "figures" / "atuav_assessment"


def as_str(path: Path) -> str:
    """Return a normalized string path for argparse defaults."""

    return os.path.relpath(path, CODE_ROOT)
