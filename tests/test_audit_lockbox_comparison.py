import numpy as np
import pandas as pd
import pytest

from scripts.audit_lockbox_comparison import build_audit, holm_adjust, paired_audit


def test_paired_audit_reports_wins_and_effect_direction():
    audit = paired_audit(np.array([0.9, 0.8, 0.7]), np.array([0.8, 0.8, 0.6]))
    assert audit["mean_delta"] == pytest.approx(2.0 / 30.0)
    assert audit["wins"] == 2
    assert audit["ties"] == 1
    assert audit["cohen_dz"] > 0


def test_build_audit_pairs_by_seed_not_row_order():
    rows = [
        {"seed": 2, "model": "H", "task": "joint", "metric": "composite_f1", "value": 0.92},
        {"seed": 1, "model": "B", "task": "joint", "metric": "composite_f1", "value": 0.7},
        {"seed": 1, "model": "H", "task": "joint", "metric": "composite_f1", "value": 0.8},
        {"seed": 2, "model": "B", "task": "joint", "metric": "composite_f1", "value": 0.8},
    ]
    audit = build_audit(
        pd.DataFrame(rows),
        primary_model="H",
        baselines=["B"],
        task="joint",
        metric="composite_f1",
    )
    assert audit.loc[0, "mean_delta"] == pytest.approx(0.11)
    assert audit.loc[0, "wins"] == 2
    assert audit.loc[0, "paired_t_holm_p"] == pytest.approx(audit.loc[0, "paired_t_p"])
    assert audit.loc[0, "wilcoxon_holm_p"] == pytest.approx(audit.loc[0, "wilcoxon_p"])
    assert audit.loc[0, "alternative"] == "two-sided"
    assert audit.loc[0, "p_adjustment"] == "Holm"


def test_holm_adjust_preserves_order_and_monotonicity():
    adjusted = holm_adjust([0.04, 0.01, 0.02])
    assert adjusted == pytest.approx([0.04, 0.03, 0.04])
