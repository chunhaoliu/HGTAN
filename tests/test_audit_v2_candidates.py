from __future__ import annotations

import json

from scripts.audit_v2_candidates import audit_candidates, holm_adjust


def _record(root, candidate: str, seed: int, final: float, temporal: float) -> None:
    path = root / candidate / f"seed_{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "metrics": {
                    "final_composite_f1": final,
                    "threat_temporal_macro_f1": temporal,
                },
            }
        ),
        encoding="utf-8",
    )


def test_audit_candidates_pairs_by_seed_and_orients_advantage(tmp_path) -> None:
    for seed, reference, comparator in ((3, 0.90, 0.87), (1, 0.88, 0.86), (2, 0.89, 0.88)):
        _record(tmp_path, "reference", seed, reference, reference - 0.02)
        _record(tmp_path, "comparator", seed, comparator, comparator - 0.03)

    rows = audit_candidates(
        tmp_path,
        "reference",
        ["comparator"],
        ["final_composite_f1", "threat_temporal_macro_f1"],
    )

    assert len(rows) == 2
    assert rows[0]["seeds"] == [1, 2, 3]
    assert rows[0]["advantage_mean"] > 0
    assert rows[0]["wins"] == 3
    assert rows[0]["advantage_ci95_low"] > 0


def test_holm_adjust_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.02])
    assert adjusted[0] <= adjusted[2] <= adjusted[1]
    assert all(0.0 <= value <= 1.0 for value in adjusted)
