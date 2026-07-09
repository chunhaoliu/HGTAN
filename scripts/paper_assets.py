"""Curated paper-asset selections for the ATUAV manuscript."""

from __future__ import annotations

from pathlib import Path


# Current official paper source for the compact two-experiment stage.
DEFAULT_PAPER_TAG = "taes_main"

PAPER_FIGURE_LAYERS = {
    "main": [
        "fig_assessment_protocol_quantification",
        "fig_assessment_protocol_tracks",
        "fig_assessment_protocol_degradation",
        "fig_observed_time_composite_f1",
        "fig_observed_time_dynamic_accuracy",
        "fig_observed_time_false_alarm",
        "fig_distance_degradation_composite_f1",
        "fig_distance_degradation_dynamic_accuracy",
        "fig_distance_degradation_drop",
        "fig_operational_case_trajectory",
        "fig_operational_case_signals",
        "fig_operational_case_curves",
        "fig_operational_case_alarm_timing",
    ],
    "appendix": [],
}

PAPER_TABLE_LAYERS = {
    "main": [
        "comparison",
        "ablation",
    ],
    "appendix": [],
}

TABLE_KEY_TO_LABEL = {
    "comparison": "tab:comparison_experiment",
    "ablation": "tab:ablation_experiment",
}


def selected_figure_stems(layer: str) -> list[str]:
    return list(PAPER_FIGURE_LAYERS.get(layer, []))


def selected_table_keys(layer: str) -> list[str]:
    return list(PAPER_TABLE_LAYERS.get(layer, []))


def build_paper_manifest(*, tag: str, paper_dir: Path, figure_dir: Path, coverage: dict | None = None) -> dict:
    return {
        "paper_tag": tag,
        "paper_dir": str(paper_dir),
        "figure_dir": str(figure_dir),
        "layers": {
            "main": {
                "figures": selected_figure_stems("main"),
                "tables": selected_table_keys("main"),
                "figure_snippet": str(paper_dir / "main_figures.tex"),
                "table_snippet": str(paper_dir / "main_tables.tex"),
            },
            "appendix": {
                "figures": selected_figure_stems("appendix"),
                "tables": selected_table_keys("appendix"),
                "figure_snippet": str(paper_dir / "appendix_figures.tex"),
                "table_snippet": str(paper_dir / "appendix_tables.tex"),
            },
        },
        "internal": {
            "statistics_csv": "",
        },
        "coverage": coverage or {},
    }
