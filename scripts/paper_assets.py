"""Curated paper-asset selections for the ATUAV manuscript."""

from __future__ import annotations

from pathlib import Path


# Current official paper evidence bundle.
DEFAULT_PAPER_TAG = "r3_c5_fixed_endpoint"

PAPER_FIGURE_LAYERS = {
    "main": [
        "fig_assessment_protocol_quantification",
        "fig_assessment_protocol_tracks",
        "fig_assessment_protocol_degradation",
        "fig_overall_final_dynamic_tradeoff",
        "fig_stability_paired_delta",
        "fig_classwise_threat_f1",
        "fig_classwise_urgency_f1",
        "fig_observed_time_composite_f1",
        "fig_observed_time_dynamic_accuracy",
        "fig_distance_degradation_composite_f1",
        "fig_distance_degradation_dynamic_accuracy",
        "fig_missing_robustness_legend",
        "fig_missing_random_composite_f1",
        "fig_missing_random_temporal_f1",
        "fig_missing_burst_composite_f1",
        "fig_missing_burst_temporal_f1",
        "fig_ablation_default_composite_f1",
        "fig_ablation_fixed_summary_composite_f1",
        "fig_ablation_default_temporal_f1",
        "fig_ablation_fixed_summary_temporal_f1",
        "fig_event_timing_agreement",
        "fig_event_aligned_disagreement",
        "fig_operational_case_timeline",
    ],
    "appendix": [],
}

PAPER_TABLE_LAYERS = {
    "main": [
        "comparison",
    ],
    "appendix": [],
}

TABLE_KEY_TO_LABEL = {
    "comparison": "tab:comparison_experiment",
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
