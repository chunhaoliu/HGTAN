"""Compile compact manuscript artifacts after raw assessment experiments finish."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_assets import (
    DEFAULT_PAPER_TAG,
    TABLE_KEY_TO_LABEL,
    build_paper_manifest,
    selected_figure_stems,
    selected_table_keys,
)
from utils.project_paths import COMPILED_ROOT, EXPERIMENT_ROOT, LEGACY_BENCHMARK_ROOT, as_str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile compact manuscript assets from completed assessment runs.")
    parser.add_argument("--tag", default=DEFAULT_PAPER_TAG, help="Experiment tag prefix, e.g. taes_main.")
    parser.add_argument("--suite-prefix", default=None, help="Suite prefix filter; defaults to --tag.")
    parser.add_argument("--experiment-root", default=as_str(EXPERIMENT_ROOT), dest="experiment_root")
    parser.add_argument("--benchmark-root", dest="experiment_root", help=argparse.SUPPRESS)
    parser.add_argument("--compiled", default=as_str(COMPILED_ROOT))
    parser.add_argument("--paper-out-dir", default=None, help="Optional curated snippet directory for internal table/figure exports.")
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated compile stages to skip: collect,tables,figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = ROOT
    suite_prefix = args.suite_prefix or args.tag
    skipped = {item.strip().lower() for item in args.skip.split(",") if item.strip()}
    experiment_root = Path(args.experiment_root).resolve()
    if not experiment_root.exists() and LEGACY_BENCHMARK_ROOT.exists():
        print(f"Using legacy raw-result root: {LEGACY_BENCHMARK_ROOT}")
        experiment_root = LEGACY_BENCHMARK_ROOT.resolve()
    compiled_root = Path(args.compiled).resolve()
    paper_out_dir = Path(args.paper_out_dir).resolve() if args.paper_out_dir else None
    if paper_out_dir is not None:
        paper_out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = compiled_root / f"{args.tag}_figures"

    steps = []
    if "collect" not in skipped:
        steps.append(
            [
                "scripts/collect_results.py",
                "--root",
                as_str(experiment_root),
                "--out",
                as_str(compiled_root),
                "--tag",
                args.tag,
                "--suite-prefix",
                suite_prefix,
            ]
        )
    if "tables" not in skipped and paper_out_dir is not None:
        steps.append(
            [
                "scripts/make_taes_tables.py",
                "--compiled",
                as_str(compiled_root),
                "--tag",
                args.tag,
                "--suite-prefix",
                suite_prefix,
                "--paper-out-dir",
                as_str(paper_out_dir),
            ]
        )
    if "figures" not in skipped:
        figure_args = [
            "scripts/make_taes_figures.py",
            "--compiled",
            as_str(compiled_root),
            "--experiment-root",
            as_str(experiment_root),
            "--tag",
            args.tag,
            "--suite-prefix",
            suite_prefix,
        ]
        if paper_out_dir is not None:
            figure_args.extend(["--paper-out-dir", as_str(paper_out_dir)])
        steps.append(figure_args)

    for script_args in steps:
        command = [sys.executable, *script_args]
        print("Running:", " ".join(command))
        subprocess.run(command, check=True, cwd=repo_root)

    if paper_out_dir is not None:
        coverage = summarize_paper_bundle(paper_out_dir, figure_dir)
        manifest = build_paper_manifest(tag=args.tag, paper_dir=paper_out_dir, figure_dir=figure_dir, coverage=coverage)
        (paper_out_dir / "paper_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote {paper_out_dir / 'paper_manifest.json'}")


def summarize_paper_bundle(paper_out_dir: Path, figure_dir: Path) -> dict:
    coverage: dict[str, dict[str, list[str]]] = {}
    for layer, snippet_name in [("main", "main_tables.tex"), ("appendix", "appendix_tables.tex")]:
        table_text = (paper_out_dir / snippet_name).read_text(encoding="utf-8") if (paper_out_dir / snippet_name).exists() else ""
        figure_expected = selected_figure_stems(layer)
        table_expected = selected_table_keys(layer)
        coverage[layer] = {
            "available_figures": [stem for stem in figure_expected if (figure_dir / f"{stem}.png").exists()],
            "missing_figures": [stem for stem in figure_expected if not (figure_dir / f"{stem}.png").exists()],
            "available_tables": [
                key for key in table_expected if rf"\label{{{TABLE_KEY_TO_LABEL[key]}}}" in table_text
            ],
            "missing_tables": [
                key for key in table_expected if rf"\label{{{TABLE_KEY_TO_LABEL[key]}}}" not in table_text
            ],
        }
    return coverage


if __name__ == "__main__":
    main()
