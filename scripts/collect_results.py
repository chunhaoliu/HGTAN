"""Collect ATUAV assessment CSV artifacts into paper-ready aggregate tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.project_paths import BENCHMARK_ROOT, COMPILED_ROOT, as_str


DEFAULT_ARTIFACTS = ["summary.csv", "run_metrics.csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect assessment CSV outputs.")
    parser.add_argument("--root", default=as_str(BENCHMARK_ROOT), help="Assessment output root.")
    parser.add_argument("--out", default=as_str(COMPILED_ROOT), help="Directory for compiled CSV files.")
    parser.add_argument("--tag", default="latest", help="Output filename prefix.")
    parser.add_argument("--suite-prefix", default=None, help="Only collect experiment suites whose directory starts with this prefix.")
    parser.add_argument("--suites", default=None, help="Comma-separated experiment suite directory allowlist.")
    parser.add_argument("--artifacts", default=",".join(DEFAULT_ARTIFACTS), help="Comma-separated artifact allowlist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for artifact in parse_list(args.artifacts):
        rows = collect_artifact(
            root,
            artifact,
            suite_prefix=args.suite_prefix,
            suites=parse_suites(args.suites),
        )
        if not rows:
            continue
        output_path = out_dir / f"{args.tag}_{artifact}"
        pd.concat(rows, ignore_index=True).to_csv(output_path, index=False)
        print(f"Wrote {output_path}")


def collect_artifact(
    root: Path,
    artifact: str,
    *,
    suite_prefix: str | None = None,
    suites: set[str] | None = None,
) -> list[pd.DataFrame]:
    tables = []
    for path in sorted(root.glob(f"*/*/{artifact}")):
        suite = path.parents[1].name
        if suites is not None and suite not in suites:
            continue
        if suite_prefix is not None and not suite.startswith(suite_prefix):
            continue
        try:
            table = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if table.empty:
            continue
        setting_dir = path.parent.name
        table.insert(0, "source_suite", suite)
        table.insert(1, "source_setting_dir", setting_dir)
        table.insert(2, "source_file", str(path))
        tables.append(table)
    return tables


def parse_suites(value: str | None) -> set[str] | None:
    if value is None:
        return None
    suites = {item.strip() for item in value.split(",") if item.strip()}
    return suites or None


def parse_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
