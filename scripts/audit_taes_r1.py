"""Audit the model identity and formal evidence referenced by the TAES R1 paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model_factory import build_model
from utils.config import HGTANConfig
from utils.metrics import count_parameters


DEFAULT_MANIFEST = ROOT / "configs" / "paper" / "taes_r1_c5.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "status": "pass" if passed else "fail", "detail": detail})


def resolve_repo_path(value: str) -> Path:
    return (ROOT / value).resolve()


def verify_checksum(path: Path) -> bool:
    checksum_path = path.with_suffix(".sha256")
    if not path.exists() or not checksum_path.exists():
        return False
    expected = checksum_path.read_text(encoding="ascii").split()[0]
    return hashlib.sha256(path.read_bytes()).hexdigest() == expected


def audit(manifest: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    model_spec = manifest["paper_model"]
    model_cfg = HGTANConfig.get_model_config("hgtan")
    model_cfg["dropout"] = manifest["training"]["dropout"]
    model_cfg["prior_weight_alpha"] = manifest["training"]["prior_weight_alpha"]
    model = build_model(model_spec["registry_name"], model_cfg)
    actual_params = count_parameters(model)
    add_check(
        checks,
        "paper_model_parameter_count",
        actual_params == model_spec["expected_trainable_parameters"],
        f"expected={model_spec['expected_trainable_parameters']}, actual={actual_params}",
    )

    experiment_root = ROOT.parent / "Outputs" / "atuav_assessment" / "experiments"
    for suite_name, directory_name in manifest["formal_suites"].items():
        suite_dir = experiment_root / directory_name
        run_manifest = suite_dir / "run_manifest.json"
        add_check(checks, f"suite_{suite_name}_manifest", run_manifest.exists(), str(run_manifest))
        if not run_manifest.exists():
            continue
        payload = json.loads(run_manifest.read_text(encoding="utf-8"))
        expected_runs = 10 if suite_name == "stability" else 3
        actual_runs = int(payload["overrides"]["num_runs"])
        add_check(
            checks,
            f"suite_{suite_name}_run_count",
            actual_runs == expected_runs,
            f"expected={expected_runs}, actual={actual_runs}",
        )
        if suite_name == "stability":
            add_check(
                checks,
                "stability_model_set",
                payload["models"] == manifest["stability_models"],
                ",".join(payload["models"]),
            )

    stats_path = resolve_repo_path(manifest["paper_assets"]["dataset_statistics"])
    add_check(checks, "dataset_statistics_checksum", verify_checksum(stats_path), str(stats_path))

    stability_audit = resolve_repo_path(manifest["paper_assets"]["stability_audit"])
    audit_ok = stability_audit.exists()
    if audit_ok:
        table = pd.read_csv(stability_audit)
        required_columns = {
            "primary_model",
            "baseline",
            "n",
            "paired_t_p",
            "wilcoxon_p",
            "paired_t_holm_p",
            "wilcoxon_holm_p",
            "alternative",
            "p_adjustment",
        }
        audit_ok = (
            required_columns.issubset(table.columns)
            and set(table["primary_model"]) == {model_spec["registry_name"]}
            and set(table["n"]) == {10}
            and set(table["alternative"]) == {"two-sided"}
            and set(table["p_adjustment"]) == {"Holm"}
        )
    add_check(checks, "stability_paired_audit", audit_ok, str(stability_audit))

    return {
        "schema_version": 1,
        "paper_tag": manifest["paper_tag"],
        "status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = audit(manifest)
    out_path = args.out or resolve_repo_path(manifest["paper_assets"]["audit_report"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2) + "\n"
    temp_path = out_path.with_name(f".{out_path.name}.tmp")
    try:
        temp_path.write_text(payload, encoding="utf-8")
        try:
            os.replace(temp_path, out_path)
        finally:
            temp_path.unlink(missing_ok=True)
    except PermissionError:
        # Some synchronized Windows folders allow file updates but forbid
        # creating the temporary sibling required for atomic replacement.
        temp_path.unlink(missing_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
