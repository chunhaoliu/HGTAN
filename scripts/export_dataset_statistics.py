"""Export a deterministic audit snapshot of the paper dataset protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sequence_pipeline import prepare_sequence_data
from exp.registry import apply_assessment_setting, get_suite_settings
from utils.config import ALL_FEATURES, HGTANConfig
from utils.project_paths import RESULTS_DIR


DEFAULT_OUT = RESULTS_DIR / "paper" / "taes_r1_c5" / "dataset_statistics.json"
STREAM_KEYS = {
    "clean_reference": "clean_sequence",
    "noisy_observation": "noisy_sequence",
    "masked_model_input": "model_input_sequence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--verify-against",
        type=Path,
        help="Fail unless the generated JSON is byte-identical to this frozen snapshot.",
    )
    return parser.parse_args()


def count_values(values: np.ndarray) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in np.asarray(values).reshape(-1)).items()))


def stream_statistics(values: np.ndarray) -> dict[str, dict[str, float]]:
    values = np.asarray(values, dtype=np.float64)
    flattened = values.reshape(-1, values.shape[-1])
    return {
        feature: {
            "mean": float(np.mean(flattened[:, index])),
            "std": float(np.std(flattened[:, index], ddof=0)),
            "min": float(np.min(flattened[:, index])),
            "max": float(np.max(flattened[:, index])),
        }
        for index, feature in enumerate(ALL_FEATURES)
    }


def _concatenate_metadata(bundle: dict[str, Any], key: str) -> np.ndarray:
    return np.concatenate(
        [np.asarray(bundle[f"metadata_{split}"][key]) for split in ("train", "val", "test")],
        axis=0,
    )


def _fingerprint(bundle: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for split in ("train", "val", "test"):
        metadata = bundle[f"metadata_{split}"]
        for key in ("model_input_sequence",):
            digest.update(np.ascontiguousarray(metadata[key], dtype=np.float32).tobytes())
        digest.update(np.ascontiguousarray(bundle[f"t_{split}"], dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(bundle[f"u_{split}"], dtype=np.int64).tobytes())
    return digest.hexdigest()


def summarize_seed(bundle: dict[str, Any], seed: int) -> dict[str, Any]:
    streams = {name: _concatenate_metadata(bundle, key) for name, key in STREAM_KEYS.items()}
    clean = streams["clean_reference"]
    noisy = streams["noisy_observation"]
    model_input = streams["masked_model_input"]

    threat = np.concatenate([np.asarray(bundle[f"t_{split}"]) for split in ("train", "val", "test")])
    urgency = np.concatenate([np.asarray(bundle[f"u_{split}"]) for split in ("train", "val", "test")])
    scenario_family = np.concatenate(
        [np.asarray(bundle[f"metadata_{split}"]["scenario_family"]) for split in ("train", "val", "test")]
    )

    return {
        "seed": seed,
        "fingerprint_sha256": _fingerprint(bundle),
        "split_sizes": {split: int(len(bundle[f"t_{split}"])) for split in ("train", "val", "test")},
        "final_threat_counts": count_values(threat),
        "final_urgency_counts": count_values(urgency),
        "scenario_family_counts": count_values(scenario_family),
        "observation_degradation": {
            "mean_absolute_change": float(np.mean(np.abs(noisy - clean))),
            "root_mean_square_change": float(np.sqrt(np.mean((noisy - clean) ** 2))),
        },
        "masked_channel_checks": {
            "target_type_max_abs": float(np.max(np.abs(model_input[:, :, 0]))),
            "mission_type_max_abs": float(np.max(np.abs(model_input[:, :, 4]))),
        },
        "feature_statistics": {name: stream_statistics(values) for name, values in streams.items()},
    }


def build_snapshot(seeds: list[int], n_samples: int) -> dict[str, Any]:
    setting = get_suite_settings("comparison")[0]
    setting["n_samples"] = n_samples
    config = HGTANConfig.get_experiment_config("benchmark", mode="repro")
    config = apply_assessment_setting(config, setting)
    config["train"]["num_workers"] = 0
    config["train"]["persistent_workers"] = False

    sequence_cfg = config["sequence"]
    return {
        "schema_version": 1,
        "suite": "comparison",
        "setting": setting["setting_name"],
        "n_tracks_per_seed": n_samples,
        "seeds": seeds,
        "split_strategy": config["data"]["split_strategy"],
        "protocol": {
            key: sequence_cfg.get(key)
            for key in (
                "seq_len",
                "observed_len",
                "frame_interval",
                "range_m",
                "track_noise_std",
                "track_missing_ratio",
                "track_jitter_std",
                "type_as_input",
                "mission_as_input",
                "reference_policy_variant",
            )
        },
        "feature_order": list(ALL_FEATURES),
        "per_seed": [
            summarize_seed(prepare_sequence_data(seed=seed, config=config), seed)
            for seed in seeds
        ],
    }


def write_snapshot(snapshot: dict[str, Any], out_path: Path) -> tuple[Path, Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode("utf-8")
    out_path.write_bytes(payload)
    checksum_path = out_path.with_suffix(".sha256")
    checksum_path.write_text(f"{hashlib.sha256(payload).hexdigest()}  {out_path.name}\n", encoding="ascii")
    return out_path, checksum_path


def main() -> None:
    args = parse_args()
    out_path, checksum_path = write_snapshot(build_snapshot(args.seeds, args.n_samples), args.out)
    print(f"Wrote {out_path}")
    print(f"Wrote {checksum_path}")
    if args.verify_against is not None:
        if out_path.read_bytes() != args.verify_against.read_bytes():
            raise SystemExit(
                f"Generated snapshot differs from frozen artifact: {args.verify_against}"
            )
        print(f"Verified byte-identical to {args.verify_against}")


if __name__ == "__main__":
    main()
