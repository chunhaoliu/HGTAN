"""Export a small sanitized subset from the paper's normalized generator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sequence_pipeline import prepare_sequence_data
from exp.registry import apply_assessment_setting, get_suite_settings
from utils.config import ALL_FEATURES, HGTANConfig


DEFAULT_OUTPUT_DIR = ROOT / "data" / "public"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-tracks", type=int, default=240)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_bundle(seed: int, n_tracks: int) -> dict[str, Any]:
    setting = get_suite_settings("comparison")[0]
    setting["n_samples"] = n_tracks
    config = HGTANConfig.get_experiment_config("benchmark", mode="repro")
    config = apply_assessment_setting(config, setting)
    config["train"]["num_workers"] = 0
    config["train"]["persistent_workers"] = False
    return prepare_sequence_data(seed=seed, config=config)


def write_public_subset(
    bundle: dict[str, Any],
    output_dir: Path,
    *,
    seed: int,
    n_tracks: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "atuav_public_subset.csv"
    metadata_path = output_dir / "atuav_public_metadata.json"
    fieldnames = [
        "split",
        "track_id",
        "frame_index",
        *ALL_FEATURES,
        "threat_label",
        "urgency_label",
    ]

    split_sizes: dict[str, int] = {}
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for split in ("train", "val", "test"):
            metadata = bundle[f"metadata_{split}"]
            sequences = np.asarray(metadata["model_input_sequence"], dtype=np.float32)
            threat = np.asarray(bundle[f"threat_seq_{split}"], dtype=np.int64)
            urgency = np.asarray(bundle[f"urgency_seq_{split}"], dtype=np.int64)
            track_ids = np.asarray(metadata["track_id"])
            split_sizes[split] = int(len(sequences))
            for track_index, sequence in enumerate(sequences):
                for frame_index, features in enumerate(sequence):
                    row: dict[str, Any] = {
                        "split": split,
                        "track_id": int(track_ids[track_index]),
                        "frame_index": frame_index,
                        "threat_label": int(threat[track_index, frame_index]),
                        "urgency_label": int(urgency[track_index, frame_index]),
                    }
                    row.update(
                        {
                            feature: f"{float(features[index]):.8f}"
                            for index, feature in enumerate(ALL_FEATURES)
                        }
                    )
                    writer.writerow(row)

    csv_digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "source": "Temporal HGTAN normalized simulation generator",
        "protocol": "latent_state_masked",
        "reference_policy_variant": "balanced",
        "seed": seed,
        "n_tracks": n_tracks,
        "seq_len": int(bundle["seq_len"]),
        "n_rows": int(n_tracks * bundle["seq_len"]),
        "frame_interval_seconds": 0.2,
        "split_sizes": split_sizes,
        "feature_order": list(ALL_FEATURES),
        "masked_input_channels": ["target_type", "mission_type"],
        "label_ranges": {"threat_label": [1, 5], "urgency_label": [1, 3]},
        "csv_file": csv_path.name,
        "csv_sha256": csv_digest,
        "scope": (
            "Sanitized normalized observations for format and generator checks; "
            "not field-recorded UAV tracks or independent operational ground truth."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, metadata_path


def main() -> None:
    args = parse_args()
    bundle = build_bundle(args.seed, args.n_tracks)
    csv_path, metadata_path = write_public_subset(
        bundle,
        args.output_dir,
        seed=args.seed,
        n_tracks=args.n_tracks,
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
