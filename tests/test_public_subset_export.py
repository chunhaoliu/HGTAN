import csv
import hashlib
import json

from scripts.export_public_subset import build_bundle, write_public_subset


def test_public_subset_uses_masked_normalized_protocol(tmp_path):
    bundle = build_bundle(seed=2026, n_tracks=120)
    csv_path, metadata_path = write_public_subset(
        bundle,
        tmp_path,
        seed=2026,
        n_tracks=120,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["protocol"] == "latent_state_masked"
    assert metadata["split_sizes"] == {"train": 84, "val": 18, "test": 18}
    assert metadata["n_rows"] == 120 * 64
    assert metadata["csv_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 120 * 64
    assert {row["target_type"] for row in rows} == {"0.00000000"}
    assert {row["mission_type"] for row in rows} == {"0.00000000"}
    assert {int(row["threat_label"]) for row in rows} <= set(range(1, 6))
    assert {int(row["urgency_label"]) for row in rows} <= set(range(1, 4))
