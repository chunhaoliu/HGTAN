from pathlib import Path

from run import build_out_root, build_parser


def test_out_root_overrides_default_experiment_root():
    parser = build_parser()
    args = parser.parse_args(["--suite", "seq_smoke", "--out-root", "local_runs", "--out-subdir", "baseline_check"])

    assert build_out_root(args) == Path("local_runs") / "baseline_check"
