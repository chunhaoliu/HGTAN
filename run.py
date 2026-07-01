"""Main entry for ATUAV sequential threat-assessment experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.exp_main import AssessmentOptions, Exp_Main, validate_models  # noqa: E402
from exp.registry import ASSESSMENT_SUITES, MODEL_GROUPS, get_suite_settings, parse_model_list  # noqa: E402
from utils.config import PROJECT_NAME  # noqa: E402
from utils.project_paths import EXPERIMENT_ROOT  # noqa: E402


TASK_FORMS = ["instantaneous", "sequential"]
RUN_MODES = ["gpu", "speed", "repro"]
SETTING_OVERRIDE_KEYS = (
    "seq_len",
    "observed_len",
    "frame_interval",
    "range_m",
    "track_noise_std",
    "track_missing_ratio",
    "track_jitter_std",
    "type_as_input",
)


def add_toggle_flag(
    parser: argparse._ArgumentGroup | argparse.ArgumentParser,
    *,
    name: str,
    dest: str,
    help_text: str,
) -> None:
    parser.add_argument(f"--{name}", dest=dest, action="store_true", default=None, help=help_text)
    parser.add_argument(f"--no_{name}", dest=dest, action="store_false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run {PROJECT_NAME}.")
    selection = parser.add_argument_group("Selection")
    output = parser.add_argument_group("Output")
    training = parser.add_argument_group("Training")
    runtime = parser.add_argument_group("Runtime")
    sequence = parser.add_argument_group("Sequence Overrides")

    selection.add_argument("--suite", default="smoke", choices=sorted(ASSESSMENT_SUITES))
    selection.add_argument("--task_form", default=None, choices=TASK_FORMS)
    selection.add_argument("--mode", default="speed", choices=RUN_MODES)
    selection.add_argument("--models", default="lite", help="Model group or comma-separated model names.")
    selection.add_argument("--max-settings", type=int, default=None)
    selection.add_argument("--list-suites", action="store_true")

    output.add_argument("--dry-run", action="store_true", help="Print and save the run manifest only.")
    output.add_argument("--skip-existing", action="store_true", help="Skip complete setting outputs and reuse their summaries.")
    output.add_argument("--out-subdir", default=None)

    training.add_argument("--n-samples", type=int, default=None)
    training.add_argument("--itr", type=int, default=None, help="Number of repeated seeded runs.")
    training.add_argument("--num-runs", type=int, default=None, help="Alias for --itr.")
    training.add_argument("--seed", type=int, default=None)
    training.add_argument("--train_epochs", type=int, default=None)
    training.add_argument("--epochs", type=int, default=None, help="Alias for --train_epochs.")
    training.add_argument("--batch_size", type=int, default=None)
    training.add_argument("--learning_rate", type=float, default=None)
    training.add_argument("--weight_decay", type=float, default=None)
    training.add_argument("--patience", type=int, default=None)
    training.add_argument("--d_model", type=int, default=None)
    training.add_argument("--n_heads", type=int, default=None)
    training.add_argument("--e_layers", type=int, default=None)
    training.add_argument("--d_ff", type=int, default=None)
    training.add_argument("--dropout", type=float, default=None)

    runtime.add_argument("--num_workers", type=int, default=None, help="DataLoader workers for GPU feeding.")
    runtime.add_argument("--prefetch_factor", type=int, default=None, help="Batches prefetched per DataLoader worker.")
    add_toggle_flag(runtime, name="pin_memory", dest="pin_memory", help_text="Pin host memory for DataLoader batches.")
    add_toggle_flag(runtime, name="persistent_workers", dest="persistent_workers", help_text="Keep DataLoader workers alive between epochs.")
    add_toggle_flag(runtime, name="amp", dest="use_amp", help_text="Enable CUDA mixed precision.")
    add_toggle_flag(runtime, name="compile", dest="compile_model", help_text="Use torch.compile when available.")
    add_toggle_flag(runtime, name="tf32", dest="allow_tf32", help_text="Allow TF32 matmul on Ampere+ GPUs.")
    runtime.add_argument("--matmul_precision", default=None, choices=["highest", "high", "medium"])

    sequence.add_argument("--seq_len", type=int, default=None)
    sequence.add_argument("--observed_len", type=int, default=None)
    sequence.add_argument("--frame_interval", type=float, default=None)
    sequence.add_argument("--range_m", type=float, default=None)
    sequence.add_argument("--track_noise_std", type=float, default=None)
    sequence.add_argument("--track_missing_ratio", type=float, default=None)
    sequence.add_argument("--track_jitter_std", type=float, default=None)
    add_toggle_flag(sequence, name="type_as_input", dest="type_as_input", help_text="Append target-type hints to sequential model inputs.")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def normalize_cli_aliases(args: argparse.Namespace) -> None:
    if args.num_runs is not None and args.itr is None:
        args.itr = args.num_runs
    if args.epochs is not None and args.train_epochs is None:
        args.train_epochs = args.epochs


def resolve_settings(args: argparse.Namespace) -> list[dict]:
    settings = get_suite_settings(args.suite)
    if args.max_settings is not None:
        settings = settings[: args.max_settings]
    return [apply_setting_overrides(setting, args) for setting in settings]


def resolve_selected_models(args: argparse.Namespace, task_form: str) -> list[str]:
    model_arg = "seq_lite" if task_form == "sequential" and args.models == "lite" else args.models
    return validate_models(parse_model_list(model_arg))


def build_out_root(args: argparse.Namespace) -> Path:
    return EXPERIMENT_ROOT / (args.out_subdir or args.suite)


def build_assessment_options(args: argparse.Namespace, selected_models: list[str], out_root: Path) -> AssessmentOptions:
    return AssessmentOptions(
        suite=args.suite,
        mode=args.mode,
        models=selected_models,
        out_root=out_root,
        num_runs=args.itr,
        epochs=args.train_epochs,
        skip_existing=args.skip_existing,
        cli_args=args,
    )


def main() -> None:
    args = parse_args()
    if args.list_suites:
        print_available_options()
        return

    normalize_cli_aliases(args)
    settings = resolve_settings(args)
    task_form = infer_task_form(settings)
    selected_models = resolve_selected_models(args, task_form)
    out_root = build_out_root(args)
    options = build_assessment_options(args, selected_models, out_root)

    exp = Exp_Main(settings=settings, options=options)
    if args.dry_run:
        exp.dry_run()
    else:
        exp.run()


def apply_setting_overrides(setting: dict, args: argparse.Namespace) -> dict:
    updated = setting.copy()
    if args.n_samples is not None:
        updated["n_samples"] = args.n_samples
    if args.task_form is not None:
        updated["task_form"] = args.task_form
    for key in SETTING_OVERRIDE_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            updated[key] = value
    return updated


def infer_task_form(settings: list[dict]) -> str:
    task_forms = {setting.get("task_form", "instantaneous") for setting in settings}
    if len(task_forms) != 1:
        raise ValueError(f"Mixed task forms are not supported in one run: {sorted(task_forms)}")
    return next(iter(task_forms))


def print_available_options() -> None:
    print("Assessment experiment suites:")
    for name, suite in ASSESSMENT_SUITES.items():
        n_settings = len(suite["settings"]) if "settings" in suite else len(suite["datasets"]) * len(suite["protocols"])
        print(f"  {name:<16} {n_settings:>3} settings | {suite['description']}")

    print("\nModel groups:")
    for name, models in MODEL_GROUPS.items():
        print(f"  {name:<16} {', '.join(models)}")


if __name__ == "__main__":
    main()
