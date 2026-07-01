"""General runtime and CLI utilities used by ATUAV."""

from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from utils.config import device


CLI_OVERRIDE_MAP = {
    "batch_size": ("train", "batch_size"),
    "learning_rate": ("train", "learning_rate"),
    "weight_decay": ("train", "weight_decay"),
    "train_epochs": ("train", "num_epochs"),
    "patience": ("train", "patience"),
    "dropout": ("model", "dropout"),
    "d_model": ("model", "embed_dim"),
    "n_heads": ("model", "num_heads"),
    "e_layers": ("model", "num_layers"),
    "d_ff": ("model", "hidden_dim"),
    "num_workers": ("train", "num_workers"),
    "prefetch_factor": ("train", "prefetch_factor"),
    "pin_memory": ("train", "pin_memory"),
    "persistent_workers": ("train", "persistent_workers"),
    "use_amp": ("train", "use_amp"),
    "compile_model": ("train", "compile_model"),
    "allow_tf32": ("reproducibility", "allow_tf32"),
    "matmul_precision": ("reproducibility", "matmul_precision"),
    "seq_len": ("sequence", "seq_len"),
    "observed_len": ("sequence", "observed_len"),
    "frame_interval": ("sequence", "frame_interval"),
    "range_m": ("sequence", "range_m"),
    "track_noise_std": ("sequence", "track_noise_std"),
    "type_as_input": ("sequence", "type_as_input"),
}


def apply_cli_overrides(config: dict[str, Any], args: Namespace) -> dict[str, Any]:
    """Apply run.py overrides to the nested experiment config."""
    for arg_name, (section, key) in CLI_OVERRIDE_MAP.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            config[section][key] = value

    if getattr(args, "itr", None) is not None:
        config["run"]["num_runs"] = args.itr

    if getattr(args, "seed", None) is not None:
        config["run"]["seed"] = args.seed
        config["run"]["seeds"] = [args.seed + idx * 13 for idx in range(config["run"].get("num_runs", 1))]

    return config
def format_setting(setting: dict[str, Any]) -> str:
    """Return a compact setting string for logs."""
    return f"{setting['dataset']}::{setting['protocol']}"


def cuda_amp_enabled(use_amp: bool) -> bool:
    """Return True when CUDA AMP should be enabled for the current runtime."""
    return bool(use_amp and device.type == "cuda" and torch.cuda.is_available())


def cuda_autocast(enabled: bool):
    """Return the best available autocast context for CUDA inference/training."""
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda")
    return torch.cuda.amp.autocast(enabled=True)


def move_to_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move one tensor to the configured runtime device."""
    return tensor.to(device, non_blocking=True)
