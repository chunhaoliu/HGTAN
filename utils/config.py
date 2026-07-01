"""Task constants and default runtime configuration for ATUAV.

Autoformer/FEDformer keep most experiment choices in CLI scripts because their
datasets already define the task. This project also needs a compact place for
threat-assessment constants: indicators, label spaces, prior weights, and
default training modes. Runtime overrides still belong to run.py.
"""

from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.project_paths import CODE_ROOT, FIGURE_DIR, PROJECT_ROOT, RESULTS_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_NAME = "ATUAV"
PRIMARY_MODEL_NAME = "HGTAN"


CAPABILITY_FEATURES = [
    "target_type",
    "payload_capability",
    "adversarial_capability",
    "endurance_margin",
]

INTENT_FEATURES = [
    "mission_type",
    "coordination_level",
    "heading_angle",
    "route_deviation",
]

OPPORTUNITY_FEATURES = [
    "distance",
    "velocity",
    "altitude",
    "time_to_arrival",
]

CONTEXT_FEATURES = [
    "swarm_size",
    "defense_capability",
    "target_asset_value",
    "track_confidence",
]

ALL_FEATURES = CAPABILITY_FEATURES + INTENT_FEATURES + OPPORTUNITY_FEATURES + CONTEXT_FEATURES
N_FEATURES = len(ALL_FEATURES)

FEATURE_GROUPS = {
    "capability": {"features": CAPABILITY_FEATURES, "dim": len(CAPABILITY_FEATURES)},
    "intent": {"features": INTENT_FEATURES, "dim": len(INTENT_FEATURES)},
    "opportunity": {"features": OPPORTUNITY_FEATURES, "dim": len(OPPORTUNITY_FEATURES)},
    "context": {"features": CONTEXT_FEATURES, "dim": len(CONTEXT_FEATURES)},
}
GROUP_DIMS = [4, 4, 4, 4]
NUM_GROUPS = len(GROUP_DIMS)

CATEGORICAL_FEATURES = {
    "target_type": {"index": 0, "n_categories": 4},
    "mission_type": {"index": 4, "n_categories": 4},
}

TARGET_TYPE_LABELS = {
    0: "ISR_UAV",
    1: "EW_UAV",
    2: "Strike_UAV",
    3: "Loitering_Munition",
}

MISSION_TYPE_LABELS = {
    0: "Recon_Probe",
    1: "EW_Harass",
    2: "Precision_Strike",
    3: "Saturation_Breakthrough",
}

CONTINUOUS_FEATURE_INDICES = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

FEATURE_METADATA = {
    "target_type": {
        "group": "capability",
        "description": "Air-target platform type.",
        "range": "{0,1,2,3}",
        "effect": "higher_type_usually_more_lethal",
    },
    "payload_capability": {
        "group": "capability",
        "description": "Payload and strike capability.",
        "range": "[0,1]",
        "effect": "higher_more_threat",
    },
    "adversarial_capability": {
        "group": "capability",
        "description": "Stealth, jamming, autonomy, and adversarial capability.",
        "range": "[0,1]",
        "effect": "higher_more_threat",
    },
    "endurance_margin": {
        "group": "capability",
        "description": "Remaining endurance and mission persistence.",
        "range": "[0,1]",
        "effect": "higher_more_threat",
    },
    "mission_type": {
        "group": "intent",
        "description": "Mission intent category.",
        "range": "{0,1,2,3}",
        "effect": "higher_type_usually_more_aggressive",
    },
    "coordination_level": {
        "group": "intent",
        "description": "Swarm coordination and formation consistency.",
        "range": "[0,1]",
        "effect": "higher_more_threat",
    },
    "heading_angle": {
        "group": "intent",
        "description": "Normalized heading angle relative to the defended asset.",
        "range": "[0,1]",
        "effect": "lower_more_threat",
    },
    "route_deviation": {
        "group": "intent",
        "description": "Deviation from nominal/civil air corridors.",
        "range": "[0,1]",
        "effect": "higher_more_threat",
    },
    "distance": {
        "group": "opportunity",
        "description": "Normalized distance to the defended asset.",
        "range": "[0,1]",
        "effect": "lower_more_threat",
    },
    "velocity": {
        "group": "opportunity",
        "description": "Normalized flight speed.",
        "range": "[0,1]",
        "effect": "higher_more_threat",
    },
    "altitude": {
        "group": "opportunity",
        "description": "Normalized altitude under air-defense engagement constraints.",
        "range": "[0,1]",
        "effect": "context_dependent",
    },
    "time_to_arrival": {
        "group": "opportunity",
        "description": "Estimated time before arrival at the defended asset.",
        "range": "[0,1]",
        "effect": "lower_more_threat_and_urgency",
    },
    "swarm_size": {
        "group": "context",
        "description": "Relative swarm size and saturation pressure.",
        "range": "[0,1]",
        "effect": "higher_more_threat",
    },
    "defense_capability": {
        "group": "context",
        "description": "Available sensing, tracking, and interception capability.",
        "range": "[0,1]",
        "effect": "higher_less_threat",
    },
    "target_asset_value": {
        "group": "context",
        "description": "Operational value of the defended asset.",
        "range": "[0,1]",
        "effect": "higher_more_threat",
    },
    "track_confidence": {
        "group": "context",
        "description": "Sensor and track association confidence.",
        "range": "[0,1]",
        "effect": "lower_more_operational_risk",
    },
}

FEATURE_RISK_DIRECTION = {
    "target_type": "high",
    "payload_capability": "high",
    "adversarial_capability": "high",
    "endurance_margin": "high",
    "mission_type": "high",
    "coordination_level": "high",
    "heading_angle": "low",
    "route_deviation": "high",
    "distance": "low",
    "velocity": "high",
    "altitude": "low",
    "time_to_arrival": "low",
    "swarm_size": "high",
    "defense_capability": "low",
    "target_asset_value": "high",
    "track_confidence": "low",
}

PRIOR_WEIGHTS = {
    "target_type": 0.05,
    "payload_capability": 0.08,
    "adversarial_capability": 0.07,
    "endurance_margin": 0.04,
    "mission_type": 0.07,
    "coordination_level": 0.07,
    "heading_angle": 0.05,
    "route_deviation": 0.05,
    "distance": 0.08,
    "velocity": 0.06,
    "altitude": 0.04,
    "time_to_arrival": 0.10,
    "swarm_size": 0.07,
    "defense_capability": 0.08,
    "target_asset_value": 0.06,
    "track_confidence": 0.03,
}
PRIOR_WEIGHTS_TENSOR = [PRIOR_WEIGHTS[name] for name in ALL_FEATURES]

THREAT_LEVELS = {
    1: "Surveillance",
    2: "Harassment",
    3: "Precision_Strike",
    4: "Saturation_Attack",
    5: "Swarm_Penetration",
}
N_CLASSES = len(THREAT_LEVELS)

URGENCY_LEVELS = {
    1: "Potential",
    2: "Imminent",
    3: "Immediate",
}
N_URGENCY = len(URGENCY_LEVELS)

CRITICAL_THREAT_CLASSES = [4, 5]
CRITICAL_URGENCY_CLASSES = [3]


class HGTANConfig:
    """Central defaults for data generation, model construction, and training."""

    TRAIN_MODE = "repro"
    TRAIN_MODE_PRESETS = {
        "gpu": {
            "batch_size": 1024,
            "deterministic": False,
            "cudnn_benchmark": True,
            "allow_tf32": True,
            "matmul_precision": "high",
            "num_runs": 3,
            "random_seeds": [42, 123, 456],
            "num_epochs": 80,
            "patience": 20,
            "num_workers": 8,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "use_amp": True,
            "compile_model": False,
            "require_cuda": True,
            "adaptive_batching": True,
            "min_train_steps_per_epoch": 4,
            "min_batch_size": 64,
            "adaptive_num_workers": True,
        },
        "speed": {
            "batch_size": 512,
            "deterministic": False,
            "cudnn_benchmark": True,
            "allow_tf32": True,
            "matmul_precision": "high",
            "num_runs": 2,
            "random_seeds": [42, 123],
            "num_epochs": 50,
            "patience": 15,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "use_amp": True,
            "compile_model": False,
            "require_cuda": False,
            "adaptive_batching": True,
            "min_train_steps_per_epoch": 4,
            "min_batch_size": 64,
            "adaptive_num_workers": True,
        },
        "repro": {
            "batch_size": 256,
            "deterministic": True,
            "cudnn_benchmark": False,
            "allow_tf32": False,
            "matmul_precision": "highest",
            "num_runs": 5,
            "random_seeds": [42, 123, 456, 789, 1024],
            "num_epochs": 150,
            "patience": 40,
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "use_amp": False,
            "compile_model": False,
            "require_cuda": False,
            "adaptive_batching": True,
            "min_train_steps_per_epoch": 4,
            "min_batch_size": 32,
            "adaptive_num_workers": True,
        },
    }

    DATA = {
        "n_samples": 4000,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "split_strategy": "stratified",
        "scenario_holdout_ratio": 0.2,
        "noise_std": 0.02,
        "boundary_ratio": 0.08,
        "confusing_ratio": 0.03,
        "min_class_samples": 40,
    }

    SEQUENCE = {
        "seq_len": 64,
        "observed_len": 64,
        "frame_interval": 0.2,
        "track_noise_std": 0.015,
        "range_m": 1000,
        "track_missing_ratio": 0.0,
        "track_jitter_std": 0.0,
        "sensor_profile": "ATUAV-Core",
        "type_as_input": False,
    }

    MODEL = {
        "embed_dim": 128,
        "num_heads": 4,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.12,
        "use_prior_weights": True,
        "prior_weight_alpha": 0.3,
    }

    BASELINE = {
        "mlp_hidden_dim": 128,
        "mlp_dropout": 0.1,
        "transformer_embed_dim": 128,
        "transformer_num_heads": 4,
        "transformer_num_layers": 2,
        "transformer_dropout": 0.1,
    }

    TRAIN = {
        "batch_size": 256,
        "learning_rate": 8e-4,
        "num_epochs": 150,
        "weight_decay": 5e-4,
        "urgency_weight": 0.25,
        "threat_weight": 0.75,
        "patience": 40,
        "min_delta": 1e-4,
        "min_epochs": 60,
        "ema_alpha": 0.3,
        "warmup_epochs": 8,
        "lr_decay_factor": 0.01,
        "loss_type_threat": "ce",
        "loss_type_urgency": "ce",
        "focal_gamma": 2.0,
        "label_smoothing": 0.05,
        "auto_class_weight": True,
        "class_weight_max": 4.0,
        "class_weight_threat": None,
        "class_weight_urgency": None,
        "gradient_clip_norm": 1.0,
        "use_two_stage": False,
        "num_epochs_stage1": 80,
        "num_epochs_stage2": 70,
        "stage2_unfreeze_last_sam": True,
        "use_mixup": True,
        "mixup_alpha": 0.15,
        "ablation_train_recipe": "joint",
        "ablation_reference_variant": "HGTAN (Core)",
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "use_amp": False,
        "compile_model": False,
        "adaptive_batching": True,
        "min_train_steps_per_epoch": 4,
        "min_batch_size": 32,
        "adaptive_num_workers": True,
    }

    RUN = {
        "generate_figures": True,
        "run_ablation": False,
        "run_sensitivity": False,
        "export_pdf": True,
        "verbose": False,
        "require_cuda": False,
    }

    REPRODUCIBILITY = {
        "deterministic": True,
        "cudnn_benchmark": False,
        "allow_tf32": False,
        "matmul_precision": "highest",
    }

    @classmethod
    def get_config(cls, mode: str | None = None) -> dict[str, Any]:
        """Return a complete config dictionary with the selected mode preset."""
        config = {
            "data": deepcopy(cls.DATA),
            "sequence": deepcopy(cls.SEQUENCE),
            "model": deepcopy(cls.MODEL),
            "baseline": deepcopy(cls.BASELINE),
            "train": deepcopy(cls.TRAIN),
            "run": deepcopy(cls.RUN),
            "reproducibility": deepcopy(cls.REPRODUCIBILITY),
        }

        selected_mode = mode or cls.TRAIN_MODE
        if selected_mode not in cls.TRAIN_MODE_PRESETS:
            valid = ", ".join(sorted(cls.TRAIN_MODE_PRESETS))
            raise ValueError(f"Unknown training mode={selected_mode!r}. Valid modes: {valid}")

        preset = cls.TRAIN_MODE_PRESETS[selected_mode]
        config["train"]["batch_size"] = preset["batch_size"]
        config["train"]["num_epochs"] = preset["num_epochs"]
        config["train"]["patience"] = preset["patience"]
        config["train"]["num_workers"] = preset["num_workers"]
        config["train"]["pin_memory"] = preset["pin_memory"]
        config["train"]["persistent_workers"] = preset["persistent_workers"]
        config["train"]["prefetch_factor"] = preset["prefetch_factor"]
        config["train"]["use_amp"] = preset["use_amp"]
        config["train"]["compile_model"] = preset["compile_model"]
        config["train"]["adaptive_batching"] = preset.get("adaptive_batching", config["train"]["adaptive_batching"])
        config["train"]["min_train_steps_per_epoch"] = preset.get(
            "min_train_steps_per_epoch",
            config["train"]["min_train_steps_per_epoch"],
        )
        config["train"]["min_batch_size"] = preset.get("min_batch_size", config["train"]["min_batch_size"])
        config["train"]["adaptive_num_workers"] = preset.get(
            "adaptive_num_workers",
            config["train"]["adaptive_num_workers"],
        )
        config["run"]["num_runs"] = preset["num_runs"]
        config["run"]["seeds"] = list(preset["random_seeds"])
        config["run"]["require_cuda"] = preset.get("require_cuda", False)
        config["reproducibility"]["deterministic"] = preset["deterministic"]
        config["reproducibility"]["cudnn_benchmark"] = preset["cudnn_benchmark"]
        config["reproducibility"]["allow_tf32"] = preset["allow_tf32"]
        config["reproducibility"]["matmul_precision"] = preset["matmul_precision"]
        return config

    @classmethod
    def get_experiment_config(cls, experiment_type: str = "default", mode: str | None = None) -> dict[str, Any]:
        """Return experiment defaults; experiment_type is reserved for future presets."""
        del experiment_type
        return cls.get_config(mode)

    @classmethod
    def get_model_config(cls, model_type: str = "hgtan") -> dict[str, Any]:
        """Return model defaults for HGTAN or generic baselines."""
        return deepcopy(cls.MODEL if model_type == "hgtan" else cls.BASELINE)

    @classmethod
    def get_train_config(cls, mode: str | None = None) -> dict[str, Any]:
        """Return training defaults with a backward-compatible lr alias."""
        config = cls.get_config(mode)["train"]
        config["lr"] = config["learning_rate"]
        return config


EMBED_DIM = HGTANConfig.MODEL["embed_dim"]
NUM_HEADS = HGTANConfig.MODEL["num_heads"]
HIDDEN_DIM = HGTANConfig.MODEL["hidden_dim"]
NUM_LAYERS = HGTANConfig.MODEL["num_layers"]
DROPOUT = HGTANConfig.MODEL["dropout"]

MLP_HIDDEN_DIM = HGTANConfig.BASELINE["mlp_hidden_dim"]
MLP_DROPOUT = HGTANConfig.BASELINE["mlp_dropout"]
TRANS_EMBED_DIM = HGTANConfig.BASELINE["transformer_embed_dim"]
TRANS_NUM_HEADS = HGTANConfig.BASELINE["transformer_num_heads"]
TRANS_NUM_LAYERS = HGTANConfig.BASELINE["transformer_num_layers"]
TRANS_DROPOUT = HGTANConfig.BASELINE["transformer_dropout"]

BATCH_SIZE = HGTANConfig.TRAIN["batch_size"]
LEARNING_RATE = HGTANConfig.TRAIN["learning_rate"]
NUM_EPOCHS = HGTANConfig.TRAIN["num_epochs"]
WEIGHT_DECAY = HGTANConfig.TRAIN["weight_decay"]
URGENCY_WEIGHT = HGTANConfig.TRAIN["urgency_weight"]
PATIENCE = HGTANConfig.TRAIN["patience"]

N_SAMPLES = HGTANConfig.DATA["n_samples"]
TRAIN_RATIO = HGTANConfig.DATA["train_ratio"]


def set_random_seed(seed: int, config: dict[str, Any] | None = None) -> None:
    """Set Python, NumPy, and PyTorch seeds with optional reproducibility flags."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    repro = (config or {}).get("reproducibility", {})
    deterministic = repro.get("deterministic", True)
    benchmark = repro.get("cudnn_benchmark", False)
    allow_tf32 = repro.get("allow_tf32", False)
    matmul_precision = repro.get("matmul_precision", "highest")

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)


def get_run_seeds(run_config: dict[str, Any]) -> list[int]:
    """Return the configured seed list, extending it deterministically if needed."""
    seeds = list(run_config.get("seeds", []))
    num_runs = int(run_config.get("num_runs", 1))
    base_seed = int(run_config.get("seed", 42))
    idx = 0
    while len(seeds) < num_runs:
        seeds.append(base_seed + (idx + 1) * 13)
        idx += 1
    return seeds[:num_runs]


def to_serializable(obj: Any) -> Any:
    """Recursively convert numpy, torch, and Path objects into JSON-safe values."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def get_prior_weights_tensor() -> torch.Tensor:
    """Return prior indicator weights as a float tensor."""
    return torch.tensor(PRIOR_WEIGHTS_TENSOR, dtype=torch.float32)


def validate_class_distribution(labels: np.ndarray, min_samples: int | None = None) -> bool:
    """Check that every observed label class has enough samples."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return False
    if min_samples is None:
        min_samples = HGTANConfig.DATA.get("min_class_samples", 30)

    unique, counts = np.unique(labels, return_counts=True)
    expected = np.arange(int(unique.min()), int(unique.max()) + 1)
    if set(unique.tolist()) != set(expected.tolist()):
        return False
    return bool(np.all(counts >= min_samples))
