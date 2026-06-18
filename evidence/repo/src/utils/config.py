# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""Configuration loading and validation for the mini-hackathon."""

import json
import warnings
from pathlib import Path
from typing import Any


ALLOWED_MODELS = (
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
)
ALLOWED_BATCH_SIZES = (32, 64, 128, 256, 512)
ALLOWED_NUM_WORKERS = (0, 2, 4, 8)
ALLOWED_CPUS_PER_TASK = (1, 2, 4, 8)
ALLOWED_AUGMENTATIONS = ("none", "basic")

REQUIRED_FIELDS = (
    "experiment_name",
    "model_name",
    "batch_size",
    "num_workers",
    "cpus_per_task",
    "dataset_dir",
    "image_size",
    "num_classes",
    "epochs",
    "optimizer",
    "learning_rate",
    "momentum",
    "weight_decay",
    "scheduler",
    "amp",
    "gradient_accumulation_steps",
    "seed",
    "augmentation",
    "log_every",
    "save_best_checkpoint",
    "save_latest_checkpoint",
)

INSTRUCTOR_DEFAULTS = {
    "epochs": 20,
    "optimizer": "sgd",
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "scheduler": "cosine",
    "gradient_accumulation_steps": 1,
    "seed": 42,
    "augmentation": "basic",
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {config_path} at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a JSON object: {config_path}")
    return config


def _require_int(config: dict[str, Any], field: str) -> int:
    value = config[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return value


def validate_config(config: dict[str, Any]) -> list[str]:
    missing = [field for field in REQUIRED_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Config is missing required fields: {', '.join(missing)}")

    errors: list[str] = []
    model_name = str(config["model_name"]).lower()
    if model_name not in ALLOWED_MODELS:
        errors.append(
            f"model_name must be one of {list(ALLOWED_MODELS)}, "
            f"got {config['model_name']!r}"
        )

    batch_size = _require_int(config, "batch_size")
    num_workers = _require_int(config, "num_workers")
    cpus_per_task = _require_int(config, "cpus_per_task")
    image_size = _require_int(config, "image_size")
    num_classes = _require_int(config, "num_classes")

    if batch_size not in ALLOWED_BATCH_SIZES:
        errors.append(
            f"batch_size must be one of {list(ALLOWED_BATCH_SIZES)}, got {batch_size}"
        )
    if num_workers not in ALLOWED_NUM_WORKERS:
        errors.append(
            f"num_workers must be one of {list(ALLOWED_NUM_WORKERS)}, got {num_workers}"
        )
    if cpus_per_task not in ALLOWED_CPUS_PER_TASK:
        errors.append(
            "cpus_per_task must be one of "
            f"{list(ALLOWED_CPUS_PER_TASK)}, got {cpus_per_task}"
        )
    if num_workers > cpus_per_task:
        errors.append(
            f"num_workers ({num_workers}) must not exceed cpus_per_task "
            f"({cpus_per_task})"
        )
    if config["amp"] is not False:
        errors.append("amp must be false for the main mini-hackathon")
    if image_size != 32:
        errors.append(f"image_size must be 32 for CIFAR-100, got {image_size}")
    if num_classes != 100:
        errors.append(f"num_classes must be 100 for CIFAR-100, got {num_classes}")

    experiment_name = str(config["experiment_name"]).strip()
    if not experiment_name:
        errors.append("experiment_name must not be empty")
    if any(character in experiment_name for character in ("/", "\\")):
        errors.append("experiment_name must not contain path separators")

    if str(config["optimizer"]).lower() != "sgd":
        errors.append("optimizer must be 'sgd'")
    if str(config["scheduler"]).lower() != "cosine":
        errors.append("scheduler must be 'cosine'")
    if str(config["augmentation"]).lower() not in ALLOWED_AUGMENTATIONS:
        errors.append(
            f"augmentation must be one of {list(ALLOWED_AUGMENTATIONS)}"
        )
    if _require_int(config, "epochs") <= 0:
        errors.append("epochs must be positive")
    if _require_int(config, "gradient_accumulation_steps") <= 0:
        errors.append("gradient_accumulation_steps must be positive")
    if _require_int(config, "log_every") <= 0:
        errors.append("log_every must be positive")

    for field in ("learning_rate", "momentum", "weight_decay"):
        value = config[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{field} must be numeric, got {value!r}")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int):
        errors.append(f"seed must be an integer, got {config['seed']!r}")
    for field in ("save_best_checkpoint", "save_latest_checkpoint"):
        if not isinstance(config[field], bool):
            errors.append(f"{field} must be true or false")

    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))

    warning_messages: list[str] = []
    for field, default in INSTRUCTOR_DEFAULTS.items():
        if config[field] != default:
            warning_messages.append(
                f"{field}={config[field]!r} differs from the instructor default "
                f"{default!r}"
            )
    for message in warning_messages:
        warnings.warn(message, UserWarning, stacklevel=2)

    return warning_messages
