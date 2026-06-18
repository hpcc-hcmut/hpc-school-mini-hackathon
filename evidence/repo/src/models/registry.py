# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""Small registry for the custom CIFAR ResNet implementations."""

from collections.abc import Callable
from typing import Any

import torch.nn as nn


ModelBuilder = Callable[..., nn.Module]
_MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    normalized_name = name.strip().lower()

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if normalized_name in _MODEL_REGISTRY:
            raise ValueError(f"Model {normalized_name!r} is already registered")
        _MODEL_REGISTRY[normalized_name] = builder
        return builder

    return decorator


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_REGISTRY))


def resolve_num_classes(num_classes: int, kwargs: dict[str, Any]) -> int:
    """Resolve class count while rejecting pretrained-model arguments."""
    for key in ("pretrained", "weights"):
        if key in kwargs:
            raise ValueError(
                f"{key!r} is not supported. This repository trains custom ResNets "
                "from scratch."
            )

    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected model arguments: {unexpected}")

    resolved = int(num_classes)
    if resolved <= 0:
        raise ValueError("num_classes must be positive")
    return resolved


def create_model(name: str, num_classes: int, **kwargs: Any) -> nn.Module:
    normalized_name = name.strip().lower()
    try:
        builder = _MODEL_REGISTRY[normalized_name]
    except KeyError as exc:
        allowed = ", ".join(available_models())
        raise ValueError(
            f"Unknown model {name!r}. Allowed custom models: {allowed}"
        ) from exc
    return builder(num_classes=num_classes, **kwargs)
