# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""ImageFolder datasets and deterministic data loaders."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data.transforms import build_transforms
from utils.seed import seed_worker


EXPECTED_SPLIT_SIZES = {"train": 45_000, "val": 5_000, "test": 10_000}


def load_datasets(
    dataset_dir: str | Path, augmentation: str
) -> dict[str, ImageFolder]:
    root = Path(dataset_dir)
    train_transform, evaluation_transform = build_transforms(augmentation)
    transforms_by_split = {
        "train": train_transform,
        "val": evaluation_transform,
        "test": evaluation_transform,
    }

    datasets: dict[str, ImageFolder] = {}
    for split, transform in transforms_by_split.items():
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Missing {split!r} split at {split_dir}. Prepare CIFAR-100 first."
            )
        datasets[split] = ImageFolder(split_dir, transform=transform)

    reference_classes = datasets["train"].classes
    for split, dataset in datasets.items():
        if dataset.classes != reference_classes:
            raise ValueError(
                f"Class folders in {split} do not match the train split"
            )
        expected = EXPECTED_SPLIT_SIZES[split]
        if len(dataset) != expected:
            raise ValueError(
                f"Expected {expected} images in {split}, found {len(dataset)}"
            )
    if len(reference_classes) != 100:
        raise ValueError(
            f"Expected 100 CIFAR-100 class folders, found {len(reference_classes)}"
        )
    return datasets


def create_dataloaders(
    dataset_dir: str | Path,
    batch_size: int,
    num_workers: int,
    augmentation: str,
    seed: int,
) -> tuple[dict[str, DataLoader[Any]], dict[str, ImageFolder]]:
    datasets = load_datasets(dataset_dir, augmentation)
    generator = torch.Generator()
    generator.manual_seed(seed)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            shuffle=True,
            generator=generator,
            **common,
        ),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
    return loaders, datasets
