# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""Evaluation loop shared by training and standalone evaluation."""

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.metrics import accuracy_counts


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        top1_count, top5_count = accuracy_counts(outputs, targets)
        total_loss += loss.item() * batch_size
        total_top1 += top1_count
        total_top5 += top5_count
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Cannot evaluate an empty dataset")

    return {
        "loss": total_loss / total_samples,
        "top1": 100.0 * total_top1 / total_samples,
        "top5": 100.0 * total_top5 / total_samples,
        "samples": float(total_samples),
    }
