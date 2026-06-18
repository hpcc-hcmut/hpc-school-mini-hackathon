# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""Measured training loop for one epoch."""

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.metrics import (
    STEP_FIELDS,
    accuracy_counts,
    append_csv_row,
    cuda_memory_metrics,
    rounded,
)


def train_one_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    gradient_accumulation_steps: int,
    log_every: int,
    step_metrics_path: Path,
    metadata: dict[str, Any],
) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_samples = 0
    data_times: list[float] = []
    compute_times: list[float] = []
    step_times: list[float] = []

    number_of_batches = len(dataloader)
    data_start = time.perf_counter()

    for batch_index, (images, targets) in enumerate(dataloader, start=1):
        data_end = time.perf_counter()
        data_time = data_end - data_start
        compute_start = time.perf_counter()

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        group_start = (
            (batch_index - 1) // gradient_accumulation_steps
        ) * gradient_accumulation_steps + 1
        group_end = min(
            group_start + gradient_accumulation_steps - 1,
            number_of_batches,
        )
        accumulation_group_size = group_end - group_start + 1
        (loss / accumulation_group_size).backward()

        should_step = (
            batch_index % gradient_accumulation_steps == 0
            or batch_index == number_of_batches
        )
        if should_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        compute_time = time.perf_counter() - compute_start
        step_time = data_time + compute_time

        batch_size = targets.size(0)
        top1_count, top5_count = accuracy_counts(outputs.detach(), targets)
        batch_top1 = 100.0 * top1_count / batch_size
        batch_top5 = 100.0 * top5_count / batch_size
        samples_per_sec = batch_size / step_time if step_time > 0 else 0.0

        total_loss += loss.item() * batch_size
        total_top1 += top1_count
        total_top5 += top5_count
        total_samples += batch_size
        data_times.append(data_time)
        compute_times.append(compute_time)
        step_times.append(step_time)
        global_step += 1

        memory = cuda_memory_metrics()
        append_csv_row(
            step_metrics_path,
            {
                **metadata,
                "epoch": epoch,
                "step": global_step,
                "loss": rounded(loss.item()),
                "top1": rounded(batch_top1),
                "top5": rounded(batch_top5),
                "data_time_sec": rounded(data_time),
                "compute_time_sec": rounded(compute_time),
                "step_time_sec": rounded(step_time),
                "samples_per_sec": rounded(samples_per_sec, 4),
                **{key: rounded(value, 4) for key, value in memory.items()},
            },
            STEP_FIELDS,
        )

        if global_step == 1 or global_step % log_every == 0:
            print(
                f"epoch={epoch} step={global_step} loss={loss.item():.4f} "
                f"top1={batch_top1:.2f} top5={batch_top5:.2f} "
                f"data={data_time:.4f}s compute={compute_time:.4f}s "
                f"throughput={samples_per_sec:.2f} samples/s",
                flush=True,
            )

        data_start = time.perf_counter()

    if total_samples == 0:
        raise RuntimeError("Training dataset is empty")

    total_step_time = sum(step_times)
    avg_step_time = total_step_time / len(step_times)
    avg_data_time = sum(data_times) / len(data_times)
    avg_compute_time = sum(compute_times) / len(compute_times)
    return (
        {
            "loss": total_loss / total_samples,
            "top1": 100.0 * total_top1 / total_samples,
            "top5": 100.0 * total_top5 / total_samples,
            "samples": float(total_samples),
            "step_count": float(len(step_times)),
            "total_step_time": total_step_time,
            "total_data_time": sum(data_times),
            "total_compute_time": sum(compute_times),
            "avg_samples_per_sec": total_samples / total_step_time,
            "avg_step_time_sec": avg_step_time,
            "avg_data_time_sec": avg_data_time,
            "avg_compute_time_sec": avg_compute_time,
            "data_time_ratio": (
                sum(data_times) / total_step_time if total_step_time > 0 else 0.0
            ),
        },
        global_step,
    )
