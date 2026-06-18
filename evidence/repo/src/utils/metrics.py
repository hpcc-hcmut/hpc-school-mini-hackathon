# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""Metric calculation and CSV helpers."""

import csv
from pathlib import Path
from typing import Any, Iterable

import torch


STEP_FIELDS = [
    "job_id",
    "experiment_name",
    "model_name",
    "epoch",
    "step",
    "batch_size",
    "num_workers",
    "cpus_per_task",
    "loss",
    "top1",
    "top5",
    "data_time_sec",
    "compute_time_sec",
    "step_time_sec",
    "samples_per_sec",
    "torch_mem_allocated_mb",
    "torch_mem_reserved_mb",
    "torch_max_mem_allocated_mb",
    "torch_max_mem_reserved_mb",
]

EPOCH_FIELDS = [
    "job_id",
    "experiment_name",
    "model_name",
    "epoch",
    "batch_size",
    "num_workers",
    "cpus_per_task",
    "train_loss",
    "train_top1",
    "train_top5",
    "val_loss",
    "val_top1",
    "val_top5",
    "epoch_time_sec",
    "avg_samples_per_sec",
    "avg_step_time_sec",
    "avg_data_time_sec",
    "avg_compute_time_sec",
    "data_time_ratio",
    "torch_peak_allocated_mb",
    "torch_peak_reserved_mb",
]

TRAINING_SUMMARY_FIELDS = [
    "job_id",
    "experiment_name",
    "model_name",
    "batch_size",
    "num_workers",
    "cpus_per_task",
    "epochs",
    "optimizer",
    "learning_rate",
    "scheduler",
    "amp",
    "total_runtime_sec",
    "avg_samples_per_sec",
    "avg_step_time_sec",
    "avg_data_time_sec",
    "avg_compute_time_sec",
    "data_time_ratio",
    "best_val_loss",
    "best_val_top1",
    "best_val_top5",
    "final_test_loss",
    "final_test_top1",
    "final_test_top5",
    "torch_peak_allocated_mb",
    "torch_peak_reserved_mb",
    "success",
    "error",
]


def accuracy_counts(
    logits: torch.Tensor, targets: torch.Tensor, topk: tuple[int, ...] = (1, 5)
) -> list[int]:
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits, got shape {tuple(logits.shape)}")
    max_k = min(max(topk), logits.size(1))
    predictions = logits.topk(max_k, dim=1, largest=True, sorted=True).indices
    matches = predictions.eq(targets.view(-1, 1))
    return [int(matches[:, : min(k, max_k)].any(dim=1).sum().item()) for k in topk]


def cuda_memory_metrics() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "torch_mem_allocated_mb": 0.0,
            "torch_mem_reserved_mb": 0.0,
            "torch_max_mem_allocated_mb": 0.0,
            "torch_max_mem_reserved_mb": 0.0,
        }
    divisor = 1024**2
    return {
        "torch_mem_allocated_mb": torch.cuda.memory_allocated() / divisor,
        "torch_mem_reserved_mb": torch.cuda.memory_reserved() / divisor,
        "torch_max_mem_allocated_mb": torch.cuda.max_memory_allocated() / divisor,
        "torch_max_mem_reserved_mb": torch.cuda.max_memory_reserved() / divisor,
    }


def append_csv_row(
    path: str | Path, row: dict[str, Any], fieldnames: list[str]
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_csv_rows(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)
