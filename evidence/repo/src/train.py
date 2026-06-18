# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""Train and measure a custom ResNet on CIFAR-100 ImageFolder data."""

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import nn

from data.datasets import create_dataloaders
from engine.checkpointing import load_model_checkpoint, save_checkpoint
from engine.evaluator import evaluate_model
from engine.trainer import train_one_epoch
from models import create_model
from utils.config import load_config, validate_config
from utils.metrics import (
    EPOCH_FIELDS,
    TRAINING_SUMMARY_FIELDS,
    append_csv_row,
    cuda_memory_metrics,
    rounded,
    write_csv_rows,
)
from utils.seed import seed_everything
from utils.slurm import resolve_cpus_per_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a custom CIFAR ResNet and write resource metrics."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--job-id", default=os.environ.get("SLURM_JOB_ID", "nojob")
    )
    parser.add_argument(
        "--cpus-per-task",
        default=os.environ.get("SLURM_CPUS_PER_TASK", "unknown"),
    )
    return parser.parse_args()


def optional_round(value: float | None, digits: int = 6) -> float | str:
    return "" if value is None else rounded(value, digits)


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = {}
    experiment_name = Path(args.config).stem
    model_name = ""
    cpus_per_task: int | str = args.cpus_per_task
    success = False
    error_message = ""
    start_time = time.perf_counter()

    total_samples = 0.0
    total_steps = 0.0
    total_step_time = 0.0
    total_data_time = 0.0
    total_compute_time = 0.0
    best_val_loss: float | None = None
    best_val_top1: float | None = None
    best_val_top5: float | None = None
    final_test_loss: float | None = None
    final_test_top1: float | None = None
    final_test_top5: float | None = None
    global_peak_allocated = 0.0
    global_peak_reserved = 0.0

    results_dir = Path("results")
    checkpoints_dir = Path("checkpoints")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = load_config(args.config)
        experiment_name = str(config.get("experiment_name", experiment_name))
        model_name = str(config.get("model_name", ""))
        validate_config(config)
        cpus_per_task = resolve_cpus_per_task(
            args.cpus_per_task, int(config["cpus_per_task"])
        )
        if int(config["num_workers"]) > cpus_per_task:
            raise ValueError(
                f"num_workers ({config['num_workers']}) exceeds the actual Slurm "
                f"allocation ({cpus_per_task} CPUs)"
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Request a Slurm GPU and run Apptainer "
                "with --nv."
            )

        seed_everything(int(config["seed"]))
        device = torch.device("cuda")
        print("Training configuration:")
        print(json.dumps(config, indent=2))
        print(f"Job ID: {args.job_id}")
        print(f"Actual CPUs per task: {cpus_per_task}")
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

        loaders, datasets = create_dataloaders(
            dataset_dir=config["dataset_dir"],
            batch_size=int(config["batch_size"]),
            num_workers=int(config["num_workers"]),
            augmentation=str(config["augmentation"]),
            seed=int(config["seed"]),
        )
        print(
            "Dataset counts: "
            + ", ".join(f"{name}={len(data)}" for name, data in datasets.items())
        )

        model = create_model(
            name=config["model_name"],
            num_classes=config["num_classes"],
        )
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(config["learning_rate"]),
            momentum=float(config["momentum"]),
            weight_decay=float(config["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(config["epochs"])
        )

        step_metrics_path = (
            results_dir
            / f"step_metrics_{args.job_id}_{experiment_name}.csv"
        )
        epoch_metrics_path = (
            results_dir
            / f"epoch_metrics_{args.job_id}_{experiment_name}.csv"
        )
        for path in (step_metrics_path, epoch_metrics_path):
            if path.exists():
                path.unlink()

        best_checkpoint_path = (
            checkpoints_dir / f"best_{args.job_id}_{experiment_name}.pt"
        )
        latest_checkpoint_path = (
            checkpoints_dir / f"latest_{args.job_id}_{experiment_name}.pt"
        )
        metadata = {
            "job_id": args.job_id,
            "experiment_name": experiment_name,
            "model_name": model_name,
            "batch_size": int(config["batch_size"]),
            "num_workers": int(config["num_workers"]),
            "cpus_per_task": cpus_per_task,
        }
        global_step = 0

        for epoch in range(1, int(config["epochs"]) + 1):
            epoch_start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            train_stats, global_step = train_one_epoch(
                model=model,
                dataloader=loaders["train"],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                global_step=global_step,
                gradient_accumulation_steps=int(
                    config["gradient_accumulation_steps"]
                ),
                log_every=int(config["log_every"]),
                step_metrics_path=step_metrics_path,
                metadata=metadata,
            )
            memory = cuda_memory_metrics()
            epoch_peak_allocated = memory["torch_max_mem_allocated_mb"]
            epoch_peak_reserved = memory["torch_max_mem_reserved_mb"]
            global_peak_allocated = max(
                global_peak_allocated, epoch_peak_allocated
            )
            global_peak_reserved = max(global_peak_reserved, epoch_peak_reserved)

            val_stats = evaluate_model(
                model, loaders["val"], criterion, device
            )
            is_best = (
                best_val_top1 is None or val_stats["top1"] > best_val_top1
            )
            if is_best:
                best_val_loss = val_stats["loss"]
                best_val_top1 = val_stats["top1"]
                best_val_top5 = val_stats["top5"]

            scheduler.step()
            if config["save_best_checkpoint"] and is_best:
                save_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_top1=float(best_val_top1),
                    config=config,
                )
            if config["save_latest_checkpoint"]:
                save_checkpoint(
                    latest_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_top1=float(best_val_top1),
                    config=config,
                )

            epoch_time = time.perf_counter() - epoch_start
            append_csv_row(
                epoch_metrics_path,
                {
                    **metadata,
                    "epoch": epoch,
                    "train_loss": rounded(train_stats["loss"]),
                    "train_top1": rounded(train_stats["top1"]),
                    "train_top5": rounded(train_stats["top5"]),
                    "val_loss": rounded(val_stats["loss"]),
                    "val_top1": rounded(val_stats["top1"]),
                    "val_top5": rounded(val_stats["top5"]),
                    "epoch_time_sec": rounded(epoch_time),
                    "avg_samples_per_sec": rounded(
                        train_stats["avg_samples_per_sec"], 4
                    ),
                    "avg_step_time_sec": rounded(
                        train_stats["avg_step_time_sec"]
                    ),
                    "avg_data_time_sec": rounded(
                        train_stats["avg_data_time_sec"]
                    ),
                    "avg_compute_time_sec": rounded(
                        train_stats["avg_compute_time_sec"]
                    ),
                    "data_time_ratio": rounded(
                        train_stats["data_time_ratio"]
                    ),
                    "torch_peak_allocated_mb": rounded(
                        epoch_peak_allocated, 4
                    ),
                    "torch_peak_reserved_mb": rounded(
                        epoch_peak_reserved, 4
                    ),
                },
                EPOCH_FIELDS,
            )

            total_samples += train_stats["samples"]
            total_steps += train_stats["step_count"]
            total_step_time += train_stats["total_step_time"]
            total_data_time += train_stats["total_data_time"]
            total_compute_time += train_stats["total_compute_time"]
            print(
                f"Epoch {epoch}/{config['epochs']}: "
                f"train_loss={train_stats['loss']:.4f} "
                f"train_top1={train_stats['top1']:.2f} "
                f"val_loss={val_stats['loss']:.4f} "
                f"val_top1={val_stats['top1']:.2f} "
                f"throughput={train_stats['avg_samples_per_sec']:.2f} samples/s",
                flush=True,
            )

        if config["save_best_checkpoint"] and best_checkpoint_path.is_file():
            load_model_checkpoint(best_checkpoint_path, model, device)
            print(f"Loaded best checkpoint for test evaluation: {best_checkpoint_path}")
        else:
            print("Evaluating the final model because no best checkpoint was saved")

        test_stats = evaluate_model(model, loaders["test"], criterion, device)
        final_test_loss = test_stats["loss"]
        final_test_top1 = test_stats["top1"]
        final_test_top5 = test_stats["top5"]
        success = True
        print(
            f"Test: loss={final_test_loss:.4f} top1={final_test_top1:.2f} "
            f"top5={final_test_top5:.2f}"
        )
        print(f"Step metrics: {step_metrics_path}")
        print(f"Epoch metrics: {epoch_metrics_path}")

    except Exception as exc:
        error_message = str(exc)
        print(f"ERROR: {error_message}")
        traceback.print_exc()

    total_runtime = time.perf_counter() - start_time
    summary_path = (
        results_dir / f"training_summary_{args.job_id}_{experiment_name}.csv"
    )
    average_step_time = total_step_time / total_steps if total_steps else 0.0
    average_data_time = total_data_time / total_steps if total_steps else 0.0
    average_compute_time = (
        total_compute_time / total_steps if total_steps else 0.0
    )
    average_samples_per_sec = (
        total_samples / total_step_time if total_step_time else 0.0
    )
    data_time_ratio = (
        total_data_time / total_step_time if total_step_time else 0.0
    )
    summary_row = {
        "job_id": args.job_id,
        "experiment_name": experiment_name,
        "model_name": model_name or config.get("model_name", ""),
        "batch_size": config.get("batch_size", ""),
        "num_workers": config.get("num_workers", ""),
        "cpus_per_task": cpus_per_task,
        "epochs": config.get("epochs", ""),
        "optimizer": config.get("optimizer", ""),
        "learning_rate": config.get("learning_rate", ""),
        "scheduler": config.get("scheduler", ""),
        "amp": str(config.get("amp", "")).lower(),
        "total_runtime_sec": rounded(total_runtime),
        "avg_samples_per_sec": rounded(average_samples_per_sec, 4),
        "avg_step_time_sec": rounded(average_step_time),
        "avg_data_time_sec": rounded(average_data_time),
        "avg_compute_time_sec": rounded(average_compute_time),
        "data_time_ratio": rounded(data_time_ratio),
        "best_val_loss": optional_round(best_val_loss),
        "best_val_top1": optional_round(best_val_top1),
        "best_val_top5": optional_round(best_val_top5),
        "final_test_loss": optional_round(final_test_loss),
        "final_test_top1": optional_round(final_test_top1),
        "final_test_top5": optional_round(final_test_top5),
        "torch_peak_allocated_mb": rounded(global_peak_allocated, 4),
        "torch_peak_reserved_mb": rounded(global_peak_reserved, 4),
        "success": str(success).lower(),
        "error": error_message,
    }
    write_csv_rows(
        summary_path, [summary_row], TRAINING_SUMMARY_FIELDS
    )
    print(f"Training summary: {summary_path}")

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
