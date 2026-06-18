# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""Join training and nvidia-smi summaries for run-to-run comparison."""

import argparse
import csv
from pathlib import Path


COMPARISON_FIELDS = [
    "job_id",
    "experiment_name",
    "model_name",
    "batch_size",
    "num_workers",
    "cpus_per_task",
    "epochs",
    "avg_samples_per_sec",
    "avg_step_time_sec",
    "avg_data_time_sec",
    "avg_compute_time_sec",
    "data_time_ratio",
    "best_val_top1",
    "best_val_top5",
    "final_test_top1",
    "final_test_top5",
    "torch_peak_allocated_mb",
    "torch_peak_reserved_mb",
    "nvidia_smi_max_memory_used_mb",
    "nvidia_smi_avg_memory_used_mb",
    "nvidia_smi_avg_gpu_utilization",
    "nvidia_smi_max_gpu_utilization",
    "nvidia_smi_avg_power_draw_w",
    "success",
    "error",
]


def read_rows(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_paths: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        if not path.is_file():
            raise FileNotFoundError(f"Summary file not found: {path}")
        with path.open("r", newline="", encoding="utf-8") as handle:
            file_rows = list(csv.DictReader(handle))
        if not file_rows:
            raise RuntimeError(f"No data rows in {path}")
        rows.extend(file_rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-summaries", nargs="+", required=True)
    parser.add_argument("--gpu-summaries", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    training_rows = read_rows(args.training_summaries)
    gpu_rows = read_rows(args.gpu_summaries)
    gpu_by_run = {
        (row.get("job_id", ""), row.get("experiment_name", "")): row
        for row in gpu_rows
    }

    output_rows: list[dict[str, str]] = []
    for training in training_rows:
        key = (
            training.get("job_id", ""),
            training.get("experiment_name", ""),
        )
        gpu = gpu_by_run.get(key, {})
        output_rows.append(
            {
                "job_id": training.get("job_id", ""),
                "experiment_name": training.get("experiment_name", ""),
                "model_name": training.get("model_name", ""),
                "batch_size": training.get("batch_size", ""),
                "num_workers": training.get("num_workers", ""),
                "cpus_per_task": training.get("cpus_per_task", ""),
                "epochs": training.get("epochs", ""),
                "avg_samples_per_sec": training.get(
                    "avg_samples_per_sec", ""
                ),
                "avg_step_time_sec": training.get("avg_step_time_sec", ""),
                "avg_data_time_sec": training.get("avg_data_time_sec", ""),
                "avg_compute_time_sec": training.get(
                    "avg_compute_time_sec", ""
                ),
                "data_time_ratio": training.get("data_time_ratio", ""),
                "best_val_top1": training.get("best_val_top1", ""),
                "best_val_top5": training.get("best_val_top5", ""),
                "final_test_top1": training.get("final_test_top1", ""),
                "final_test_top5": training.get("final_test_top5", ""),
                "torch_peak_allocated_mb": training.get(
                    "torch_peak_allocated_mb", ""
                ),
                "torch_peak_reserved_mb": training.get(
                    "torch_peak_reserved_mb", ""
                ),
                "nvidia_smi_max_memory_used_mb": gpu.get(
                    "nvidia_smi_max_memory_used_mb", ""
                ),
                "nvidia_smi_avg_memory_used_mb": gpu.get(
                    "nvidia_smi_avg_memory_used_mb", ""
                ),
                "nvidia_smi_avg_gpu_utilization": gpu.get(
                    "nvidia_smi_avg_gpu_utilization", ""
                ),
                "nvidia_smi_max_gpu_utilization": gpu.get(
                    "nvidia_smi_max_gpu_utilization", ""
                ),
                "nvidia_smi_avg_power_draw_w": gpu.get(
                    "nvidia_smi_avg_power_draw_w", ""
                ),
                "success": training.get("success", ""),
                "error": training.get("error", ""),
            }
        )

    output_rows.sort(
        key=lambda row: (row["experiment_name"], row["job_id"])
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    missing_gpu = sum(
        not row["nvidia_smi_max_memory_used_mb"] for row in output_rows
    )
    if missing_gpu:
        print(f"Warning: {missing_gpu} training run(s) have no matching GPU summary")
    print(f"Comparison CSV: {output_path} ({len(output_rows)} runs)")


if __name__ == "__main__":
    main()
