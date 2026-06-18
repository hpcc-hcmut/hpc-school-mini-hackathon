# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""Summarize a one-second nvidia-smi timeline into one CSV row."""

import argparse
import csv
from pathlib import Path


GPU_SUMMARY_FIELDS = [
    "job_id",
    "experiment_name",
    "model_name",
    "gpu_name",
    "samples",
    "gpu_memory_total_mb",
    "nvidia_smi_max_memory_used_mb",
    "nvidia_smi_avg_memory_used_mb",
    "nvidia_smi_avg_gpu_utilization",
    "nvidia_smi_max_gpu_utilization",
    "nvidia_smi_avg_memory_utilization",
    "nvidia_smi_max_memory_utilization",
    "nvidia_smi_avg_power_draw_w",
    "nvidia_smi_max_power_draw_w",
    "nvidia_smi_avg_temperature_c",
    "nvidia_smi_max_temperature_c",
]


def parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {
        "[not supported]",
        "not supported",
        "n/a",
        "[n/a]",
    }:
        return None
    for suffix in (" MiB", " MB", " W", " C", " %"):
        cleaned = cleaned.replace(suffix, "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def average(values: list[float | None]) -> float:
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else 0.0


def maximum(values: list[float | None]) -> float:
    available = [value for value in values if value is not None]
    return max(available) if available else 0.0


def get_value(row: dict[str, str], key: str) -> str | None:
    for raw_key, value in row.items():
        normalized_key = raw_key.strip()
        if normalized_key == key or normalized_key.startswith(f"{key} "):
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"GPU timeline not found: {input_path}")

    with input_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    gpu_names: list[str] = []
    memory_total: list[float | None] = []
    memory_used: list[float | None] = []
    gpu_utilization: list[float | None] = []
    memory_utilization: list[float | None] = []
    power_draw: list[float | None] = []
    temperatures: list[float | None] = []

    for row in rows:
        name = get_value(row, "name")
        if name:
            gpu_names.append(name.strip())
        memory_total.append(parse_number(get_value(row, "memory.total")))
        memory_used.append(parse_number(get_value(row, "memory.used")))
        gpu_utilization.append(parse_number(get_value(row, "utilization.gpu")))
        memory_utilization.append(
            parse_number(get_value(row, "utilization.memory"))
        )
        power_draw.append(parse_number(get_value(row, "power.draw")))
        temperatures.append(parse_number(get_value(row, "temperature.gpu")))

    summary = {
        "job_id": args.job_id,
        "experiment_name": args.experiment_name,
        "model_name": args.model_name,
        "gpu_name": gpu_names[0] if gpu_names else "unknown",
        "samples": len(rows),
        "gpu_memory_total_mb": round(maximum(memory_total), 4),
        "nvidia_smi_max_memory_used_mb": round(maximum(memory_used), 4),
        "nvidia_smi_avg_memory_used_mb": round(average(memory_used), 4),
        "nvidia_smi_avg_gpu_utilization": round(average(gpu_utilization), 4),
        "nvidia_smi_max_gpu_utilization": round(maximum(gpu_utilization), 4),
        "nvidia_smi_avg_memory_utilization": round(
            average(memory_utilization), 4
        ),
        "nvidia_smi_max_memory_utilization": round(
            maximum(memory_utilization), 4
        ),
        "nvidia_smi_avg_power_draw_w": round(average(power_draw), 4),
        "nvidia_smi_max_power_draw_w": round(maximum(power_draw), 4),
        "nvidia_smi_avg_temperature_c": round(average(temperatures), 4),
        "nvidia_smi_max_temperature_c": round(maximum(temperatures), 4),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GPU_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(summary)

    if not rows:
        print(f"Warning: no GPU samples found in {input_path}")
    print(f"GPU summary CSV: {output_path}")


if __name__ == "__main__":
    main()
