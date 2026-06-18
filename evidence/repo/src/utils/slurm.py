# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""Helpers for reconciling config values with the allocated Slurm resources."""

import warnings

from utils.config import ALLOWED_CPUS_PER_TASK


def resolve_cpus_per_task(cli_value: str | int | None, configured_value: int) -> int:
    if cli_value is None or str(cli_value).strip().lower() in {"", "unknown", "none"}:
        return configured_value

    try:
        allocated_value = int(cli_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid --cpus-per-task value: {cli_value!r}") from exc

    if allocated_value not in ALLOWED_CPUS_PER_TASK:
        raise ValueError(
            "Allocated cpus-per-task must be one of "
            f"{list(ALLOWED_CPUS_PER_TASK)}, got {allocated_value}"
        )
    if allocated_value != configured_value:
        warnings.warn(
            f"Config requests cpus_per_task={configured_value}, but Slurm allocated "
            f"{allocated_value}; metrics will record the actual allocation",
            UserWarning,
            stacklevel=2,
        )
    return allocated_value
