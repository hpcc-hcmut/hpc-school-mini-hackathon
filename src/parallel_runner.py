# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""Run independent LLM agents concurrently."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from agents import AgentSpec
from trace_logger import TraceLogger


def run_parallel_agents(
    agent_specs: list[AgentSpec],
    run_agent_fn: Callable[[AgentSpec], dict],
    *,
    max_workers: int,
    trace: TraceLogger,
) -> dict[str, dict]:
    start = time.time()
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_agent_fn, spec): spec for spec in agent_specs}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                results[spec.name] = future.result()
            except Exception as exc:
                results[spec.name] = {
                    "agent": spec.name,
                    "role": spec.role,
                    "model": spec.model,
                    "status": "failed",
                    "output": str(exc),
                }
    end = time.time()
    trace.record_step(
        step_name="parallel_evidence_extraction",
        start_time=start,
        end_time=end,
        children=[spec.name for spec in agent_specs],
        status="ok",
    )
    return results
