# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""Build leaderboard-style submission payloads."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from trace_logger import TraceLogger


def build_workflow_metadata(trace: TraceLogger, config: dict[str, Any]) -> dict[str, Any]:
    summary = trace.summary()
    llm_events = [e for e in trace.events if e.get("event_type") == "llm_call"]
    return {
        "workflow_mode": "parallel",
        "num_agents": len(summary["agents_used"]),
        "max_parallel_agents": int(config["max_parallel_agents"]),
        "models_used": summary["models_used"],
        "model_roles": {
            e["agent_name"]: e["model"]
            for e in llm_events
            if e.get("agent_name") != "preload"
        },
        "has_aggregator": any(e.get("agent_name") == "aggregator" for e in llm_events),
        "has_verifier": any(str(e.get("agent_name", "")).startswith("verifier") for e in llm_events),
        "llm_calls": summary["llm_calls"],
        "runtime_sec": summary["total_llm_runtime_sec"],
        "estimated_input_tokens": summary["estimated_input_tokens"],
        "estimated_output_tokens": summary["estimated_output_tokens"],
    }


def build_submission(
    *,
    team_id: str,
    team_name: str,
    config: dict[str, Any],
    trace: TraceLogger,
    final_answer: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "hackathon.resource_qa.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "team": {
            "team_id": team_id,
            "team_name": team_name,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        },
        "workflow_metadata": build_workflow_metadata(trace, config),
        "answer": final_answer,
        "trace_summary": trace.summary(),
    }


def write_submission(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
