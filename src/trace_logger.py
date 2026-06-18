# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""Trace logging for hackathon agent workflows."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def estimate_tokens(text_or_chars: str | int) -> int:
    chars = text_or_chars if isinstance(text_or_chars, int) else len(text_or_chars)
    return max(1, int(chars) // 4)


class TraceLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.events: list[dict[str, Any]] = []

    def record_llm_call(
        self,
        *,
        agent_name: str,
        role: str,
        model: str,
        start_time: float,
        end_time: float,
        input_chars: int,
        output_chars: int,
        status: str,
        error: str = "",
        response_chars: int | None = None,
        thinking_chars: int = 0,
        done_reason: str = "",
    ) -> None:
        self.events.append(
            {
                "event_type": "llm_call",
                "agent_name": agent_name,
                "role": role,
                "model": model,
                "start_time": start_time,
                "end_time": end_time,
                "duration_sec": round(end_time - start_time, 4),
                "input_chars": input_chars,
                "output_chars": output_chars,
                "response_chars": output_chars if response_chars is None else response_chars,
                "thinking_chars": thinking_chars,
                "estimated_input_tokens": estimate_tokens(input_chars),
                "estimated_output_tokens": estimate_tokens(output_chars),
                "status": status,
                "error": error,
                "done_reason": done_reason,
            }
        )

    def record_step(
        self,
        *,
        step_name: str,
        start_time: float,
        end_time: float,
        children: list[str] | None = None,
        status: str = "ok",
    ) -> None:
        self.events.append(
            {
                "event_type": "workflow_step",
                "step_name": step_name,
                "start_time": start_time,
                "end_time": end_time,
                "duration_sec": round(end_time - start_time, 4),
                "children": children or [],
                "status": status,
            }
        )

    def save_jsonl(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for event in self.events:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def summary(self) -> dict[str, Any]:
        llm_events = [e for e in self.events if e.get("event_type") == "llm_call"]
        workflow_events = [
            e for e in self.events if e.get("event_type") == "workflow_step"
        ]
        return {
            "llm_calls": len([e for e in llm_events if e.get("agent_name") != "preload"]),
            "models_used": sorted(
                {
                    e["model"]
                    for e in llm_events
                    if e.get("agent_name") != "preload" and e.get("model")
                }
            ),
            "agents_used": sorted(
                {
                    e["agent_name"]
                    for e in llm_events
                    if e.get("agent_name") != "preload"
                }
            ),
            "estimated_input_tokens": sum(
                int(e.get("estimated_input_tokens", 0)) for e in llm_events
            ),
            "estimated_output_tokens": sum(
                int(e.get("estimated_output_tokens", 0)) for e in llm_events
            ),
            "total_llm_runtime_sec": round(
                sum(float(e.get("duration_sec", 0.0)) for e in llm_events), 4
            ),
            "workflow_steps": workflow_events,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
