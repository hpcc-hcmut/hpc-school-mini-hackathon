# @author HCMUT HPC Summer School Hackathon
# @permission TEAM_ADVANCED_EDITABLE
# @note Teams may edit this file only if they intentionally change agent structure.

"""Agent specifications and input collection for the hackathon workflow."""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HACKATHON_INFRASTRUCTURE_FILES = {
    "HACKATHON.md",
    "FILE_OWNERSHIP.md",
    "repo.zip",
    "configs/hackathon_workflow.json",
    "slurm/hackathon-parallel-workflow.slurm",
    "src/agents.py",
    "src/hackathon_workflow.py",
    "src/ollama_client.py",
    "src/parallel_runner.py",
    "src/submission_builder.py",
    "src/submit.py",
    "src/trace_logger.py",
}

HACKATHON_INFRASTRUCTURE_PREFIXES = (
    ".agents/",
    ".codex/",
    ".git/",
    "__pycache__/",
    "internals/",
    "logs/",
    "prompts/",
)

RUNTIME_ARTIFACT_PREFIXES = (
    "checkpoints/best_",
    "checkpoints/latest_",
    "results/hackathon_submission_",
    "results/hackathon_trace_",
)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str
    model: str
    prompt_template: str
    input_globs: list[str]


def build_agent_specs(config: dict[str, Any]) -> list[AgentSpec]:
    model_plan = config["model_plan"]
    return [
        AgentSpec(
            "file_inventory",
            "file_inventory",
            model_plan["file_inventory"],
            "prompts/file_inventory.txt",
            ["**/*"],
        ),
        AgentSpec(
            "code_agent",
            "source_analysis",
            model_plan["code_agent"],
            "prompts/code_agent.txt",
            ["evidence/repo/src/**/*.py"],
        ),
        AgentSpec(
            "config_agent",
            "config_extraction",
            model_plan["config_agent"],
            "prompts/config_agent.txt",
            ["evidence/repo/configs/*.json", "evidence/repo/src/utils/config.py"],
        ),
        AgentSpec(
            "slurm_agent",
            "slurm_extraction",
            model_plan["slurm_agent"],
            "prompts/slurm_agent.txt",
            ["evidence/repo/slurm/*.slurm"],
        ),
        AgentSpec(
            "metrics_agent",
            "metrics_extraction",
            model_plan["metrics_agent"],
            "prompts/metrics_agent.txt",
            [
                "evidence/results/comparison.csv",
                "evidence/results/training_summary_*.csv",
                "evidence/results/gpu_summary_*.csv",
                "evidence/results/epoch_metrics_*.csv",
            ],
        ),
        AgentSpec(
            "qa_agent",
            "question_planning",
            model_plan["qa_agent"],
            "prompts/qa_agent.txt",
            ["qa_public.json", "qa_private.json"],
        ),
    ]


def read_text_with_line_numbers(path: Path, max_chars: int) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    numbered = "\n".join(f"{idx:04d}: {line}" for idx, line in enumerate(lines, 1))
    return numbered[:max_chars]


def path_matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def is_evidence_path(path: str) -> bool:
    """Return true for repository evidence files, not hackathon wrapper files."""
    if path in HACKATHON_INFRASTRUCTURE_FILES:
        return False
    if path in {"qa_public.json", "qa_private.json"}:
        return False
    if path.endswith(".pyc") or "/__pycache__/" in path:
        return False
    if path.startswith(HACKATHON_INFRASTRUCTURE_PREFIXES):
        return False
    if path.startswith(RUNTIME_ARTIFACT_PREFIXES):
        return False
    return True


def collect_agent_input(
    repo_root: Path,
    input_globs: list[str],
    *,
    max_chars_per_file: int,
    max_prompt_chars: int,
) -> str:
    chunks: list[str] = []
    used = 0
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root).as_posix()
        if not path_matches(rel, input_globs):
            continue
        if not is_evidence_path(rel):
            continue
        if path.suffix.lower() not in {".py", ".json", ".slurm", ".csv", ".md", ".txt"}:
            continue
        body = read_text_with_line_numbers(path, max_chars_per_file)
        block = f"\n\n### FILE: {rel}\n{body}"
        if used + len(block) > max_prompt_chars:
            break
        chunks.append(block)
        used += len(block)
    return "\n".join(chunks) if chunks else "No matching files."


def load_prompt(template_path: Path, input_text: str) -> str:
    template = template_path.read_text(encoding="utf-8")
    return template.replace("{{INPUT}}", input_text)


def load_questions(repo_root: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for name in ("qa_public.json", "qa_private.json"):
        path = repo_root / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            questions.extend(data)
    return questions
