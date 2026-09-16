# @author HCMUT HPC Summer School Hackathon
# @permission TEAM_EDITABLE
# @note Teams may edit this file to improve workflow behavior.

"""Parallel multi-agent hackathon workflow for repository Q&A."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from agents import (
    AgentSpec,
    build_agent_specs,
    collect_agent_input,
    is_evidence_path,
    load_prompt,
    load_questions,
)
from ollama_client import OllamaClient
from parallel_runner import run_parallel_agents
from submission_builder import build_submission, write_submission
from trace_logger import TraceLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a parallel, traceable, multi-model hackathon workflow."
    )
    parser.add_argument("--source-dir", default=".")
    parser.add_argument("--config", default="configs/hackathon_workflow.json")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--team-id", default=os.environ.get("TEAM_ID", "local-team"))
    parser.add_argument("--team-name", default=os.environ.get("TEAM_NAME", "Local Team"))
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--submit-url", default=os.environ.get("BACKEND_URL", ""))
    parser.add_argument("--team-token", default=os.environ.get("TEAM_TOKEN", ""))
    return parser.parse_args()


def safe_question_view(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe = []
    for item in questions:
        safe.append(
            {
                "question_id": item.get("question_id") or item.get("id"),
                "difficulty": item.get("difficulty", ""),
                "question": item.get("question", ""),
            }
        )
    return safe


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


FALLBACK_ANSWER = "No supported answer was produced for this question."

PATH_PREFIX_REWRITES = (
    ("src/", "evidence/repo/src/"),
    ("slurm/", "evidence/repo/slurm/"),
    ("configs/", "evidence/repo/configs/"),
    ("results/", "evidence/results/"),
)

PATH_TEXT_REWRITES = {
    "`src/": "`evidence/repo/src/",
    "`slurm/": "`evidence/repo/slurm/",
    "`configs/": "`evidence/repo/configs/",
    "`results/": "`evidence/results/",
}


def evidence(file: str, start: int, end: int | None = None, reason: str = "") -> dict[str, Any]:
    return {
        "file": file,
        "lines": [start, end or start],
        "reason": reason or f"Baseline evidence from {file}.",
    }


def first_matching_line(repo_root: Path, rel_path: str, pattern: str) -> int:
    path = repo_root / rel_path
    if not path.exists():
        return 1
    regex = re.compile(pattern, re.IGNORECASE)
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if regex.search(line):
            return idx
    return 1


def rewrite_evidence_path(repo_root: Path, rel_path: str) -> str:
    normalized = str(rel_path).strip().lstrip("./")
    if (repo_root / normalized).exists() and is_evidence_path(normalized):
        return normalized
    for old, new in PATH_PREFIX_REWRITES:
        if normalized.startswith(old):
            candidate = f"{new}{normalized[len(old):]}"
            if (repo_root / candidate).exists() and is_evidence_path(candidate):
                return candidate
    return normalized


def rewrite_answer_text(text: str) -> str:
    rewritten = str(text)
    for old, new in PATH_TEXT_REWRITES.items():
        rewritten = rewritten.replace(old, new)
    return rewritten


def normalize_evidence_items(
    repo_root: Path,
    items: Any,
    baseline_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            rel_path = rewrite_evidence_path(repo_root, str(item.get("file", "")))
            if not rel_path or not (repo_root / rel_path).exists() or not is_evidence_path(rel_path):
                continue
            lines = item.get("lines", [1, 1])
            if not isinstance(lines, list) or not lines:
                lines = [1, 1]
            start = int(lines[0]) if str(lines[0]).isdigit() else 1
            end = int(lines[1]) if len(lines) > 1 and str(lines[1]).isdigit() else start
            cleaned.append(
                {
                    "file": rel_path,
                    "lines": [max(1, start), max(1, end)],
                    "reason": str(item.get("reason", f"Evidence from {rel_path}.")),
                }
            )
    return cleaned or baseline_items


def load_comparison_rows(repo_root: Path) -> list[dict[str, str]]:
    path = repo_root / "evidence" / "results" / "comparison.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_answer(
    repo_root: Path,
    rows: list[dict[str, str]],
    question: str,
) -> tuple[str, list[dict[str, Any]], str] | None:
    lowered = question.lower()
    metric_map = [
        ("throughput", "highest", "avg_samples_per_sec", max),
        ("samples per second", "slowest", "avg_samples_per_sec", min),
        ("peak allocated", "lowest", "torch_peak_allocated_mb", min),
        ("test top-1", "best", "final_test_top1", max),
        ("average gpu utilization", "highest", "nvidia_smi_avg_gpu_utilization", max),
        ("maximum gpu memory", "highest", "nvidia_smi_max_memory_used_mb", max),
    ]
    for needle, direction, column, chooser in metric_map:
        if needle not in lowered:
            continue
        valid = [row for row in rows if row.get(column)]
        if not valid:
            return None
        selected = chooser(valid, key=lambda row: float(row[column]))
        experiment = selected.get("experiment_name", "unknown")
        model = selected.get("model_name", "unknown")
        value = selected.get(column, "unknown")
        line = first_matching_line(repo_root, "evidence/results/comparison.csv", selected.get("job_id", ""))
        return (
            f"Baseline metric answer: {experiment} ({model}) is the {direction} run for {column} with value {value}.",
            [evidence("evidence/results/comparison.csv", line, reason=f"Row for job {selected.get('job_id')} contains {column}={value}.")],
            "medium",
        )
    if "all six" in lowered and "success" in lowered and rows:
        all_success = all(str(row.get("success", "")).lower() == "true" for row in rows)
        return (
            f"Baseline metric answer: {'yes' if all_success else 'no'}, the comparison table marks {sum(str(row.get('success', '')).lower() == 'true' for row in rows)} of {len(rows)} runs as successful.",
            [evidence("evidence/results/comparison.csv", 1, reason="The comparison CSV includes a success column for each compared run.")],
            "medium",
        )
    return None


def route_baseline_answer(repo_root: Path, rows: list[dict[str, str]], question: dict[str, Any]) -> dict[str, Any]:
    qid = str(question.get("question_id"))
    text = str(question.get("question", ""))
    lowered = text.lower()

    metric = metric_answer(repo_root, rows, lowered)
    if metric:
        answer, proof, confidence = metric
        return {"question_id": qid, "answer": answer, "evidence": proof, "confidence": confidence}

    routes = [
        (("gpu allocation", "gres", "partition", "qos"), "evidence/repo/slurm/train-one.slurm", r"#SBATCH.*(--gres|--partition|--qos)", "GPU allocation and Slurm resource directives are defined in the training Slurm script."),
        (("apptainer", "cuda", "--nv"), "evidence/repo/slurm/train-one.slurm", r"apptainer|--nv", "The Slurm wrapper shows how CUDA/GPU access is passed into Apptainer."),
        (("gpu timeline", "nvidia-smi", "summarizes"), "evidence/repo/slurm/train-one.slurm", r"nvidia-smi|summarize_gpu_log", "The training wrapper records and summarizes GPU telemetry."),
        (("resnet", "registered", "variants", "pretrained", "weights"), "evidence/repo/src/models/resnet.py", r"resnet|pretrained|weights", "Custom ResNet variants and argument checks live in the model implementation."),
        (("split", "cifar", "imagefolder", "preparation"), "evidence/repo/src/data/prepare_cifar100_imagefolder.py", r"split|ImageFolder|CIFAR", "Dataset preparation and split behavior live in the CIFAR-100 ImageFolder preparation script."),
        (("dataset loader", "dataloader", "validation folder", "test folder"), "evidence/repo/src/data/datasets.py", r"ImageFolder|DataLoader|train|val|test", "Train, validation, and test dataloaders are constructed in the dataset module."),
        (("allowed model", "batch size", "config", "validation"), "evidence/repo/src/utils/config.py", r"allowed|required|batch_size|model_name", "Configuration validation defines accepted models and resource-related constraints."),
        (("cuda is unavailable", "optimizer", "scheduler", "training code"), "evidence/repo/src/train.py", r"CUDA is unavailable|optim|scheduler", "The training entrypoint checks CUDA availability and constructs optimizer/scheduler objects."),
        (("per-step", "step metrics", "step csv"), "evidence/repo/src/engine/trainer.py", r"step_metrics|append_csv_row|global_step", "Per-step metrics are emitted from the training loop."),
        (("benchmark array", "array task", "array choose"), "evidence/repo/slurm/run-benchmark-array.slurm", r"SLURM_ARRAY_TASK_ID|CONFIGS|array", "The benchmark array script maps Slurm array tasks to config files."),
        (("checkpoint", "save", "load"), "evidence/repo/src/engine/checkpointing.py", r"checkpoint|torch.save|torch.load", "Checkpoint save/load behavior is implemented in the checkpointing module."),
    ]
    for needles, rel_path, line_pattern, reason in routes:
        if any(needle in lowered for needle in needles):
            line = first_matching_line(repo_root, rel_path, line_pattern)
            return {
                "question_id": qid,
                "answer": f"Baseline heuristic answer: inspect `{rel_path}`. {reason}",
                "evidence": [evidence(rel_path, line, reason=reason)],
                "confidence": "low",
            }

    return {
        "question_id": qid,
        "answer": "Baseline heuristic answer: this repository question likely needs evidence from the source, Slurm scripts, configs, or benchmark CSV files. Improve the agents or prompts to produce a more specific supported answer.",
        "evidence": [
            evidence("evidence/repo/src/train.py", 1, reason="Training code is part of the repository evidence corpus."),
            evidence("evidence/repo/slurm/train-one.slurm", 1, reason="Slurm scripts are part of the repository evidence corpus."),
        ],
        "confidence": "low",
    }


def build_baseline_answers(repo_root: Path, questions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = load_comparison_rows(repo_root)
    return {"answers": [route_baseline_answer(repo_root, rows, question) for question in questions]}


def answer_is_fallback(answer: dict[str, Any]) -> bool:
    return str(answer.get("answer", "")).strip() == FALLBACK_ANSWER


def normalize_answers(
    candidate: dict[str, Any],
    questions: list[dict[str, Any]],
    repo_root: Path,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline_by_id = {
        str(item.get("question_id")): item
        for item in (baseline or {}).get("answers", [])
        if isinstance(item, dict)
    }
    by_id = {
        str(item.get("question_id")): item
        for item in candidate.get("answers", [])
        if isinstance(item, dict)
    }
    normalized = []
    for question in questions:
        qid = str(question.get("question_id"))
        answer = by_id.get(qid)
        if not answer or answer_is_fallback(answer):
            answer = baseline_by_id.get(qid, {
                "question_id": qid,
                "answer": FALLBACK_ANSWER,
                "evidence": [],
                "confidence": "low",
            })
        answer = dict(answer)
        answer.setdefault("question_id", qid)
        answer["difficulty"] = str(question.get("difficulty", ""))
        answer["question"] = str(question.get("question", ""))
        answer["answer"] = rewrite_answer_text(str(answer.get("answer", FALLBACK_ANSWER)))
        baseline_items = baseline_by_id.get(qid, {}).get("evidence", [])
        answer["evidence"] = normalize_evidence_items(
            repo_root,
            answer.get("evidence", []),
            baseline_items if isinstance(baseline_items, list) else [],
        )
        answer.setdefault("confidence", "medium")
        normalized.append(answer)
    return {"answers": normalized}


def run_agent(
    spec: AgentSpec,
    *,
    repo_root: Path,
    config: dict[str, Any],
    client: OllamaClient,
) -> dict[str, Any]:
    if spec.name == "qa_agent":
        input_text = json.dumps(
            safe_question_view(load_questions(repo_root)),
            ensure_ascii=False,
            indent=2,
        )
    elif spec.name == "file_inventory":
        paths = []
        for path in sorted(repo_root.rglob("*")):
            if path.is_file():
                rel = path.relative_to(repo_root).as_posix()
                if rel.startswith(("logs/", ".git/", "__pycache__/")):
                    continue
                if rel in {"qa_public.json", "qa_private.json", "repo.zip"}:
                    continue
                if not is_evidence_path(rel):
                    continue
                paths.append(rel)
        input_text = "\n".join(paths)
    else:
        input_text = collect_agent_input(
            repo_root,
            spec.input_globs,
            max_chars_per_file=int(config["max_chars_per_file"]),
            max_prompt_chars=int(config["max_prompt_chars"]),
        )
    prompt = load_prompt(repo_root / spec.prompt_template, input_text)
    output = client.generate(
        model=spec.model,
        prompt=prompt,
        agent_name=spec.name,
        role=spec.role,
    )
    return {
        "agent": spec.name,
        "role": spec.role,
        "model": spec.model,
        "status": "ok",
        "output": output,
    }


def run_single_prompt(
    *,
    repo_root: Path,
    template_name: str,
    input_obj: Any,
    agent_name: str,
    role: str,
    model: str,
    client: OllamaClient,
) -> str:
    input_text = (
        input_obj if isinstance(input_obj, str) else json.dumps(input_obj, ensure_ascii=False, indent=2)
    )
    prompt = load_prompt(repo_root / "prompts" / template_name, input_text)
    return client.generate(
        model=model,
        prompt=prompt,
        agent_name=agent_name,
        role=role,
    )


def maybe_preload_models(config: dict[str, Any], client: OllamaClient) -> None:
    if not config.get("preload_models"):
        return
    for model in sorted(set(config["model_plan"].values())):
        client.generate(
            model=model,
            prompt="",
            agent_name="preload",
            role="model_preload",
        )


def split_questions(questions: list[dict[str, Any]], num_batches: int) -> list[list[dict[str, Any]]]:
    batches = [[] for _ in range(max(1, num_batches))]
    for idx, question in enumerate(questions):
        batches[idx % len(batches)].append(question)
    return [batch for batch in batches if batch]


def run_reasoner_batch(
    *,
    batch_id: int,
    repo_root: Path,
    questions: list[dict[str, Any]],
    evidence_pack: str,
    model: str,
    client: OllamaClient,
) -> dict[str, Any]:
    output = run_single_prompt(
        repo_root=repo_root,
        template_name="reasoner.txt",
        input_obj={"questions": questions, "evidence_pack": evidence_pack},
        agent_name=f"reasoner_{batch_id}",
        role="answer_reasoning",
        model=model,
        client=client,
    )
    return {"batch_id": batch_id, "questions": questions, "output": output}


def run_verifier_batch(
    *,
    batch_id: int,
    repo_root: Path,
    questions: list[dict[str, Any]],
    evidence_pack: str,
    proposed_answers: str,
    model: str,
    client: OllamaClient,
) -> dict[str, Any]:
    output = run_single_prompt(
        repo_root=repo_root,
        template_name="verifier.txt",
        input_obj={
            "questions": questions,
            "evidence_pack": evidence_pack,
            "proposed_answers": proposed_answers,
        },
        agent_name=f"verifier_{batch_id}",
        role="verification",
        model=model,
        client=client,
    )
    return {
        "batch_id": batch_id,
        "questions": questions,
        "proposed_answers": proposed_answers,
        "output": output,
    }


def run_parallel_reasoners(
    *,
    repo_root: Path,
    question_batches: list[list[dict[str, Any]]],
    evidence_pack: str,
    model: str,
    client: OllamaClient,
    trace: TraceLogger,
) -> list[dict[str, Any]]:
    start = time.time()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(question_batches)) as pool:
        futures = [
            pool.submit(
                run_reasoner_batch,
                batch_id=idx,
                repo_root=repo_root,
                questions=batch,
                evidence_pack=evidence_pack,
                model=model,
                client=client,
            )
            for idx, batch in enumerate(question_batches, 1)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    trace.record_step(
        step_name="parallel_answer_reasoning",
        start_time=start,
        end_time=time.time(),
        children=[f"reasoner_{idx}" for idx in range(1, len(question_batches) + 1)],
        status="ok",
    )
    return sorted(results, key=lambda item: int(item["batch_id"]))


def run_parallel_verifiers(
    *,
    repo_root: Path,
    reasoner_results: list[dict[str, Any]],
    evidence_pack: str,
    model: str,
    client: OllamaClient,
    trace: TraceLogger,
) -> list[dict[str, Any]]:
    start = time.time()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(reasoner_results)) as pool:
        futures = [
            pool.submit(
                run_verifier_batch,
                batch_id=int(item["batch_id"]),
                repo_root=repo_root,
                questions=item["questions"],
                evidence_pack=evidence_pack,
                proposed_answers=item["output"],
                model=model,
                client=client,
            )
            for item in reasoner_results
        ]
        for future in as_completed(futures):
            results.append(future.result())
    trace.record_step(
        step_name="parallel_answer_verification",
        start_time=start,
        end_time=time.time(),
        children=[f"verifier_{item['batch_id']}" for item in reasoner_results],
        status="ok",
    )
    return sorted(results, key=lambda item: int(item["batch_id"]))


def normalize_batch_result(
    *,
    repo_root: Path,
    verified: str,
    reasoned: str,
    questions: list[dict[str, Any]],
    baseline_answer: dict[str, Any],
) -> dict[str, Any]:
    try:
        return normalize_answers(extract_json_object(verified), questions, repo_root, baseline_answer)
    except Exception:
        try:
            return normalize_answers(extract_json_object(reasoned), questions, repo_root, baseline_answer)
        except Exception:
            return normalize_answers({"answers": []}, questions, repo_root, baseline_answer)


def merge_answer_batches(batch_answers: list[dict[str, Any]]) -> dict[str, Any]:
    answers: list[dict[str, Any]] = []
    for batch in batch_answers:
        answers.extend(batch.get("answers", []))
    return {"answers": answers}


def main() -> int:
    args = parse_args()
    repo_root = Path(args.source_dir).resolve()
    config = json.loads((repo_root / args.config).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    trace_path = output_dir / f"hackathon_trace_{job_id}.jsonl"
    submission_path = output_dir / f"hackathon_submission_{job_id}.json"

    trace = TraceLogger(trace_path)
    client = OllamaClient(
        args.ollama_host,
        trace,
        temperature=float(config["temperature"]),
        num_ctx=int(config["num_ctx"]),
        keep_alive=str(config["keep_alive"]),
        mock=args.mock,
    )
    maybe_preload_models(config, client)

    questions = safe_question_view(load_questions(repo_root))
    if not questions:
        raise RuntimeError("No qa_public.json or qa_private.json questions were found.")
    baseline_answer = build_baseline_answers(repo_root, questions)

    specs = build_agent_specs(config)
    partial_results = run_parallel_agents(
        specs,
        lambda spec: run_agent(spec, repo_root=repo_root, config=config, client=client),
        max_workers=int(config["max_parallel_agents"]),
        trace=trace,
    )

    model_plan = config["model_plan"]
    evidence_pack = run_single_prompt(
        repo_root=repo_root,
        template_name="aggregator.txt",
        input_obj=partial_results,
        agent_name="aggregator",
        role="evidence_aggregation",
        model=model_plan["aggregator"],
        client=client,
    )
    question_batches = split_questions(
        questions,
        int(config.get("answer_batches", 2)),
    )
    reasoner_results = run_parallel_reasoners(
        repo_root=repo_root,
        question_batches=question_batches,
        evidence_pack=evidence_pack,
        model=model_plan["reasoner"],
        client=client,
        trace=trace,
    )
    verifier_results = run_parallel_verifiers(
        repo_root=repo_root,
        reasoner_results=reasoner_results,
        evidence_pack=evidence_pack,
        model=model_plan["verifier"],
        client=client,
        trace=trace,
    )
    batch_answers = [
        normalize_batch_result(
            repo_root=repo_root,
            verified=item["output"],
            reasoned=item["proposed_answers"],
            questions=item["questions"],
            baseline_answer=baseline_answer,
        )
        for item in verifier_results
    ]
    final_answer = normalize_answers(
        merge_answer_batches(batch_answers),
        questions,
        repo_root,
        baseline_answer,
    )

    payload = build_submission(
        team_id=args.team_id,
        team_name=args.team_name,
        config=config,
        trace=trace,
        final_answer=final_answer,
    )
    write_submission(submission_path, payload)
    trace.save_jsonl()

    print(f"Trace: {trace_path}")
    print(f"Submission: {submission_path}")

    if args.submit_url:
        if not args.team_token:
            raise RuntimeError("--team-token or TEAM_TOKEN is required when submitting.")
        from submit import submit_to_backend

        response = submit_to_backend(
            path=submission_path,
            backend_url=args.submit_url,
            team_id=args.team_id,
            team_token=args.team_token,
        )
        print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
