# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Reference workload or workflow infrastructure; teams should not modify this file.

"""Small Ollama client that supports per-agent model routing."""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

from trace_logger import TraceLogger


class OllamaClient:
    def __init__(
        self,
        host: str,
        trace: TraceLogger,
        *,
        temperature: float = 0.1,
        num_ctx: int = 8192,
        keep_alive: str = "30m",
        timeout: int = 600,
        mock: bool = False,
    ):
        self.host = host.rstrip("/")
        self.trace = trace
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.mock = mock

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        agent_name: str,
        role: str,
    ) -> str:
        start = time.time()
        status = "ok"
        error = ""
        text = ""
        thinking = ""
        done_reason = ""
        try:
            if self.mock:
                text = mock_response(agent_name)
            else:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": self.keep_alive,
                    "options": {
                        "temperature": self.temperature,
                        "num_ctx": self.num_ctx,
                    },
                }
                data = self._post_json("/api/generate", payload)
                text = str(data.get("response", "")).strip()
                thinking = str(data.get("thinking", "")).strip()
                done_reason = str(data.get("done_reason", ""))
        except Exception as exc:
            status = "failed"
            error = str(exc)
            text = json.dumps({"error": error})
        finally:
            end = time.time()
            self.trace.record_llm_call(
                agent_name=agent_name,
                role=role,
                model=model,
                start_time=start,
                end_time=end,
                input_chars=len(prompt),
                output_chars=len(text) + len(thinking),
                response_chars=len(text),
                thinking_chars=len(thinking),
                status=status,
                error=error,
                done_reason=done_reason,
            )
        return text


def mock_response(agent_name: str) -> str:
    responses = {
        "file_inventory": "Important files include evidence/repo/slurm/train-one.slurm, evidence/repo/src/train.py, evidence/repo/src/data/datasets.py, evidence/repo/src/models/resnet.py, evidence/repo/src/utils/config.py, evidence/results/comparison.csv, and gpu/training summaries.",
        "code_agent": "evidence/repo/src/train.py requires CUDA, creates dataloaders, builds a custom ResNet, trains with SGD and CosineAnnealingLR, writes step and epoch CSV metrics. evidence/repo/src/engine/trainer.py writes per-step CUDA memory and timing metrics.",
        "config_agent": "Configs define model_name, batch_size, num_workers, cpus_per_task, epochs, optimizer hyperparameters, dataset_dir, and checkpoint options. Allowed values are validated in evidence/repo/src/utils/config.py.",
        "slurm_agent": "evidence/repo/slurm/train-one.slurm requests partition gpu-queue, qos gpu-q, cpus-per-task=4, gres=gpu:1, time=01:00:00, starts nvidia-smi logging, and runs apptainer exec --nv.",
        "metrics_agent": "evidence/results/comparison.csv compares successful ResNet runs. GPU summaries and step metrics provide memory, utilization, throughput, and data/compute timing evidence.",
        "qa_agent": "Questions require evidence from Slurm scripts, data preparation/loading code, model registry, training loop, metrics writers, and benchmark result CSV files.",
        "aggregator": "Evidence pack: GPU allocation is in evidence/repo/slurm/train-one.slurm; CUDA enters Apptainer via --nv; datasets use ImageFolder; config validation is in evidence/repo/src/utils/config.py; metrics are written by trainer/metrics utilities and summarized in evidence CSV files.",
        "reasoner": json.dumps({"answers": []}),
        "verifier": json.dumps({"answers": []}),
        "preload": "",
    }
    return responses.get(agent_name, "No mock response for this agent.")
