# HPC School Mini Hackathon

> **Public Release Note:** This repository is the student-facing package for the EduHPC Peachy assignment paper. 
> The paired instructor-facing leaderboard package is available at: 
> https://github.com/hpcc-hcmut/hpc-summer-school-leaderboard
>
> For the student handout, see [HACKATHON.md](HACKATHON.md).

Starter code for the HCMUT HPC Summer School mini hackathon. Your team will build
a traceable multi-agent LLM workflow that answers questions about an HPC training
repository using evidence from source code, Slurm scripts, configs, and benchmark
CSV files.

## What You Are Building

Your workflow should read the evidence in this repository, run several LLM agents,
and produce a submission JSON with:

- answers for the provided questions
- file and line evidence for each answer
- a trace of agent calls, models, timings, and token estimates
- workflow metadata such as number of agents, model routing, verifier presence,
  and runtime

The default starter already runs a complete baseline. Your job is to improve
answer quality, evidence quality, runtime, and workflow design.

### What Students Receive
- **Evidence corpus:** Frozen training repository (`evidence/repo/`) and benchmark logs (`evidence/results/`).
- **Starter workflow:** Orchestration scripts, Slurm wrapper, and parallel runner (`src/`, `slurm/`).
- **Starter prompts & configs:** Baseline files to get started (`prompts/`, `configs/`).
- **Sample questions:** Public scaffolding questions (`qa_public.json`).

### What Students Should Modify
- Prompts, workflow configs, agent roles, and verification logic.
- Most teams focus on `configs/hackathon_workflow.json` and `prompts/`. Advanced teams may edit `src/agents.py`.

### What Students Must Not Modify
- Frozen evidence, scoring inputs, generated output structures, and core infrastructure utilities (unless directed).

## Repository Layout

```text
.
├── HACKATHON.md                     # Short hackathon instructions
├── FILE_OWNERSHIP.md                # Which files teams may edit
├── configs/
│   └── hackathon_workflow.json      # Model routing and workflow knobs
├── prompts/
│   ├── aggregator.txt
│   ├── code_agent.txt
│   ├── config_agent.txt
│   ├── file_inventory.txt
│   ├── metrics_agent.txt
│   ├── qa_agent.txt
│   ├── reasoner.txt
│   ├── slurm_agent.txt
│   └── verifier.txt
├── src/
│   ├── agents.py
│   ├── hackathon_workflow.py
│   ├── ollama_client.py
│   ├── parallel_runner.py
│   ├── submission_builder.py
│   ├── submit.py
│   └── trace_logger.py
├── slurm/
│   └── hackathon-parallel-workflow.slurm
├── evidence/
│   ├── repo/                        # Frozen reference training repository
│   └── results/                     # Frozen benchmark CSV evidence
├── qa_public.json                   # Public sample questions
├── data/                            # Team local data area
├── checkpoints/                     # Team local checkpoint area
├── logs/                            # Generated at runtime, ignored by Git
└── results/                         # Generated submissions/traces, ignored by Git
```

> **Note on `qa_public.json`:** The public QA file is included for demonstration, testing, and artifact review. Instructors running a live event should provide question-only files to students and keep private questions and ground-truth answers in the instructor-facing leaderboard package.

Important distinction:

- `src/` is the hackathon agent workflow code.
- `evidence/repo/src/` is the frozen training repository that agents should
  inspect as evidence.
- `results/` is for generated hackathon outputs.
- `evidence/results/` is frozen benchmark evidence and should not be modified.

## Files You May Edit

Most teams should focus on:

- `configs/hackathon_workflow.json`
- `prompts/*.txt`
- `src/hackathon_workflow.py`

Advanced teams may also edit:

- `src/agents.py`

Do not edit frozen evidence, scoring inputs, generated outputs, or the Slurm
wrapper unless organizers explicitly ask you to. See `FILE_OWNERSHIP.md` for the
complete policy.

## Data Setup

The public repository does not track image data. If organizers provide:

```text
hackathon/evidence/repo/data/demo_images/
```

copy or keep it at:

```text
evidence/repo/data/demo_images/
```

That directory is intentionally ignored by Git. The starter keeps only:

```text
evidence/repo/data/.gitkeep
```

so the folder exists in a fresh checkout.

## Cluster-Specific Paths

The full path uses site-provided deployment assets; the public repository does
not invent container build recipes that were not available for this release.
Set all four variables before `sbatch`:

- `CLIENT_IMAGE`: Apptainer image with Python 3.10+ for the workflow.
- `SERVER_IMAGE`: Apptainer image containing the Ollama server.
- `OLLAMA_CACHE`: read-only, pre-staged Ollama cache containing at least
  `llama3.2:3b` and `qwen3:4b` for the public baseline.
- `ALLOWED_MODELS_FILE`: site-managed text allow-list for student models.

The server must provide Ollama `/api/tags` and `/api/generate` behavior. Models
should normally be staged before class; compute-node downloads are not part of
the reference deployment.

The original HCMUT deployment requested one node, one task, four Slurm CPUs,
one GPU, and 15 minutes, and used a V100 32 GB. V100 is not intrinsic.
Partition/QoS names, shared paths, images, and cache locations are site-specific;
cluster size and interconnect are irrelevant to this single-node workflow.

## First Run On Slurm

From the repository root:

```bash
sbatch slurm/hackathon-parallel-workflow.slurm
```

The Slurm wrapper will:

1. allocate one GPU
2. choose an available Ollama port from `11434 11435 11436 11437`
3. start the Ollama server inside the provided serving container
4. run `src/hackathon_workflow.py`
5. write trace and submission files into `results/`

Architecture: Slurm job → one GPU node → Apptainer Ollama server → local model
cache → loopback Ollama API → Apptainer Python workflow → JSON trace and
submission → optional leaderboard POST.

Example generated files:

```text
results/hackathon_trace_<job_id>.jsonl
results/hackathon_submission_<job_id>.json
logs/hackathon-agents-<job_id>.out
logs/hackathon-agents-<job_id>.err
logs/ollama-server-<job_id>.log
```

Check job status with:

```bash
squeue -j <job_id>
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed,NodeList
```

## Team Identity And Optional Backend Submission

Set these environment variables before `sbatch`:

```bash
export TEAM_ID=team01
export TEAM_NAME="Team 01"
sbatch slurm/hackathon-parallel-workflow.slurm
```

If organizers provide a backend URL and team token:

```bash
export TEAM_ID=team01
export TEAM_NAME="Team 01"
export BACKEND_URL="https://example-leaderboard"
export TEAM_TOKEN="<your-team-token>"
sbatch slurm/hackathon-parallel-workflow.slurm
```

Do not commit tokens, logs, generated submissions, or local data.

## Configuration

Edit `configs/hackathon_workflow.json` to change model routing and workflow
settings:

```json
{
  "model_plan": {
    "file_inventory": "llama3.2:3b",
    "code_agent": "llama3.2:3b",
    "config_agent": "llama3.2:3b",
    "slurm_agent": "llama3.2:3b",
    "metrics_agent": "llama3.2:3b",
    "qa_agent": "llama3.2:3b",
    "aggregator": "qwen3:4b",
    "reasoner": "qwen3:4b",
    "verifier": "llama3.2:3b"
  },
  "max_parallel_agents": 4,
  "answer_batches": 2,
  "num_ctx": 16384,
  "temperature": 0.1
}
```

Useful knobs:

- `model_plan`: choose which model each agent uses.
- `max_parallel_agents`: limit concurrent evidence extraction agents.
- `answer_batches`: split questions across parallel reasoners.
- `num_ctx`: context length sent to Ollama.
- `temperature`: generation randomness.
- `max_chars_per_file`: per-file truncation limit.
- `max_prompt_chars`: total prompt input limit for file-reading agents.

The checked-in dual-model mapping is a reusable teaching baseline that
demonstrates role-based routing, not an exact snapshot of every classroom run.
Instructors may adapt it to locally available models.

## Default Workflow

The starter workflow has these stages:

1. Parallel evidence extraction:
   - `file_inventory`
   - `code_agent`
   - `config_agent`
   - `slurm_agent`
   - `metrics_agent`
   - `qa_agent`
2. `aggregator` combines extracted evidence.
3. Two `reasoner_*` agents answer question batches in parallel.
4. Two `verifier_*` agents verify the proposed answers.
5. The workflow normalizes answers, evidence paths, and fallback behavior.
6. `submission_builder` writes the final JSON.

The baseline also includes deterministic fallback answers so a run should not
produce an all-empty submission. Improving beyond that fallback is the point of
the hackathon.

## Prompt Improvement Ideas

Good prompts usually:

- demand JSON-only output when the next stage parses JSON
- require exact file paths and line ranges
- tell agents to cite only `evidence/repo/...` and `evidence/results/...`
- ask extraction agents to summarize facts, not speculate
- ask reasoners to answer each `question_id` exactly once
- ask verifiers to reject unsupported claims

Avoid putting huge unrelated context into every prompt. Smaller, focused prompts
are usually faster and easier to verify.

## Thinking Models

Some allowed models may produce separate reasoning or thinking content. The
starter traces both visible response length and thinking length when available.
Thinking is useful for quality, but it costs time and output tokens.

Practical tips:

- Use stronger thinking models for aggregation or reasoning.
- Use smaller models for file extraction and verification when possible.
- Keep `num_ctx` large enough for the selected model and evidence pack.
- Split questions with `answer_batches` instead of asking one model to answer
  everything in one prompt.

## Local Smoke Test

For checking whether Python code paths still run, you may use mock mode:

```bash
python3 src/hackathon_workflow.py \
  --source-dir . \
  --config configs/hackathon_workflow.json \
  --output-dir results \
  --team-id team01 \
  --team-name "Team 01" \
  --mock
```

Mock mode requires Python 3.10+, this public checkout, and a writable output
directory. It does not require Slurm, GPU/CUDA, Apptainer, Ollama, or a model
cache. No evidence-based minimum RAM requirement is claimed, and this release
does not claim verified macOS testing. It checks structure/control flow; it does
not reproduce classroom outcomes or evaluate LLM answer quality.

The client calls Ollama's `/api/generate`. A remote Ollama-compatible service is
supported, but a generic OpenAI-compatible endpoint is not a drop-in
replacement; that backend requires a client adapter.

## Inspecting A Submission

After a Slurm run:

```bash
python3 -m json.tool results/hackathon_submission_<job_id>.json | less
```

Useful fields:

- `answer.answers`: the final answers
- `workflow_metadata.llm_calls`: number of LLM calls
- `workflow_metadata.models_used`: models used
- `workflow_metadata.has_verifier`: whether verifier agents ran
- `workflow_metadata.runtime_sec`: total LLM runtime
- `trace_summary`: compact trace metadata

The JSONL trace contains one event per line:

```bash
head results/hackathon_trace_<job_id>.jsonl
```

## What Not To Commit

The repository already ignores runtime and local files:

- `qa_private.json`
- `evidence/repo/data/demo_images/`
- `logs/`
- `results/hackathon_*.json`
- `checkpoints/*` except `.gitkeep`
- `data/*` except `.gitkeep`
- Python bytecode and cache folders
- tokens or local secrets

Before sharing code:

```bash
git status --short --ignored
```

Only intentional source, prompt, config, markdown, and frozen evidence changes
should be tracked.

## Quick Checklist

Before submitting a run:

1. `sbatch slurm/hackathon-parallel-workflow.slurm`
2. Confirm Slurm completed with exit code `0:0`.
3. Confirm `results/hackathon_submission_<job_id>.json` exists.
4. Confirm the submission has answers for all public questions.
5. Inspect a few evidence citations manually.
6. Make sure generated logs/results are not committed.

## Common Problems

`Ollama server did not become ready`

Check `logs/ollama-server-<job_id>.log`. The wrapper tries four ports by
default, but another local process or a model load failure can still break
startup.

`No supported answer was produced...`

The model output was missing or unparsable for that question. Improve the
reasoner/verifier prompts, reduce prompt size, or make the output schema stricter.

`done_reason=length` in the trace

The model hit an output or context limit. Try smaller batches, shorter evidence
summaries, stricter JSON output, or a larger context setting if the model and GPU
can handle it.

Wrong evidence path in an answer

Answers should cite frozen evidence paths such as:

```text
evidence/repo/src/train.py
evidence/repo/slurm/train-one.slurm
evidence/results/comparison.csv
```

Do not cite hackathon infrastructure files such as `HACKATHON.md`,
`configs/hackathon_workflow.json`, or `src/hackathon_workflow.py` as evidence for
the training repository questions.

## Educator Documentation

For instructors and reviewers, please refer to the following resources:
- [ARTIFACT.md](ARTIFACT.md): Context on the artifact and paired repository.
- [LEARNING_OBJECTIVES.md](LEARNING_OBJECTIVES.md): Expected student outcomes.
- [PREREQUISITES.md](PREREQUISITES.md): Required skills before attempting the assignment.
- [ADAPTATION_NOTES.md](ADAPTATION_NOTES.md): Guidance on reusing this assignment.
- [CUSTOM_WORKLOAD_GUIDE.md](CUSTOM_WORKLOAD_GUIDE.md): How to adapt this to a new HPC workload.
- [sample_outputs/](sample_outputs/): Mock examples of generated traces and submissions.

## License

Code in this repository is released under the MIT License.

Educational materials, documentation, prompts, handouts, and assignment descriptions are released under the Creative Commons Attribution 4.0 International (CC BY 4.0) License, unless otherwise noted.
