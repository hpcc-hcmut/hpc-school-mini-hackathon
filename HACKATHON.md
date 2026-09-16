<!--
@author HCMUT HPC Summer School Hackathon Organizers
@permission DO_NOT_EDIT
@note Student-facing instructions. Teams should follow this file, not modify it.
-->

# Hackathon: Parallel LLM Agents for HPC Resource Q&A

This starter keeps the original CIFAR-100 ResNet training package intact and adds a Session 7 style LLM workflow for the hackathon.

## Goal

Build a traceable workflow that answers repository questions with evidence from code, Slurm scripts, configs, and benchmark CSV files under `evidence/results/`.

The provided starter demonstrates:

- multi-model routing by agent role
- parallel evidence extraction agents
- aggregator, reasoner, and verifier stages
- JSONL trace events for every LLM call
- leaderboard-style submission JSON
- optional backend submission

## Run Locally With Mock Outputs

This structure/control-flow check requires Python 3.10+, the public checkout,
and a writable output directory. It does not need Slurm, GPU/CUDA, Apptainer,
Ollama, or cached models, and it is not an LLM-quality evaluation.

```bash
python3 src/hackathon_workflow.py \
  --source-dir . \
  --config configs/hackathon_workflow.json \
  --output-dir results \
  --team-id team01 \
  --team-name "Team 01" \
  --mock
```

This produces:

```text
results/hackathon_trace_local.jsonl
results/hackathon_submission_local.json
```

The `results/` directory is for runtime outputs. Static benchmark evidence lives in `evidence/results/`.

## Run On Slurm

The reference path is: Slurm job → one GPU node → Apptainer Ollama server →
pre-staged local model cache → loopback Ollama API → Apptainer Python workflow
→ JSON trace and submission → optional leaderboard POST.

Before submitting, instructors must set `CLIENT_IMAGE`, `SERVER_IMAGE`,
`OLLAMA_CACHE`, and `ALLOWED_MODELS_FILE`. The configured cache/allow-list must
contain `llama3.2:3b` and `qwen3:4b`. Models should normally be staged before the
class rather than pulled from compute nodes.

```bash
sbatch slurm/hackathon-parallel-workflow.slurm
```

Optional environment variables:

```bash
TEAM_ID=team01
TEAM_NAME="Team 01"
BACKEND_URL=http://leaderboard.example
TEAM_TOKEN=<token>
OLLAMA_MAX_LOADED_MODELS=2
OLLAMA_NUM_PARALLEL=4
OLLAMA_CONTEXT_LENGTH=16384
```

## Files To Modify During The Hackathon

- `configs/hackathon_workflow.json`: route different agents to different models and tune parallelism.
- `prompts/*.txt`: improve extraction, reasoning, and verification prompts.
- `src/hackathon_workflow.py`: change workflow structure if your team wants a different agent design.

## Scoring Signals

The backend can score both answer quality and workflow design using:

- exact answers and evidence citations
- number of agents
- whether extraction agents ran in parallel
- models used per role
- number of LLM calls
- estimated token usage
- total runtime
- verifier and aggregator presence

## Important Warning

Parallel agents are useful, but not free. Higher `OLLAMA_NUM_PARALLEL`, longer context, and more loaded models increase memory pressure. A practical strategy is:

- small model for extraction agents
- stronger model for aggregation and reasoning
- small model for verification
- use enough context for thinking models, or compact prompts with line-numbered evidence
