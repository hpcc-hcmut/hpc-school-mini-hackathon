# Artifact Information

This repository is the student-facing package for the Peachy assignment paper:

"A Slurm-Based Multi-Agent LLM Hackathon for Teaching Evidence-Grounded AI/HPC Workflows"

The paired instructor-facing leaderboard package is available at:
https://github.com/hpcc-hcmut/hpc-summer-school-leaderboard

## Student-Facing Components

This repository contains:
- **Evidence Corpus:** Frozen HPC repository (`evidence/repo/`) and benchmark results (`evidence/results/`).
- **Starter Workflow:** A working multi-agent orchestration pipeline (`src/`).
- **Prompts & Configs:** Baseline LLM prompts (`prompts/`) and workflow configurations (`configs/`).
- **Slurm Infrastructure:** Job wrappers (`slurm/`) to spawn agents inside Apptainer.
- **Sample Questions:** A set of public scaffolding questions (`qa_public.json`).

## Supported Paper Claims

This artifact provides the reusable foundation discussed in the paper, enabling instructors to easily deploy a realistic AI/HPC workflow hackathon with minimal infrastructure overhead, adaptable along content, workflow, and infrastructure axes.

## Execution envelopes

Mock mode is a Python 3.10+ structure/control-flow test and needs no HPC or LLM
runtime. The full reference path is a single Slurm job on one GPU node, running
an Apptainer Ollama server from a pre-staged cache and the Python workflow in a
client image; the workflow talks to Ollama over loopback and writes JSON outputs,
then may POST to the leaderboard.

The original HCMUT deployment requested one node, one task, four CPUs, one GPU,
and 15 minutes, and used a V100 32 GB. That GPU model is historical, not an
intrinsic requirement. Partition/QoS names, shared paths, images, and cache
locations are site-specific. Cluster size and interconnect do not affect this
single-node workflow.
