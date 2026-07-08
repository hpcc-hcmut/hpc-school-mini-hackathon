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
