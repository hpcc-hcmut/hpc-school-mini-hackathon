# Artifact Information

This repository is the student-facing mini-hackathon package accompanying the
EduHPC'26 paper *An AI-Anchored Pathway into HPC Practice: Design and Initial
Evaluation of a Three-Day Summer School in Vietnam*. It is one instructional
component of the three-day summer school described in that paper, rather than
the complete school or a record of every historical classroom run.

The paired instructor-facing leaderboard package is available at:
https://github.com/hpcc-hcmut/hpc-summer-school-leaderboard

## Student-Facing Components

This repository contains:
- **Evidence Corpus:** Frozen HPC repository (`evidence/repo/`) and benchmark results (`evidence/results/`).
- **Starter Workflow:** A working multi-agent orchestration pipeline (`src/`).
- **Prompts & Configs:** Baseline LLM prompts (`prompts/`) and workflow configurations (`configs/`).
- **Slurm Infrastructure:** Job wrappers (`slurm/`) to spawn agents inside Apptainer.
- **Sample Questions:** A set of public scaffolding questions (`qa_public.json`).

## Artifact scope

This public release supports inspection, execution, and adaptation of the
student-facing mini-hackathon. It includes a public dual-model baseline, mock
mode for structure/control-flow testing, and documentation for the reference
Slurm/Ollama path. Site-provided containers, model caches, and local deployment
configuration remain necessary for the full path; mock mode does not reproduce
student performance or historical competition scores.

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
