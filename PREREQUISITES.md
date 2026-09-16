# Prerequisites

## Mock artifact check

The public mock workflow needs only Python 3.10 or newer, this checkout, and a
writable output directory. It does not require Slurm, a GPU, CUDA, Apptainer,
Ollama, or a model cache. No evidence-based minimum RAM requirement is claimed,
and the release has not been verified on macOS.

Before attempting this hackathon, students should be familiar with the following concepts:

## Full Slurm and LLM teaching path
- **Basic Linux command-line usage:** Navigating directories, reading files, standard I/O redirection.
- **Basic Python:** Understanding how to run scripts, read and parse JSON configurations, and basic dictionary/list manipulation.
- **Introduction to Slurm job submission:** Knowing how to read a batch script and use `sbatch`, `squeue`, and `sacct`.
- One CUDA-capable GPU node, Apptainer, a site-provided Ollama server image,
  a Python client image, and a pre-staged model cache containing the configured
  models.

## Recommended Pre-Lab Checklist
- Login/SSH into the cluster environment.
- File navigation (clone repo, explore `evidence/` directory).
- Submit a simple Slurm job and inspect `stdout`/`stderr` logs.
- Run the provided baseline script locally in mock mode (`--mock`).

## Helpful but Optional
- Familiarity with LLM prompting and context window limits.
- Basic understanding of GPU/AI workloads (e.g., PyTorch training scripts).
- General container concepts (Apptainer/Docker).
