#!/bin/bash
# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Cluster/run wrapper; teams should configure via environment variables, not by editing this file.
#SBATCH --job-name=cifar100-compare
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/compare-runs-%j.out
#SBATCH --error=logs/compare-runs-%j.err

set -euo pipefail

IMAGE="/share/summer-school/g001/containers/hpc-school-ai.sif"
mkdir -p logs results

apptainer exec "$IMAGE" python3 src/compare_runs.py \
  --training-summaries results/training_summary_*.csv \
  --gpu-summaries results/gpu_summary_*.csv \
  --output results/comparison.csv

cat results/comparison.csv
