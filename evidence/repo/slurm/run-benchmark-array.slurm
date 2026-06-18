#!/bin/bash
# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Cluster/run wrapper; teams should configure via environment variables, not by editing this file.
#SBATCH --job-name=cifar100-benchmark
#SBATCH --partition=gpu-queue
#SBATCH --qos=gpu-q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/benchmark-%A_%a.out
#SBATCH --error=logs/benchmark-%A_%a.err

set -euo pipefail

CONFIGS=(
  configs/resnet18_bs128_workers4_cpus4.json
  configs/resnet34_bs128_workers4_cpus4.json
  configs/resnet50_bs128_workers4_cpus4.json
  configs/resnet101_bs64_workers4_cpus4.json
  configs/resnet152_bs64_workers4_cpus4.json
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Array task $SLURM_ARRAY_TASK_ID selected $CONFIG"
bash slurm/train-one.slurm "$CONFIG"
