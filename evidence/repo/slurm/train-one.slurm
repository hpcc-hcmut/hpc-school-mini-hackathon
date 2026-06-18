#!/bin/bash
# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Cluster/run wrapper; teams should configure via environment variables, not by editing this file.
#SBATCH --job-name=cifar100-resnet
#SBATCH --partition=gpu-queue
#SBATCH --qos=gpu-q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/train-one-%j.out
#SBATCH --error=logs/train-one-%j.err

set -euo pipefail

CONFIG="${1:?Usage: sbatch [--cpus-per-task=N] slurm/train-one.slurm CONFIG.json}"
IMAGE="/share/summer-school/g001/containers/hpc-school-ai.sif"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  exit 2
fi

mkdir -p logs results checkpoints

CONFIG_METADATA="$(
  apptainer exec "$IMAGE" python3 -c \
    'import json,sys; c=json.load(open(sys.argv[1])); print(c["experiment_name"], c["model_name"], sep="\t")' \
    "$CONFIG"
)"
IFS=$'\t' read -r EXPERIMENT_NAME MODEL_NAME <<< "$CONFIG_METADATA"

GPU_TIMELINE="results/gpu_timeline_${SLURM_JOB_ID}_${EXPERIMENT_NAME}.csv"
GPU_SUMMARY="results/gpu_summary_${SLURM_JOB_ID}_${EXPERIMENT_NAME}.csv"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Config: $CONFIG"
echo "Experiment: $EXPERIMENT_NAME"
echo "Model: $MODEL_NAME"
nvidia-smi

nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw,temperature.gpu \
  --format=csv \
  -l 1 > "$GPU_TIMELINE" &
NSMI_PID=$!

stop_gpu_monitor() {
  if [[ -n "${NSMI_PID:-}" ]]; then
    kill "$NSMI_PID" 2>/dev/null || true
    wait "$NSMI_PID" 2>/dev/null || true
    NSMI_PID=""
  fi
}
trap stop_gpu_monitor EXIT

set +e
apptainer exec --nv "$IMAGE" python3 src/train.py \
  --config "$CONFIG" \
  --job-id "$SLURM_JOB_ID" \
  --cpus-per-task "${SLURM_CPUS_PER_TASK:-unknown}"
TRAIN_STATUS=$?
set -e

stop_gpu_monitor

set +e
apptainer exec "$IMAGE" python3 src/summarize_gpu_log.py \
  --input "$GPU_TIMELINE" \
  --output "$GPU_SUMMARY" \
  --job-id "$SLURM_JOB_ID" \
  --experiment-name "$EXPERIMENT_NAME" \
  --model-name "$MODEL_NAME"
SUMMARY_STATUS=$?
set -e

echo "Generated files:"
ls -lh results/*"${SLURM_JOB_ID}_${EXPERIMENT_NAME}"*.csv 2>/dev/null || true
ls -lh checkpoints/*"${SLURM_JOB_ID}_${EXPERIMENT_NAME}"*.pt 2>/dev/null || true

if [[ "$TRAIN_STATUS" -ne 0 ]]; then
  exit "$TRAIN_STATUS"
fi
exit "$SUMMARY_STATUS"
