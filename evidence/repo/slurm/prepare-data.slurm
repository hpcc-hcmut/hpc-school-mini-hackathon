#!/bin/bash
# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Cluster/run wrapper; teams should configure via environment variables, not by editing this file.
#SBATCH --job-name=prepare-cifar100
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/prepare-data-%j.out
#SBATCH --error=logs/prepare-data-%j.err

set -euo pipefail

# Edit path for your cluster
export IMAGE=/share/summer-school/g001/containers/hpc-school-ai.sif

mkdir -p data logs

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Image: $IMAGE"
echo "Working directory: $(pwd)"

apptainer exec "$IMAGE" python3 src/data/prepare_cifar100_imagefolder.py \
  --download-root data \
  --output-dir data/cifar100 \
  --val-count 5000 \
  --seed 42

echo "Checking dataset counts..."

echo -n "train images: "
find data/cifar100/train -type f | wc -l

echo -n "val images: "
find data/cifar100/val -type f | wc -l

echo -n "test images: "
find data/cifar100/test -type f | wc -l

echo -n "train classes: "
find data/cifar100/train -mindepth 1 -maxdepth 1 -type d | wc -l

echo -n "val classes: "
find data/cifar100/val -mindepth 1 -maxdepth 1 -type d | wc -l

echo -n "test classes: "
find data/cifar100/test -mindepth 1 -maxdepth 1 -type d | wc -l

echo "Done."