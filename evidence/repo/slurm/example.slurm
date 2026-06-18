#!/bin/bash
# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Cluster/run wrapper; teams should configure via environment variables, not by editing this file.
#SBATCH --job-name=cifar67-hihi
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:05:00
#SBATCH --output=logs/compare-runs-%j.out
#SBATCH --error=logs/compare-runs-%j.err

echo "I love HPC Summer School 2026"