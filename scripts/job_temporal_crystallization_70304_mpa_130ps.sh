#!/bin/bash
#SBATCH --job-name=temporal_70k_130ps
#SBATCH --output=output/slurm_outputs/%x_%j.out
#SBATCH --error=output/slurm_outputs/%x_%j.err
#SBATCH --partition=L40S
#SBATCH --account=s2a
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=08:00:00

set -euo pipefail

cd /home/infres/vmorozov/PointCloudMaterials
export PYTHONPATH=.
export MPLBACKEND=Agg

/home/infres/vmorozov/miniconda3/envs/pointnet/bin/python -u \
  scripts/run_temporal_crystallization_70304_mpa_130ps.py \
  --replica both
