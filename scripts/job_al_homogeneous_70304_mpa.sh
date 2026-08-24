#!/bin/bash
#SBATCH --job-name=al_homo_70k
#SBATCH --output=output/slurm_outputs/%x_%j.out
#SBATCH --error=output/slurm_outputs/%x_%j.err
#SBATCH --partition=H100
#SBATCH --account=s2a
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=23:00:00

set -euo pipefail

cd /home/infres/vmorozov/PointCloudMaterials
export PYTHONPATH=.
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=0

CONFIG_PATH="${1:?usage: sbatch job_al_homogeneous_70304_mpa.sh CAMPAIGN_CONFIG}"

/home/infres/vmorozov/miniconda3/envs/pointnet/bin/python -u \
  -m src.data_utils.synthetic.atomistic_homogeneous_campaign \
  run --config "$CONFIG_PATH" --devices 0

ANALYSIS_WORKERS="${SLURM_CPUS_PER_TASK:-1}"

/home/infres/vmorozov/miniconda3/envs/pointnet/bin/python -u \
  -m src.data_utils.synthetic.atomistic_homogeneous_campaign \
  analyze --config "$CONFIG_PATH" --workers "$ANALYSIS_WORKERS"

/home/infres/vmorozov/miniconda3/envs/pointnet/bin/python -u \
  scripts/plot_homogeneous_checkpoint.py \
  --campaign-config "$CONFIG_PATH" \
  --include-structure-slices \
  --step-stamped
