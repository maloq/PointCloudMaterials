#!/bin/bash
#SBATCH --job-name=al_70k_source_md
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

SOURCE_CONFIG="${1:?first argument must be the liquid-source config}"
CAMPAIGN_CONFIG="${2:?second argument must be the campaign config}"

/home/infres/vmorozov/miniconda3/envs/pointnet/bin/python -u \
  -m src.data_utils.synthetic.atomistic_homogeneous_liquid_source \
  --config "$SOURCE_CONFIG"

/home/infres/vmorozov/miniconda3/envs/pointnet/bin/python -u \
  -m src.data_utils.synthetic.atomistic_homogeneous_campaign \
  run --config "$CAMPAIGN_CONFIG" --devices 0

ANALYSIS_WORKERS="${SLURM_CPUS_PER_TASK:-1}"

/home/infres/vmorozov/miniconda3/envs/pointnet/bin/python -u \
  -m src.data_utils.synthetic.atomistic_homogeneous_campaign \
  analyze --config "$CAMPAIGN_CONFIG" --workers "$ANALYSIS_WORKERS"

/home/infres/vmorozov/miniconda3/envs/pointnet/bin/python -u \
  scripts/plot_homogeneous_checkpoint.py \
  --campaign-config "$CAMPAIGN_CONFIG" \
  --include-structure-slices \
  --step-stamped
