#!/bin/bash
#SBATCH --job-name=al70k_bf16_compare
#SBATCH --output=output/slurm_outputs/%x_%j.out
#SBATCH --error=output/slurm_outputs/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --account=s2a
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

cd /home/infres/vmorozov/PointCloudMaterials
export PYTHONPATH=.

comparison_root="output/synthetic_data/al_homogeneous_campaign_70304_mpa_bf16_comparison_20260724"

/home/infres/vmorozov/miniconda3/envs/pointnet/bin/python -u \
  scripts/compare_homogeneous_precision.py \
  --fp32-campaigns \
    output/synthetic_data/al_homogeneous_campaign_70304_mpa_130ps_source12345_seed35803_20260720 \
    output/synthetic_data/al_homogeneous_campaign_70304_mpa_130ps_source12346_seed35831_20260720 \
  --bf16-campaigns \
    output/synthetic_data/al_homogeneous_campaign_70304_mpa_bf16_130ps_source12345_seed35803_20260724 \
    output/synthetic_data/al_homogeneous_campaign_70304_mpa_bf16_130ps_source12346_seed35831_20260724 \
  --output-json "${comparison_root}/comparison.json" \
  --output-png "${comparison_root}/crystallization_comparison.png"
