#!/bin/bash
#SBATCH --job-name=al70k_bf16_smoke
#SBATCH --output=output/slurm_outputs/%x_%j.out
#SBATCH --error=output/slurm_outputs/%x_%j.err
#SBATCH --partition=H100
#SBATCH --account=s2a
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00

set -euo pipefail

cd /home/infres/vmorozov/PointCloudMaterials
export PYTHONPATH=.
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=0

/home/infres/vmorozov/miniconda3/envs/pointnet/bin/python -u \
  scripts/validate_mace_bf16.py \
  --campaign-config \
    configs/simulation/atomistic/al/campaign_70304_mpa_bf16_130ps_source12345_seed35803.yaml \
  --output-json \
    output/synthetic_data/al_mpa_70304_bf16_validation_20260724/source12345_full_graph.json
