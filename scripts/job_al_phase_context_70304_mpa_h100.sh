#!/bin/bash
#SBATCH --job-name=al_70304_mpa
#SBATCH --output=output/slurm_outputs/al_70304_mpa_%j.out
#SBATCH --error=output/slurm_outputs/al_70304_mpa_%j.err
#SBATCH --partition=H100
#SBATCH --account=s2a
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=23:00:00
#SBATCH --requeue

set -euo pipefail

cd /home/infres/vmorozov/PointCloudMaterials
export PYTHONPATH=.
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=0

/home/infres/vmorozov/miniconda3/envs/pointnet/bin/python -u \
  -m src.data_utils.synthetic.atomistic_generator \
  --config configs/simulation/atomistic/al/phase_context_70304_mpa.yaml
