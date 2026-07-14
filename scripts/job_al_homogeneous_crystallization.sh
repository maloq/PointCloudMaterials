#!/bin/bash
#SBATCH --job-name=al_homogeneous_crystallization
#SBATCH --output=output/synthetic_data/al_homogeneous_crystallization_%j.out
#SBATCH --error=output/synthetic_data/al_homogeneous_crystallization_%j.err
#SBATCH --partition=H100
#SBATCH --account=s2a
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --requeue

set -euo pipefail

cd /home/infres/vmorozov/PointCloudMaterials
export PYTHONPATH=.
export PYTORCH_ALLOC_CONF=expandable_segments:True

/home/infres/vmorozov/miniconda3/envs/pointnet/bin/python -u \
  -m src.data_utils.synthetic.atomistic_homogeneous_crystallization \
  --config configs/data/atomistic_al_homogeneous_crystallization.yaml
