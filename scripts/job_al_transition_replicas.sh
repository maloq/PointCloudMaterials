#!/bin/bash
#SBATCH --job-name=al_transition_replicas
#SBATCH --output=output/synthetic_data/al_transition_replicas_%j.out
#SBATCH --error=output/synthetic_data/al_transition_replicas_%j.err
#SBATCH --partition=H100
#SBATCH --account=s2a
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --requeue

set -euo pipefail

cd /home/infres/vmorozov/PointCloudMaterials
export PYTHONPATH=.
export PYTORCH_ALLOC_CONF=expandable_segments:True
CONFIG_PATH="${1:-configs/simulation/atomistic/al/phase_transition_70304_mpa.yaml}"

/home/infres/vmorozov/miniconda3/envs/pointnet/bin/python -u \
  -m src.data_utils.synthetic.atomistic_transition_replicas \
  --config "$CONFIG_PATH"
