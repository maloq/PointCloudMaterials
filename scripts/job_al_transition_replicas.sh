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
#SBATCH --time=12:00:00
#SBATCH --requeue

set -euo pipefail

cd /home/infres/vmorozov/PointCloudMaterials
export PYTHONPATH=.
export PYTORCH_ALLOC_CONF=expandable_segments:True

/home/infres/vmorozov/miniconda3/envs/pointnet/bin/python -u \
  -m src.data_utils.synthetic.atomistic_transition_replicas \
  --config configs/data/atomistic_al_phase_transition.yaml \
  --replica 24680 output/synthetic_data/al_direct_coexistence_70304 \
  --replica 24681 output/synthetic_data/al_direct_coexistence_70304_replica_001 \
  --replica 24682 output/synthetic_data/al_direct_coexistence_70304_replica_002 \
  --replica 24683 output/synthetic_data/al_direct_coexistence_70304_replica_003
