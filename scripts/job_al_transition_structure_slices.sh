#!/bin/bash
#SBATCH --job-name=al_transition_slices
#SBATCH --output=output/synthetic_data/al_transition_slices_%j.out
#SBATCH --error=output/synthetic_data/al_transition_slices_%j.err
#SBATCH --partition=CPU
#SBATCH --account=s2a
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --requeue

set -euo pipefail

cd /home/infres/vmorozov/PointCloudMaterials
export PYTHONPATH=.

/home/infres/vmorozov/miniconda3/envs/pointnet/bin/python -u \
  -m src.data_utils.synthetic.atomistic_transition_structure_slices \
  --config configs/data/atomistic_al_phase_transition.yaml \
  --dataset output/synthetic_data/al_direct_coexistence_70304 \
  --dataset output/synthetic_data/al_direct_coexistence_70304_replica_001 \
  --dataset output/synthetic_data/al_direct_coexistence_70304_replica_002 \
  --dataset output/synthetic_data/al_direct_coexistence_70304_replica_003
