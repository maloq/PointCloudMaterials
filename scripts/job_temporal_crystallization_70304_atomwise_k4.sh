#!/bin/bash
#SBATCH --job-name=temporal_70k_atom_k4
#SBATCH --output=output/slurm_outputs/%x_%j.out
#SBATCH --error=output/slurm_outputs/%x_%j.err
#SBATCH --partition=H100
#SBATCH --account=s2a
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <source12345_seed35803|source12346_seed35831>" >&2
  exit 2
fi

cd /home/infres/vmorozov/PointCloudMaterials
export PYTHONPATH=.
export MPLBACKEND=Agg

/home/infres/vmorozov/miniconda3/envs/pointnet/bin/python -u \
  scripts/run_temporal_crystallization_70304_atomwise_k4.py \
  --replica "$1" \
  --inference-batch-size 4096
