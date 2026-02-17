#!/bin/bash
#SBATCH --job-name=VICREG_MN40_ENC
#SBATCH --output=output/slurm_outputs/%x_%A_%a.out
#SBATCH --error=output/slurm_outputs/%x_%A_%a.err
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
# Run all 5 encoder tasks in parallel (up to 5 concurrent array jobs).
#SBATCH --array=0-4%5

set -eo pipefail

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Some conda activation hooks reference optional env vars (e.g. MAGPLUS_HOME).
# Disable nounset around activation to avoid premature job exit.
set +u
source /home/infres/vmorozov/miniconda3/etc/profile.d/conda.sh
conda activate pointnet
set -u

cd /home/infres/vmorozov/PointCloudMaterials
export PYTHONPATH=${PYTHONPATH:-}:/home/infres/vmorozov/PointCloudMaterials
mkdir -p output/slurm_outputs

ENCODER_KEYS=(
  "pointnet"
  "vn_pointnet"
  "dgcnn"
  "vn_dgcnn"
  "vn_revnet"
)

ENCODER_OVERRIDES=(
  "encoder.name=PnE_L"
  "encoder.name=PnE_VN"
  "encoder.name=DGCNN"
  "encoder.name=VN_DGCNN"
  "encoder.name=VN_REVNET_Backbone"
)

task_idx="${SLURM_ARRAY_TASK_ID:-0}"
if (( task_idx < 0 || task_idx >= ${#ENCODER_KEYS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${task_idx}. Expected 0..$(( ${#ENCODER_KEYS[@]} - 1 ))"
  exit 1
fi

encoder_key="${ENCODER_KEYS[$task_idx]}"
encoder_override="${ENCODER_OVERRIDES[$task_idx]}"

# Keep invariant latent width identical across all architectures.
latent_size=96
run_dir_override="hydra.run.dir=output/modelnet40_vicreg/${encoder_key}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"

COMMON_OVERRIDES=(
  "latent_size=${latent_size}"
  "analysis_grain_enabled=false"
  "analysis_hdbscan_enabled=false"
  "test_cluster_eval_k=40"
  "experiment_name=VICREG_MODELNET40_${encoder_key}"
  "${encoder_override}"
  "${run_dir_override}"
  "devices=[0]"
  "auto_batch_size_search=true"
  "accumulate_grad_batches=10"
)

echo "Selected encoder: ${encoder_key}"
echo "Hydra overrides: ${COMMON_OVERRIDES[*]} $*"

# Run directly inside the batch allocation. Some clusters reject a nested srun
# step with "Invalid generic resource (gres) specification".
python src/training_methods/contrastive_learning/train_contrastive.py \
  --config-name vicreg_vn_modelnet40.yaml \
  "${COMMON_OVERRIDES[@]}" \
  "$@"

echo "Job finished at: $(date)"
