#!/bin/bash
set -euo pipefail

cd /home/infres/vmorozov/PointCloudMaterials

ENCODER_KEYS=(
  "pointnet"
  "vn_pointnet"
  "dgcnn"
  "vn_dgcnn"
  "vn_revnet"
)

echo "Submitting ModelNet40 VICReg jobs (parallel encoders)..."

job_ids=()
for idx in "${!ENCODER_KEYS[@]}"; do
  key="${ENCODER_KEYS[$idx]}"
  submit_out=$(sbatch \
    --job-name="VICREG_MN40_${key}" \
    --array="${idx}-${idx}" \
    scripts/job_modelnet40_vicreg_encoders.sh \
    "$@")
  job_id=$(echo "${submit_out}" | awk '{print $4}')
  job_ids+=("${job_id}")
  echo "  submitted ${key} -> job ${job_id}"
done

echo "All submissions done."
echo "Job IDs: ${job_ids[*]}"
