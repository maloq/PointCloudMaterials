#!/bin/bash

set -euo pipefail

cd /home/infres/vmorozov/PointCloudMaterials

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable on host $(hostname); run this script from the Slurm login host." >&2
  exit 1
fi

a_config="configs/simulation/atomistic/al/campaign_70304_mpa_bf16_130ps_source12345_seed35803.yaml"
b_config="configs/simulation/atomistic/al/campaign_70304_mpa_bf16_130ps_source12346_seed35831.yaml"

# One matched full-graph FP32/BF16 force+stress evaluation gates both replicas.
smoke_job=$(sbatch --parsable --nodelist=node53 \
  scripts/job_validate_mace_bf16_70304.sh)

submit_chain() {
  local replica_name="$1"
  local config_path="$2"
  local dependency="$3"
  local job_id
  local segment

  job_id=$(sbatch --parsable --nodelist=node53 \
    --dependency="afterok:${dependency}" \
    --job-name="${replica_name}_1" \
    scripts/job_al_homogeneous_70304_mpa.sh "${config_path}")
  printf '%s' "${job_id}"
  for segment in 2 3 4; do
    job_id=$(sbatch --parsable --nodelist=node53 \
      --dependency="afterany:${job_id}" \
      --job-name="${replica_name}_${segment}" \
      scripts/job_al_homogeneous_70304_mpa.sh "${config_path}")
    printf ' -> %s' "${job_id}"
  done
  LAST_CHAIN_JOB_ID="${job_id}"
  printf '\n'
}

echo "BF16 validation job: ${smoke_job}"
printf 'Replica A jobs: '
submit_chain "al70k_bf16_a" "${a_config}" "${smoke_job}"
a_final_job="${LAST_CHAIN_JOB_ID}"
printf 'Replica B jobs: '
submit_chain "al70k_bf16_b" "${b_config}" "${smoke_job}"
b_final_job="${LAST_CHAIN_JOB_ID}"

comparison_job=$(sbatch --parsable \
  --dependency="afterok:${a_final_job}:${b_final_job}" \
  scripts/job_compare_homogeneous_70304_mpa_bf16.sh)
echo "Completed-campaign comparison job: ${comparison_job}"
