#!/bin/bash

set -euo pipefail

cd /home/infres/vmorozov/PointCloudMaterials

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable on host $(hostname); run this script from the Slurm login host." >&2
  exit 1
fi

a_config="configs/simulation/atomistic/al/campaign_70304_mpa_130ps_source12345_seed35803.yaml"
b_config="configs/simulation/atomistic/al/campaign_70304_mpa_130ps_source12346_seed35831.yaml"
a_checkpoint="output/synthetic_data/al_homogeneous_campaign_70304_mpa_130ps_source12345_seed35803_20260720/checkpoints/replica_000"
b_checkpoint="output/synthetic_data/al_homogeneous_campaign_70304_mpa_130ps_source12346_seed35831_20260720/checkpoints/replica_000"

if [[ "$(<"${a_checkpoint}/LATEST")" != "step_000000115000" ]]; then
  echo "A extension must start from verified global step 115000." >&2
  exit 1
fi
if [[ "$(<"${b_checkpoint}/LATEST")" != "step_000000086000" ]]; then
  echo "B extension must start from verified global step 86000." >&2
  exit 1
fi

PYTHONPATH=. /home/infres/vmorozov/miniconda3/envs/pointnet/bin/python - <<'PY'
from pathlib import Path

from src.data_utils.synthetic.atomistic.homogeneous_resumable import (
    _load_and_verify_named_snapshot,
)

roots = (
    Path("output/synthetic_data/al_homogeneous_campaign_70304_mpa_130ps_source12345_seed35803_20260720/checkpoints/replica_000"),
    Path("output/synthetic_data/al_homogeneous_campaign_70304_mpa_130ps_source12346_seed35831_20260720/checkpoints/replica_000"),
)
for root in roots:
    snapshot_name = (root / "LATEST").read_text(encoding="utf-8").strip()
    _load_and_verify_named_snapshot(root / snapshot_name)
PY

# Both independent first segments may run concurrently on node53's H100 GPUs.
a_job=$(sbatch --parsable --nodelist=node53 --job-name=al70k_a_130ps \
  scripts/job_al_homogeneous_70304_mpa.sh "${a_config}")
b_job=$(sbatch --parsable --nodelist=node53 --job-name=al70k_b_130ps_1 \
  scripts/job_al_homogeneous_70304_mpa.sh "${b_config}")

# B has 49k steps remaining (~24.2 GPU-hours), just beyond one 23-hour allocation.
b_continuation=$(sbatch --parsable --nodelist=node53 \
  --dependency="afterany:${b_job}" --job-name=al70k_b_130ps_2 \
  scripts/job_al_homogeneous_70304_mpa.sh "${b_config}")

echo "A extension job: ${a_job}"
echo "B extension jobs: ${b_job} -> ${b_continuation}"
