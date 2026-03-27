#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

predict_script="${repo_root}/src/analysis/pipeline.py"

checkpoints=(
  "output/2026-03-02/17-22-18/VICREG_FT_l512_N128_M80_RI_MAE_Invariant-epoch=11.ckpt"
  "output/2026-02-26/19-53-51/VICREG_l512_N128_M80_RI_MAE_Invariant-epoch=59.ckpt"
)

analysis_dirs=(
  "outputs/analysis_1"
  "outputs/analysis_2"
)

# Order must match checkpoints and analysis_dirs above. Each entry must be either:
#   default
#   IDS[;IDS...]
# where each IDS token is a comma-separated cluster list like 0,1,2.
visible_cluster_specs=(
  "1,3,4,5"
  "0,1,2"
  "0,1,2,3"
)

die() {
  echo "$1" >&2
  exit 1
}

validate_cluster_set() {
  local cluster_set="$1"
  local context="$2"
  if [[ ! "${cluster_set}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    die "Invalid ${context}: '${cluster_set}'. Expected comma-separated non-negative integer cluster IDs like '3,4,5'."
  fi

  local -A seen_cluster_ids=()
  local -a cluster_ids=()
  IFS=',' read -r -a cluster_ids <<< "${cluster_set}"
  for cluster_id in "${cluster_ids[@]}"; do
    if [[ -n "${seen_cluster_ids[${cluster_id}]+x}" ]]; then
      die "Invalid ${context}: duplicate cluster ID '${cluster_id}' in '${cluster_set}'."
    fi
    seen_cluster_ids["${cluster_id}"]=1
  done
}

validate_visible_cluster_spec() {
  local spec="$1"
  local context="$2"

  if [[ "${spec}" == "default" ]]; then
    return 0
  fi

  if [[ -z "${spec}" ]]; then
    die "Invalid ${context}: empty specification. Expected 'default' or 'IDS[;IDS...]'."
  fi

  local -a cluster_sets_in_spec=()
  IFS=';' read -r -a cluster_sets_in_spec <<< "${spec}"
  if [[ "${#cluster_sets_in_spec[@]}" -eq 0 ]]; then
    die "Invalid ${context}: '${spec}'. Expected 'default' or 'IDS[;IDS...]'."
  fi

  local set_idx
  for set_idx in "${!cluster_sets_in_spec[@]}"; do
    validate_cluster_set \
      "${cluster_sets_in_spec[$set_idx]}" \
      "${context} set $((set_idx + 1))"
  done
}

format_visible_cluster_spec() {
  local spec="$1"
  if [[ "${spec}" == "default" ]]; then
    printf "<default from analysis config>"
    return 0
  fi
  printf "%s" "${spec//;/ | }"
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_selected_checkpoint_analyses_with_clusters.sh

Per-checkpoint defaults:
  Edit visible_cluster_specs in this script.
  Each entry must be either:
    default
    IDS[;IDS...]

Shared analysis settings come from configs/analysis/checkpoint_analysis.yaml.
This helper only overrides checkpoint.path, checkpoint.output_dir,
figure_set.raytrace.enabled, and figure_set.visible_cluster_sets.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unexpected argument '$1'. Edit visible_cluster_specs or configs/analysis/checkpoint_analysis.yaml."
      ;;
  esac
done

if [[ ! -f "${predict_script}" ]]; then
  die "Missing analysis script: ${predict_script}"
fi

if [[ "${#checkpoints[@]}" -ne "${#analysis_dirs[@]}" ]]; then
  die "Internal error: checkpoints and analysis_dirs must have the same length."
fi

if [[ "${#checkpoints[@]}" -ne "${#visible_cluster_specs[@]}" ]]; then
  die "Internal error: checkpoints and visible_cluster_specs must have the same length."
fi

for idx in "${!visible_cluster_specs[@]}"; do
  validate_visible_cluster_spec \
    "${visible_cluster_specs[$idx]}" \
    "visible_cluster_specs[$((idx + 1))]"
done

cd "${repo_root}"

for idx in "${!checkpoints[@]}"; do
  ckpt_rel="${checkpoints[$idx]}"
  out_rel="${analysis_dirs[$idx]}"
  ckpt_path="${repo_root}/${ckpt_rel}"
  out_dir="${repo_root}/${out_rel}"
  visible_spec="${visible_cluster_specs[$idx]}"

  if [[ ! -f "${ckpt_path}" ]]; then
    die "Checkpoint does not exist: ${ckpt_path}"
  fi

  mkdir -p "${out_dir}"

  echo "[$((idx + 1))/${#checkpoints[@]}] Running analysis for ${ckpt_rel}"
  echo "    Output: ${out_rel}"
  echo "    Visible cluster sets: $(format_visible_cluster_spec "${visible_spec}")"

  python - "${ckpt_path}" "${out_dir}" "${visible_spec}" <<'PY'
import sys
from omegaconf import open_dict

from src.analysis import (
    load_checkpoint_analysis_config,
    run_post_training_analysis,
)

checkpoint_path, output_dir, visible_spec = sys.argv[1:4]
analysis_cfg = load_checkpoint_analysis_config()
with open_dict(analysis_cfg):
    analysis_cfg.checkpoint.path = checkpoint_path
    analysis_cfg.checkpoint.output_dir = output_dir
    analysis_cfg.figure_set.raytrace.enabled = True
    if visible_spec == "default":
        analysis_cfg.figure_set.visible_cluster_sets = list(
            analysis_cfg.figure_set.visible_cluster_sets
        )
    else:
        analysis_cfg.figure_set.visible_cluster_sets = [
            [int(cluster_id) for cluster_id in cluster_set.split(",")]
            for cluster_set in visible_spec.split(";")
        ]

run_post_training_analysis(analysis_cfg=analysis_cfg)
PY
done
