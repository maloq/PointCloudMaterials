#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

predict_script="${repo_root}/src/training_methods/contrastive_learning/predict_and_visualize.py"

checkpoints=(
  "output/2026-03-02/17-22-18/VICREG_FT_l512_N128_M80_RI_MAE_Invariant-epoch=11.ckpt"
  "output/2026-02-26/19-53-51/VICREG_l512_N128_M80_RI_MAE_Invariant-epoch=59.ckpt"
  "output/2026-02-25/23-55-50/VICREG_l768_N128_M80_RI_MAE_Invariant-epoch=119.ckpt"
)

analysis_dirs=(
  "outputs/analysis_1"
  "outputs/analysis_2"
  "outputs/analysis_3"
)

if [[ ! -f "${predict_script}" ]]; then
  echo "Missing analysis script: ${predict_script}" >&2
  exit 1
fi

if [[ "$#" -ne 0 ]]; then
  echo "This helper no longer forwards CLI arguments to predict_and_visualize.py." >&2
  echo "Edit configs/analysis/checkpoint_analysis.yaml for shared settings." >&2
  exit 1
fi

if [[ "${#checkpoints[@]}" -ne "${#analysis_dirs[@]}" ]]; then
  echo "Internal error: checkpoints and analysis_dirs must have the same length." >&2
  exit 1
fi

cd "${repo_root}"

for idx in "${!checkpoints[@]}"; do
  ckpt_rel="${checkpoints[$idx]}"
  out_rel="${analysis_dirs[$idx]}"
  ckpt_path="${repo_root}/${ckpt_rel}"
  out_dir="${repo_root}/${out_rel}"

  if [[ ! -f "${ckpt_path}" ]]; then
    echo "Checkpoint does not exist: ${ckpt_path}" >&2
    exit 1
  fi

  mkdir -p "${out_dir}"

  echo "[$((idx + 1))/${#checkpoints[@]}] Running analysis for ${ckpt_rel}"
  echo "    Output: ${out_rel}"

  python - "${ckpt_path}" "${out_dir}" <<'PY'
import sys
from omegaconf import open_dict

from src.training_methods.contrastive_learning.predict_and_visualize import (
    load_checkpoint_analysis_config,
    run_post_training_analysis,
)

checkpoint_path, output_dir = sys.argv[1:3]
analysis_cfg = load_checkpoint_analysis_config()
with open_dict(analysis_cfg):
    analysis_cfg.checkpoint.path = checkpoint_path
    analysis_cfg.checkpoint.output_dir = output_dir
    analysis_cfg.figure_set.raytrace.enabled = True

run_post_training_analysis(analysis_cfg=analysis_cfg)
PY
done
