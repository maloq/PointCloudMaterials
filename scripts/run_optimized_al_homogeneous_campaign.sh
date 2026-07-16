#!/bin/bash

set -euo pipefail

REPOSITORY_ROOT=/home/infres/vmorozov/PointCloudMaterials
PYTHON=${PYTHON:?PYTHON must point to the isolated MACE 0.3.16 runtime}
SELECTION_REPORT=output/synthetic_data/al_potential_benchmark/selection.json
DEVICES=${DEVICES:-0,1}

cd "$REPOSITORY_ROOT"

if [ ! -x "$PYTHON" ]; then
    echo "PYTHON is not executable: $PYTHON" >&2
    exit 1
fi
if [ ! -f "$SELECTION_REPORT" ]; then
    echo "Required potential-selection report is missing: $SELECTION_REPORT" >&2
    exit 1
fi

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

"$PYTHON" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

expected = {
    "mace-torch": "0.3.16",
    "ase": "3.28.0",
    "cuequivariance": "0.10.0",
    "cuequivariance-torch": "0.10.0",
    "cuequivariance-ops-torch-cu12": "0.10.0",
}
for package, required in expected.items():
    try:
        observed = version(package)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"Required package {package}=={required} is absent from PYTHON."
        ) from exc
    if observed != required:
        raise RuntimeError(
            f"Optimized campaign requires {package}=={required}, observed {observed}."
        )
PY

selected_model=$("$PYTHON" -c '
import json, sys

path, devices_raw = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    report = json.load(handle)
if (
    report.get("schema_version") != 4
    or report.get("report_type") != "al_crystallization_mlip_selection"
    or report.get("policy_version")
    != "scientific_quality_runtime_advisory_v3"
):
    raise RuntimeError(f"{path}: invalid potential-selection report")
devices = [item.strip() for item in devices_raw.split(",")]
if not devices or any(not item for item in devices) or len(set(devices)) != len(devices):
    raise RuntimeError(
        f"DEVICES must contain unique non-empty comma-separated IDs, got {devices_raw!r}"
    )
model = report.get("selected_model_name")
baseline_name = report.get("baseline_model_name")
candidate_name = report.get("candidate_model_name")
if model not in {baseline_name, candidate_name}:
    raise RuntimeError(
        f"{path}: selected model {model!r} is neither baseline nor candidate"
    )
print(model)
' "$SELECTION_REPORT" "$DEVICES")

case "$selected_model" in
    mace-mpa-0-medium)
        source_root=output/synthetic_data/al_liquid_source_16384_compiled_mpa_500K
        campaign_config=configs/simulation/atomistic/al/campaign_16384_mpa.yaml
        ;;
    mace-mh-1-omat-pbe)
        source_root=output/synthetic_data/al_liquid_source_16384_compiled_mh1_omat_pbe_500K
        campaign_config=configs/simulation/atomistic/al/campaign_16384_mh1.yaml
        ;;
    *)
        echo "Unsupported selected model: $selected_model" >&2
        exit 1
        ;;
esac

if [ ! -d "$source_root" ]; then
    echo "Selected immutable source is missing: $source_root" >&2
    exit 1
fi

# Bind the selection report, campaign workload, source hashes, and phase semantics before
# constructing a GPU model.
"$PYTHON" -c '
import sys
from src.data_utils.synthetic.atomistic.homogeneous_campaign_config import load_homogeneous_campaign_config
from src.data_utils.synthetic.atomistic.homogeneous_generator import _load_source_liquid
config = load_homogeneous_campaign_config(sys.argv[1])
_load_source_liquid(config.homogeneous)
' "$campaign_config"

echo "Starting/resuming $selected_model on devices=$DEVICES through its configured endpoint."

exec "$PYTHON" -u \
    -m src.data_utils.synthetic.atomistic_homogeneous_campaign \
    run \
    --config "$campaign_config" \
    --devices "$DEVICES"
