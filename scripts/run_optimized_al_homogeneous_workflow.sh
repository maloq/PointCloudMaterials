#!/bin/bash

set -euo pipefail

REPOSITORY_ROOT=/home/infres/vmorozov/PointCloudMaterials
PYTHON=${PYTHON:?PYTHON must point to the isolated MACE 0.3.16 runtime}
RUN_DIR=${RUN_DIR:?RUN_DIR is required}
DEVICES=${DEVICES:-0,1}
STAGE=preflight

cd "$REPOSITORY_ROOT"

if [ ! -x "$PYTHON" ]; then
    echo "PYTHON is not executable: $PYTHON" >&2
    exit 1
fi
IFS=',' read -r -a DEVICE_ARRAY <<< "$DEVICES"
if [ "${#DEVICE_ARRAY[@]}" -ne 2 ] || \
   [ -z "${DEVICE_ARRAY[0]}" ] || [ -z "${DEVICE_ARRAY[1]}" ] || \
   [ "${DEVICE_ARRAY[0]}" = "${DEVICE_ARRAY[1]}" ]; then
    echo "DEVICES must contain exactly two distinct GPU IDs, got $DEVICES" >&2
    exit 1
fi

mkdir -p "$RUN_DIR/logs"
if [ -e "$RUN_DIR/status.txt" ]; then
    echo "RUN_DIR already contains status.txt and will not be reused: $RUN_DIR" >&2
    exit 1
fi
printf '%s\n' "$$" > "$RUN_DIR/workflow.pid"

status() {
    printf '%s stage=%s %s\n' "$(date --iso-8601=seconds)" "$STAGE" "$1" \
        > "$RUN_DIR/status.txt"
}

declare -a ACTIVE_PIDS=()
terminate_children() {
    local pid
    for pid in "${ACTIVE_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${ACTIVE_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}

on_signal() {
    status "TERMINATING received_signal=$1"
    terminate_children
    exit 124
}

on_exit() {
    local exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        status "FAILED exit_code=$exit_code"
    fi
}

trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap on_exit EXIT

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
        raise RuntimeError(f"Required package is absent: {package}=={required}") from exc
    if observed != required:
        raise RuntimeError(
            f"Optimized workflow requires {package}=={required}, observed {observed}"
        )
PY

MPA_SOURCE=output/synthetic_data/al_liquid_source_16384_compiled_mpa_500K
MH1_SOURCE=output/synthetic_data/al_liquid_source_16384_compiled_mh1_omat_pbe_500K
SCIENTIFIC_REPORT=output/synthetic_data/al_potential_benchmark/report.json
PERFORMANCE_REPORT=output/synthetic_data/al_potential_benchmark/performance.json
SELECTION_REPORT=output/synthetic_data/al_potential_benchmark/selection.json

if [ -d "$MPA_SOURCE" ] && [ -d "$MH1_SOURCE" ]; then
    prepare_sources=false
elif [ ! -e "$MPA_SOURCE" ] && [ ! -e "$MH1_SOURCE" ]; then
    prepare_sources=true
else
    echo "Exactly one model-specific source exists; refusing mixed evidence." >&2
    exit 1
fi
run_scientific=$([ -f "$SCIENTIFIC_REPORT" ] && echo false || echo true)
run_performance=$([ -f "$PERFORMANCE_REPORT" ] && echo false || echo true)
run_selection=$([ -f "$SELECTION_REPORT" ] && echo false || echo true)
if [ "$prepare_sources" = true ] && [ "$run_performance" = false ]; then
    echo "Performance evidence exists while its source artifacts are absent." >&2
    exit 1
fi
if [ "$run_selection" = false ] && \
   { [ "$run_scientific" = true ] || [ "$run_performance" = true ]; }; then
    echo "Selection exists while prerequisite reports are absent." >&2
    exit 1
fi

if [ "$prepare_sources" = true ]; then
    STAGE=liquid_sources
    status "RUNNING devices=$DEVICES"
    CUDA_VISIBLE_DEVICES="${DEVICE_ARRAY[0]}" "$PYTHON" -u \
        -m src.data_utils.synthetic.atomistic_homogeneous_liquid_source \
        --config configs/simulation/atomistic/al/liquid_source_16384_mpa.yaml \
        > "$RUN_DIR/logs/liquid_source_mpa_gpu${DEVICE_ARRAY[0]}.log" 2>&1 &
    MPA_SOURCE_PID=$!
    CUDA_VISIBLE_DEVICES="${DEVICE_ARRAY[1]}" "$PYTHON" -u \
        -m src.data_utils.synthetic.atomistic_homogeneous_liquid_source \
        --config configs/simulation/atomistic/al/liquid_source_16384_mh1.yaml \
        > "$RUN_DIR/logs/liquid_source_mh1_gpu${DEVICE_ARRAY[1]}.log" 2>&1 &
    MH1_SOURCE_PID=$!
    ACTIVE_PIDS=("$MPA_SOURCE_PID" "$MH1_SOURCE_PID")
    if wait "$MPA_SOURCE_PID"; then
        :
    else
        exit_code=$?
        terminate_children
        echo "MPA liquid-source process failed with exit code $exit_code" >&2
        exit "$exit_code"
    fi
    if wait "$MH1_SOURCE_PID"; then
        :
    else
        exit_code=$?
        terminate_children
        echo "MH-1 liquid-source process failed with exit code $exit_code" >&2
        exit "$exit_code"
    fi
    ACTIVE_PIDS=()

fi

if [ "$run_scientific" = true ]; then
    STAGE=scientific_benchmark
    status "RUNNING device=${DEVICE_ARRAY[0]}"
    CUDA_VISIBLE_DEVICES="${DEVICE_ARRAY[0]}" "$PYTHON" -u \
        -m src.data_utils.synthetic.atomistic_potential_benchmark \
        --config configs/simulation/atomistic/al/potential_benchmark.yaml \
        > "$RUN_DIR/logs/scientific_benchmark_gpu${DEVICE_ARRAY[0]}.log" 2>&1 &
    ACTIVE_PIDS=("$!")
    wait "${ACTIVE_PIDS[0]}"
    ACTIVE_PIDS=()

fi

if [ "$run_performance" = true ]; then
    STAGE=performance_benchmark
    status "RUNNING device=${DEVICE_ARRAY[0]}"
    CUDA_VISIBLE_DEVICES="${DEVICE_ARRAY[0]}" "$PYTHON" -u \
        -m src.data_utils.synthetic.atomistic_potential_performance \
        --config configs/simulation/atomistic/al/potential_performance.yaml \
        > "$RUN_DIR/logs/performance_benchmark_gpu${DEVICE_ARRAY[0]}.log" 2>&1 &
    ACTIVE_PIDS=("$!")
    wait "${ACTIVE_PIDS[0]}"
    ACTIVE_PIDS=()
fi

if [ "$run_selection" = true ]; then
    STAGE=potential_selection
    status "RUNNING"
    "$PYTHON" -u -m src.data_utils.synthetic.atomistic_potential_selection \
        --config configs/simulation/atomistic/al/potential_selection.yaml \
        > "$RUN_DIR/logs/potential_selection.log" 2>&1
fi

selected_model=$("$PYTHON" -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["selected_model_name"])
' output/synthetic_data/al_potential_benchmark/selection.json)

STAGE=campaign
status "RUNNING selected_model=$selected_model through configured endpoint"
env PYTHON="$PYTHON" DEVICES="$DEVICES" \
    ./scripts/run_optimized_al_homogeneous_campaign.sh &
ACTIVE_PIDS=("$!")
wait "${ACTIVE_PIDS[0]}"
ACTIVE_PIDS=()
STAGE=finished
status "EXITED_CLEANLY selected_model=$selected_model; inspect campaign_status.json"
