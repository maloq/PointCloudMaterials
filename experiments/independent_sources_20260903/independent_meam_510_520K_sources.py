#!/usr/bin/env python3
"""Generate independent 510/520 K histories for predictive-parent selection.

This is a parameter-locked specialization of the validated independent-source
producer. The broad 1--99 atom candidate interval is diagnostic only; basin A
and production parent interfaces are calibrated from the completed histories.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.simulation.campaigns import independent_meam_source as source


CAMPAIGN_ROOT = Path(
    "/home/ids/vmorozov/simulations/"
    "al_meam_independent_sources_70304_510-520K_30perT_float16_20260903"
)


def _configure() -> None:
    source.CAMPAIGN_SEED = 20_260_903
    source.TEMPERATURES_K = (510.0, 520.0)
    source.RUNS_PER_TEMPERATURE = 30
    source.SPLIT_COUNTS_PER_TEMPERATURE = {
        "optimization": 18,
        "model_selection": 6,
        "final_validation": 6,
    }
    source.BOUNDARY_BANDS = {510.0: (1, 99), 520.0: (1, 99)}
    source.ARRAY_CONCURRENCY = 2


def prepare(root: Path) -> dict[str, object]:
    _configure()
    manifest = source.prepare_campaign(root)
    manifest["candidate_interval_status"] = (
        "diagnostic broad non-basin interval only; temperature-specific basin-A and "
        "transition interfaces must be calibrated before parent selection"
    )
    manifest["production_parent_selection_from_future_outcomes"] = False
    source._write_json_atomic(root / "manifest.json", manifest)
    (root / "manifest.sha256").write_text(
        source._sha256_file(root / "manifest.json") + "  manifest.json\n", encoding="ascii"
    )
    original_runner = Path(source.__file__).resolve()
    this_runner = Path(__file__).resolve()
    for path in (
        root / "slurm/run_source.sbatch",
        root / "slurm/submit_wave.sbatch",
        root / "slurm/summarize.sbatch",
    ):
        text = path.read_text(encoding="utf-8")
        if str(original_runner) not in text:
            raise RuntimeError(f"Generated source Slurm script does not reference {original_runner}: {path}.")
        text = text.replace(str(original_runner), str(this_runner))
        text = text.replace("--job-name=al_ind_source", "--job-name=al_ind_hiT")
        text = text.replace("--job-name=al_ind_src_ctl", "--job-name=al_ind_hiT_ctl")
        text = text.replace("--job-name=al_ind_src_sum", "--job-name=al_ind_hiT_sum")
        path.write_text(text, encoding="utf-8")
    return manifest


def _verify_manifest(root: Path) -> None:
    checksum_path = root / "manifest.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Immutable source-manifest checksum is missing: {checksum_path}.")
    expected = checksum_path.read_text(encoding="ascii").split()[0]
    observed = source._sha256_file(root / "manifest.json")
    if observed != expected:
        raise RuntimeError(
            f"Immutable high-temperature source manifest changed: expected={expected}, "
            f"observed={observed}, path={root / 'manifest.json'}."
        )


def _arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--campaign-root", type=Path, default=CAMPAIGN_ROOT)
    run = subparsers.add_parser("run-task")
    run.add_argument("--campaign-root", type=Path, required=True)
    run.add_argument("--task-index", type=int, required=True)
    local_run = subparsers.add_parser("run-local-task")
    local_run.add_argument("--campaign-root", type=Path, required=True)
    local_run.add_argument("--task-index", type=int, required=True)
    submit = subparsers.add_parser("submit-next-wave")
    submit.add_argument("--campaign-root", type=Path, required=True)
    submit.add_argument("--start-index", type=int, required=True)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--campaign-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _arguments(argv)
    _configure()
    if args.action == "prepare":
        manifest = prepare(args.campaign_root.expanduser().resolve())
        print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    elif args.action == "run-task":
        _verify_manifest(args.campaign_root)
        source.run_source_task(args.campaign_root, args.task_index)
    elif args.action == "run-local-task":
        _verify_manifest(args.campaign_root)
        source.run_source_task(
            args.campaign_root,
            args.task_index,
            launcher="local_mpiexec",
        )
    elif args.action == "submit-next-wave":
        _verify_manifest(args.campaign_root)
        source.submit_next_wave(args.campaign_root, args.start_index)
    elif args.action == "summarize":
        _verify_manifest(args.campaign_root)
        source.summarize_campaign(args.campaign_root)
    else:
        raise AssertionError(f"Unhandled action: {args.action}.")


if __name__ == "__main__":
    main()
