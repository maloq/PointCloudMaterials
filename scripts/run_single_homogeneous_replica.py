#!/usr/bin/env python3
"""Run exactly one queued homogeneous replica through its natural endpoint."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_utils.synthetic.atomistic.generator import select_calculator  # noqa: E402
from src.data_utils.synthetic.atomistic.homogeneous_campaign import (  # noqa: E402
    run_campaign_replica,
)
from src.data_utils.synthetic.atomistic.homogeneous_campaign_config import (  # noqa: E402
    load_homogeneous_campaign_config,
)
from src.data_utils.synthetic.atomistic.homogeneous_campaign_queue import (  # noqa: E402
    claim_md_task,
    complete_md_task,
    fail_md_task,
)
from src.data_utils.synthetic.atomistic.homogeneous_generator import (  # noqa: E402
    _load_source_liquid,
)
from src.data_utils.synthetic.atomistic.homogeneous_online import (  # noqa: E402
    OnlineCrystallinityDetector,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume and complete exactly one queued homogeneous-crystallization replica, "
            "then exit without claiming another seed."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--replica-name", required=True)
    parser.add_argument("--worker-name", default="single_replica_completion")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = load_homogeneous_campaign_config(args.config)
    source = _load_source_liquid(config.homogeneous)
    calculator, execution_provenance = select_calculator(
        config.homogeneous.generator,
        calculator=None,
        injected_calculator_identity=None,
    )
    detector = OnlineCrystallinityDetector(
        ptm_rmsd_cutoff=config.homogeneous.analysis.ptm_rmsd_cutoff,
        crystalline_cluster_cutoff_A=(
            config.homogeneous.analysis.crystalline_cluster_cutoff_A
        ),
    )
    task = claim_md_task(config, worker_name=args.worker_name)
    if task is None:
        raise RuntimeError(
            f"No queued replica exists in {config.output_root / 'campaign.sqlite3'}."
        )
    if task.replica_name != args.replica_name:
        fail_md_task(
            config,
            task=task,
            error=(
                f"Single-replica completion expected {args.replica_name!r}, but the queue "
                f"returned {task.replica_name!r}."
            ),
        )
        raise RuntimeError(
            f"Expected to claim {args.replica_name!r}, got {task.replica_name!r}. "
            "The unexpected task was marked failed rather than run ambiguously."
        )
    print(
        f"{args.worker_name}: completing only {task.replica_name}; "
        "the process will exit at its natural endpoint.",
        flush=True,
    )
    try:
        result = run_campaign_replica(
            config,
            task=task,
            source_atoms=source.atoms,
            calculator=calculator,
            execution_provenance=execution_provenance,
            detector=detector,
            progress=lambda message: print(message, flush=True),
        )
        complete_md_task(
            config,
            task=task,
            outcome=result.outcome,
            raw_directory=result.raw_directory,
            run_metadata_sha256=result.run_metadata_sha256,
            online_threshold_event=result.online_threshold_event,
        )
    except BaseException:
        error = traceback.format_exc()
        fail_md_task(config, task=task, error=error)
        raise
    print(
        f"{args.worker_name}: completed {task.replica_name} with "
        f"outcome={result.outcome}; exiting without claiming another replica.",
        flush=True,
    )


if __name__ == "__main__":
    main()
