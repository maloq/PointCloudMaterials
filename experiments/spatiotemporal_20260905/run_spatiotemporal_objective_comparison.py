#!/usr/bin/env python3
"""Run corrected VICReg, VISReg, and their matched evaluation sequentially."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

REPOSITORY = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--static-cache", type=Path, required=True)
    parser.add_argument("--config-tag", default="corrected_20260905")
    args = parser.parse_args(argv)

    def status(state, stage, **extra):
        payload = dict(state=state, stage=stage, pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        temporary = args.root / "queue_status.tmp"
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(args.root / "queue_status.json")

    stage = "initializing"
    try:
        jobs = [(objective, [sys.executable, "-u", "src/training_methods/spatiotemporal.py",
                 "--run-dir", str(args.root / objective), "--config-name", f"{objective}_geoframe_v2_spatiotemporal_{args.config_tag}"])
                for objective in ("vicreg", "visreg")]
        jobs.append(("comparison", [sys.executable, "-u", "experiments/spatiotemporal_20260905/compare_spatiotemporal_objectives.py", "--root", str(args.root), "--static-cache", str(args.static_cache)]))
        for index, (stage, command) in enumerate(jobs):
            print(f"Starting {stage}: {command}", flush=True)
            with (args.root / f"{stage}.log").open("w") as log:
                process = subprocess.Popen(command, cwd=REPOSITORY, stdout=log, stderr=subprocess.STDOUT)
                status("running", stage, child_pid=process.pid, pending=[job[0] for job in jobs[index + 1:]])
                result = process.wait()
            if result:
                raise RuntimeError(f"{stage} exited with code {result}; see {args.root / (stage + '.log')}")
        status("complete", "complete", report=str(args.root / "comparison/RESULTS.md"))
    except BaseException as error:
        status("failed", stage, error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
