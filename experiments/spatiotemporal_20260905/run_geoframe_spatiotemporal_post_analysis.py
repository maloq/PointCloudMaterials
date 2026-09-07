#!/usr/bin/env python3
"""Run the repository structural analysis for the saved Al/Mg/Ta GFv2 model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from datetime import datetime, timezone
import json
import subprocess
import traceback

import yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, default=Path(__file__).with_name("post_analysis.yaml"))
    args = parser.parse_args(argv)
    jobs = yaml.safe_load(args.recipe.read_text())["analysis_jobs"]
    status = {"state": "running", "materials": {}}

    def save():
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = args.root / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(args.root / "status.json")

    try:
        for job in jobs:
            material = job["name"]
            status["materials"][material] = "running"
            save()
            with (args.root / f"{material}.log").open("a") as log:
                subprocess.run([sys.executable, "-u", "-m", "src.analysis.pipeline",
                                str(args.root / job["config"])],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            json.loads((args.root / job["metrics"]).read_text())
            status["materials"][material] = "complete"
            save()
        status["state"] = "complete"
        save()
    except BaseException as error:
        status.update(state="failed", error=repr(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
