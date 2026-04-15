from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time

from omegaconf import OmegaConf

from src.training_methods.vamp.common import log_progress
from src.training_methods.vamp.config import load_vamp_config


def _run_command(command: list[str]) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    start = time.perf_counter()
    log_progress("vamp.run_pipeline", f"starting: {printable}")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"Pipeline command failed with exit code {completed.returncode}: {printable}"
        )
    elapsed = time.perf_counter() - start
    log_progress("vamp.run_pipeline", f"finished in {elapsed:.1f}s: {printable}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen-encoder VAMP pipeline from a VAMP config name or YAML path. "
            "This is the Phase-1 embedding + linear VAMP baseline, not end-to-end VAMPnet training."
        )
    )
    parser.add_argument(
        "config",
        help="Config name inside configs/vamp/ or a YAML path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, config_path, _ = load_vamp_config(args.config)
    config_arg = str(config_path)

    if bool(OmegaConf.select(cfg, "pipeline.run_embed", default=True)):
        _run_command([sys.executable, "-m", "src.training_methods.vamp.embed_trajectory", config_arg])
    if bool(OmegaConf.select(cfg, "pipeline.run_fit", default=True)):
        _run_command([sys.executable, "-m", "src.training_methods.vamp.fit_vamp", config_arg])
    if bool(OmegaConf.select(cfg, "pipeline.run_analyze", default=True)):
        _run_command([sys.executable, "-m", "src.training_methods.vamp.analyze_vamp", config_arg])
    if bool(OmegaConf.select(cfg, "pipeline.run_verify", default=False)):
        _run_command([sys.executable, "-m", "src.training_methods.vamp.verify_against_deeptime", config_arg])


if __name__ == "__main__":
    main()
