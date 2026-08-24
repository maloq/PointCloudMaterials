#!/usr/bin/env python3
"""Run the step-187800 temporal analysis recipe on both completed 70k replicas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf, open_dict


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.analysis.config import load_checkpoint_analysis_config  # noqa: E402
from src.analysis.pipeline import run_post_training_analysis  # noqa: E402


REFERENCE_CONFIG = (
    REPOSITORY_ROOT / "configs/analysis/temporal_crystallization_step187800.yaml"
)
ANALYSES = {
    "source12345_seed35803": {
        "replica_root": (
            REPOSITORY_ROOT
            / "output/synthetic_data/"
            "al_homogeneous_campaign_70304_mpa_130ps_"
            "source12345_seed35803_20260720/replicas/replica_000"
        ),
        "output_dir": (
            REPOSITORY_ROOT
            / "outputs/temporal_crystallization_70304_mpa_130ps_"
            "source12345_seed35803_multiscale_vicreg"
        ),
    },
    "source12346_seed35831": {
        "replica_root": (
            REPOSITORY_ROOT
            / "output/synthetic_data/"
            "al_homogeneous_campaign_70304_mpa_130ps_"
            "source12346_seed35831_20260720/replicas/replica_000"
        ),
        "output_dir": (
            REPOSITORY_ROOT
            / "outputs/temporal_crystallization_70304_mpa_130ps_"
            "source12346_seed35831_multiscale_vicreg"
        ),
    },
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replica",
        choices=("both", *ANALYSES),
        default="both",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    selected = ANALYSES if args.replica == "both" else {args.replica: ANALYSES[args.replica]}
    for name, paths in selected.items():
        replica_root = paths["replica_root"]
        dump_file = (
            replica_root
            / "analysis_inputs/temporal/measurement_trajectory.lammpstrj"
        )
        if not dump_file.is_file():
            raise FileNotFoundError(
                f"{name}: temporal dump is missing: {dump_file}. Run "
                "scripts/export_homogeneous_trajectory_lammps.py first."
            )
        output_dir = paths["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        analysis_cfg = load_checkpoint_analysis_config(str(REFERENCE_CONFIG))
        with open_dict(analysis_cfg):
            analysis_cfg.checkpoint.output_dir = str(output_dir)
            analysis_cfg.inputs.temporal_real.dump_file = str(dump_file)
            analysis_cfg.inputs.temporal_real.cache_dir = str(
                replica_root / "analysis_inputs/temporal/cache"
            )
        resolved_config_path = output_dir / "resolved_analysis_config.yaml"
        OmegaConf.save(analysis_cfg, resolved_config_path)
        print(
            f"{name}: running {REFERENCE_CONFIG} recipe with resolved config "
            f"{resolved_config_path}",
            flush=True,
        )
        run_post_training_analysis(analysis_cfg=analysis_cfg)


if __name__ == "__main__":
    main()
