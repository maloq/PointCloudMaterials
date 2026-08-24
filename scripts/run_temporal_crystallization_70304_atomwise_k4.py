#!/usr/bin/env python3
"""Run atom-centered K=4 temporal analysis on one completed 70k replica."""

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
            "source12345_seed35803_atomwise_k4_multiscale_vicreg"
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
            "source12346_seed35831_atomwise_k4_multiscale_vicreg"
        ),
    },
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica", choices=tuple(ANALYSES), required=True)
    parser.add_argument("--inference-batch-size", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.inference_batch_size <= 0:
        raise ValueError(
            "--inference-batch-size must be positive, "
            f"got {args.inference_batch_size}."
        )
    paths = ANALYSES[args.replica]
    replica_root = paths["replica_root"]
    dump_file = (
        replica_root / "analysis_inputs/temporal/measurement_trajectory.lammpstrj"
    )
    if not dump_file.is_file():
        raise FileNotFoundError(
            f"{args.replica}: temporal dump is missing: {dump_file}. Run "
            "scripts/export_homogeneous_trajectory_lammps.py first."
        )

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_cfg = load_checkpoint_analysis_config(str(REFERENCE_CONFIG))
    with open_dict(analysis_cfg):
        analysis_cfg.checkpoint.output_dir = str(output_dir)
        analysis_cfg.inputs.dataloader_num_workers = 8
        analysis_cfg.inputs.inference_batch_size = int(args.inference_batch_size)
        analysis_cfg.inputs.temporal_real.dump_file = str(dump_file)
        analysis_cfg.inputs.temporal_real.cache_dir = str(
            replica_root / "analysis_inputs/temporal/cache"
        )
        analysis_cfg.inputs.temporal_real.center_selection = OmegaConf.create(
            {
                "mode": "atom_stride",
                "atom_stride": 1,
            }
        )
        analysis_cfg.inputs.temporal_real.snapshot_visualization.enabled = False

        analysis_cfg.clustering.primary_k = 4
        analysis_cfg.clustering.k_values = None
        analysis_cfg.figure_set.visible_cluster_sets = [[0, 1, 2, 3]]

        # The reference A/intermediate/B state prior names K=7 cluster IDs.
        # It cannot be transferred to a separately fitted K=4 model.
        analysis_cfg.gateway_phase.enabled = False
        analysis_cfg.real_md.cluster_group_order = None
        analysis_cfg.real_md.cluster_groups = None

        # Atom-centered main inference is already denser than the auxiliary grid.
        analysis_cfg.real_md.temporal.md_space.reuse_main_inference_cache = True

    resolved_config_path = output_dir / "resolved_analysis_config.yaml"
    OmegaConf.save(analysis_cfg, resolved_config_path)
    print(
        f"{args.replica}: running atom-centered inference for all 70,304 atoms "
        f"at every frame with K=4. Resolved config: {resolved_config_path}",
        flush=True,
    )
    run_post_training_analysis(analysis_cfg=analysis_cfg)


if __name__ == "__main__":
    main()
