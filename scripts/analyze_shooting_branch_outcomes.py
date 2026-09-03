#!/usr/bin/env python3
"""Classify finite-horizon crystallization outcomes of complete shooting branches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_utils.shooting_dataset import load_shooting_campaigns_snapshot
from src.temporal_vamp.shooting_outcomes import (
    analyze_shooting_endpoint_outcomes,
    plot_shooting_endpoint_outcomes,
)


def _required(cfg: Any, path: str) -> Any:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise KeyError(f"Shooting outcome analysis requires configuration key {path!r}.")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(_resolve_path(args.config))
    OmegaConf.resolve(cfg)
    snapshot = load_shooting_campaigns_snapshot(
        [_resolve_path(value) for value in _required(cfg, "data.campaign_roots")],
        temperatures_K=[float(value) for value in _required(cfg, "data.temperatures_K")],
        minimum_complete_branches_per_parent=int(
            _required(cfg, "data.minimum_complete_branches_per_parent")
        ),
    )
    outcomes = analyze_shooting_endpoint_outcomes(
        snapshot,
        output_path=_resolve_path(_required(cfg, "outcomes.path")),
        persistence_frames=int(_required(cfg, "outcomes.persistence_frames")),
        ptm_rmsd_cutoff=float(_required(cfg, "outcomes.ptm_rmsd_cutoff")),
        crystal_cluster_threshold_atoms=int(
            _required(cfg, "outcomes.crystal_cluster_threshold_atoms")
        ),
        maximum_liquid_crystalline_fraction=float(
            _required(cfg, "outcomes.maximum_liquid_crystalline_fraction")
        ),
        workers=int(_required(cfg, "outcomes.workers")),
    )
    plot_shooting_endpoint_outcomes(
        outcomes, outcomes.path / "endpoint_outcomes.png"
    )
    print(
        "[shooting-outcomes] complete "
        f"counts={outcomes.document['counts']} path={outcomes.path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
