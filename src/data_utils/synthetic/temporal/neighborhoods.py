from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TemporalBenchmarkConfig
from .dynamics import LatentTrajectories, SiteLayout
from .graph import TransitionGraph


@dataclass(frozen=True)
class NeighborhoodTrajectoryPack:
    points: np.ndarray
    state_ids: np.ndarray
    grain_ids: np.ndarray
    transition_mask: np.ndarray
    metastable_mask: np.ndarray
    site_centers: np.ndarray
    frame_times: np.ndarray


def build_neighborhood_trajectory_pack(
    config: TemporalBenchmarkConfig,
    graph: TransitionGraph,
    layout: SiteLayout,
    latent: LatentTrajectories,
    local_points_by_frame: np.ndarray,
) -> NeighborhoodTrajectoryPack:
    if local_points_by_frame.shape[:2] != latent.state_ids.shape:
        raise ValueError(
            "Neighborhood pack build requires local_points_by_frame.shape[:2] to match latent.state_ids.shape, "
            f"got local_points_by_frame.shape={local_points_by_frame.shape}, latent.state_ids.shape={latent.state_ids.shape}."
        )
    points = local_points_by_frame
    if config.trajectories.center_neighborhoods:
        points = points.copy()
    return NeighborhoodTrajectoryPack(
        points=points.astype(np.float32),
        state_ids=latent.state_ids.astype(np.int16),
        grain_ids=latent.grain_ids.astype(np.int32),
        transition_mask=latent.transition_mask.astype(bool),
        metastable_mask=latent.metastable_mask.astype(bool),
        site_centers=layout.centers.astype(np.float32),
        frame_times=np.arange(config.time.num_frames, dtype=np.float32) * float(config.time.delta_t),
    )


def neighborhood_manifest(
    pack: NeighborhoodTrajectoryPack,
    graph: TransitionGraph,
) -> dict[str, object]:
    return {
        "shape": {
            "num_frames": int(pack.points.shape[0]),
            "num_sites": int(pack.points.shape[1]),
            "points_per_site": int(pack.points.shape[2]),
        },
        "state_names": graph.state_names,
    }
