from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .config import TemporalBenchmarkConfig
from .dynamics import LatentTrajectories, SiteLayout
from .graph import TransitionGraph


@dataclass(frozen=True)
class ValidationSummary:
    summary: dict[str, object]


def validate_temporal_dataset(
    config: TemporalBenchmarkConfig,
    graph: TransitionGraph,
    layout: SiteLayout,
    latent: LatentTrajectories,
    local_points_by_frame: np.ndarray,
    frame_atom_counts: list[int],
) -> ValidationSummary:
    expected_shape = (config.time.num_frames, config.domain.site_count)
    if latent.state_ids.shape != expected_shape:
        raise ValueError(
            f"latent.state_ids must have shape {expected_shape}, got {latent.state_ids.shape}."
        )
    if local_points_by_frame.shape[:2] != expected_shape:
        raise ValueError(
            f"local_points_by_frame shape prefix must be {expected_shape}, got {local_points_by_frame.shape}."
        )
    if not frame_atom_counts:
        raise ValueError("frame_atom_counts is empty; validation requires per-frame atom counts.")
    if any(count <= 0 for count in frame_atom_counts):
        raise ValueError(
            f"Frame atom counts must all be positive, got {frame_atom_counts}."
        )
    for array_name, array_value in {
        "state_ids": latent.state_ids,
        "grain_ids": latent.grain_ids,
        "crystal_variant_ids": latent.crystal_variant_ids,
        "orientation_quaternions": latent.orientation_quaternions,
        "strain": latent.strain,
        "thermal_jitter": latent.thermal_jitter,
        "defect_amplitude": latent.defect_amplitude,
        "local_points_by_frame": local_points_by_frame,
    }.items():
        if np.issubdtype(array_value.dtype, np.floating) and np.isnan(array_value).any():
            raise ValueError(f"NaN values detected in {array_name}.")

    transition_counts: Counter[str] = Counter()
    dwell_by_state: defaultdict[str, list[int]] = defaultdict(list)

    for site_id in range(config.domain.site_count):
        segment_start = 0
        for frame_idx in range(1, config.time.num_frames):
            prev_state = int(latent.state_ids[frame_idx - 1, site_id])
            curr_state = int(latent.state_ids[frame_idx, site_id])
            if not graph.is_valid_transition(prev_state, curr_state):
                raise ValueError(
                    f"Invalid state transition at site_id={site_id}, frame_index={frame_idx}: "
                    f"{graph.name(prev_state)} -> {graph.name(curr_state)} is not in the transition graph."
                )
            if curr_state != prev_state:
                transition_counts[f"{graph.name(prev_state)}->{graph.name(curr_state)}"] += 1
                dwell_by_state[graph.name(prev_state)].append(frame_idx - segment_start)
                segment_start = frame_idx
        final_state = graph.name(int(latent.state_ids[-1, site_id]))
        dwell_by_state[final_state].append(config.time.num_frames - segment_start)

    state_counts = Counter(graph.name(int(state_idx)) for state_idx in latent.state_ids.reshape(-1))
    mean_dwell_by_state = {
        state_name: float(np.mean(durations))
        for state_name, durations in dwell_by_state.items()
        if durations
    }
    grain_ids = latent.grain_ids[latent.grain_ids >= 0]
    summary = {
        "dataset_name": config.dataset_name,
        "num_frames": int(config.time.num_frames),
        "site_count": int(layout.site_count),
        "points_per_site": int(config.domain.atoms_per_site),
        "frame_atom_count": int(frame_atom_counts[0]),
        "frame_atom_count_min": int(np.min(frame_atom_counts)),
        "frame_atom_count_max": int(np.max(frame_atom_counts)),
        "frame_atom_count_mean": float(np.mean(frame_atom_counts)),
        "num_grains": int(len(np.unique(grain_ids))) if grain_ids.size > 0 else 0,
        "transition_event_count": int(sum(transition_counts.values())),
        "state_frame_counts": dict(state_counts),
        "transition_counts": dict(transition_counts),
        "crystal_variant_fraction": float(np.mean(latent.crystal_variant_ids == 1)),
        "mean_dwell_by_state": mean_dwell_by_state,
        "metastable_fraction": float(np.mean(latent.metastable_mask)),
        "seed_site_fraction": float(np.mean(latent.seed_site_mask)),
    }
    return ValidationSummary(summary=summary)
