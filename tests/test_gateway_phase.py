from pathlib import Path

import numpy as np
import torch

from src.analysis.gateway_phase import (
    GatewayPhaseSettings,
    _extract_reactive_episodes,
    run_gateway_phase_analysis,
)


def _settings(*, cna_enabled: bool) -> GatewayPhaseSettings:
    return GatewayPhaseSettings(
        enabled=True,
        state_a=(0, 1),
        intermediate=(2,),
        state_b=(3,),
        bootstrap_samples=100,
        random_state=12,
        gateway_fraction_threshold=0.8,
        minimum_reactive_paths=2,
        lag_strides=(1, 2, 3),
        reactive_path_csv_limit=1000,
        spatial_enabled=True,
        spatial_neighbor_k=4,
        spatial_max_frames=12,
        precursor_lags=(1, 2),
        precursor_radius_edges=(0.0, 2.5, 5.0),
        precursor_max_events=100,
        cna_enabled=cna_enabled,
        cna_max_samples=90,
        cna_max_signatures=6,
        cna_fit_max_pointclouds=90,
        shell_min_neighbors=4,
        shell_max_neighbors=10,
    )


def test_reactive_episode_detection_distinguishes_gateway_and_bypass() -> None:
    states = np.asarray(
        [
            [0, 0],
            [0, 0],
            [1, 2],
            [1, 2],
            [2, 2],
        ],
        dtype=np.int8,
    )
    episodes = _extract_reactive_episodes(
        states,
        np.asarray([101, 202]),
        source_state=0,
        target_state=2,
    )
    assert len(episodes) == 2
    by_atom = {row["atom_id"]: row for row in episodes}
    assert by_atom[101]["visited_intermediate"] is True
    assert by_atom[101]["intermediate_residence_frames"] == 2
    assert by_atom[202]["visited_intermediate"] is False


def test_gateway_phase_report_uses_tracked_atoms_and_cna(tmp_path: Path) -> None:
    frame_count = 12
    atom_count = 30
    anchor_frames = np.repeat(np.arange(frame_count, dtype=np.int64), atom_count)
    atom_ids = np.tile(np.arange(1000, 1000 + atom_count, dtype=np.int64), frame_count)
    labels = np.empty((frame_count, atom_count), dtype=np.int32)
    for frame in range(frame_count):
        for atom in range(atom_count):
            phase = (frame + atom % 6) % 6
            if atom < 24:
                macro = 0 if phase < 2 else (1 if phase < 4 else 2)
            else:
                macro = 0 if phase < 2 else 2
            labels[frame, atom] = (atom % 2) if macro == 0 else (2 if macro == 1 else 3)

    base = np.column_stack(
        [
            np.linspace(1.0, 18.0, atom_count),
            np.mod(np.arange(atom_count) * 3.1, 18.0) + 1.0,
            np.mod(np.arange(atom_count) * 5.3, 18.0) + 1.0,
        ]
    ).astype(np.float32)
    coords = np.vstack(
        [np.mod(base + np.asarray([0.08 * frame, 0.0, 0.0]), 20.0) for frame in range(frame_count)]
    )

    class PointCloudDataset:
        def __len__(self) -> int:
            return frame_count * atom_count

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            rng = np.random.default_rng(int(index))
            vectors = rng.normal(size=(31, 3))
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            macro = 0 if labels.reshape(-1)[index] in {0, 1} else (1 if labels.reshape(-1)[index] == 2 else 2)
            vectors[:, macro] *= 1.0 + 0.18 * macro
            points = np.vstack([np.zeros((1, 3)), vectors]).astype(np.float32)
            return {"points": torch.from_numpy(points[None, ...])}

    class TemporalMetadata:
        box_lengths = np.full((frame_count, 3), 20.0, dtype=np.float32)

    summary = run_gateway_phase_analysis(
        labels=labels.reshape(-1),
        instance_ids=atom_ids,
        coords=coords,
        anchor_frame_indices=anchor_frames,
        out_dir=tmp_path,
        settings=_settings(cna_enabled=True),
        inference_dataset=PointCloudDataset(),
        temporal_dataset=TemporalMetadata(),
    )

    assert summary["tracked_frame_count"] == frame_count
    assert summary["tracked_atom_count"] == atom_count
    assert summary["forward_gateway"]["reactive_path_count"] > 0
    assert summary["forward_gateway"]["paths_via_intermediate"] > 0
    assert summary["forward_gateway"]["bypass_path_count"] > 0
    assert summary["empirical_intermediate_committor"]["entry_count_with_resolved_next_hit"] > 0
    assert summary["cna_phase_evidence"]["enabled"] is True
    assert len(summary["cna_phase_evidence"]["signature_names"]) > 1
    for artifact in summary["artifacts"].values():
        assert Path(artifact).is_file()
    assert Path(summary["cna_phase_evidence"]["artifacts"]["figure"]).is_file()
    report = Path(summary["artifacts"]["report"]).read_text(encoding="utf-8")
    assert "Gateway-intermediate and CNA phase validation" in report
    assert "not a shooting committor" not in report
