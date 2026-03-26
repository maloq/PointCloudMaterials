from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import TemporalBenchmarkConfig, dump_temporal_config
from .dynamics import LatentTrajectories, SiteLayout, write_transition_events_csv
from .graph import TransitionGraph
from .neighborhoods import NeighborhoodTrajectoryPack, neighborhood_manifest
from .rendering import RenderedFrame
from .validation import ValidationSummary


@dataclass(frozen=True)
class TemporalDatasetWriteResult:
    output_dir: Path
    frame_dirs: list[Path]
    frame_chunk_path: Path | None
    manifest_path: Path
    validation_summary_path: Path
    visualization_dir: Path | None = None
    visualization_manifest_path: Path | None = None


class TemporalDatasetWriter:
    def __init__(self, config: TemporalBenchmarkConfig, graph: TransitionGraph) -> None:
        self.config = config
        self.graph = graph
        self.output_dir = config.output.output_dir
        self.frame_storage = str(config.output.frame_storage)
        self.frames_dir = self.output_dir / "frames"
        self.frame_chunk_path = self.output_dir / "frames_chunk.npz"
        self.frame_metadata_path = self.output_dir / "frame_metadata.json"
        self.latent_dir = self.output_dir / "latent"
        self.neighborhoods_dir = self.output_dir / "neighborhoods"
        self.frame_dirs: list[Path] = []
        self._frame_atom_count: int | None = None
        self._chunk_write_count = 0
        self._chunk_atoms: np.ndarray | None = None
        self._chunk_site_ids: np.ndarray | None = None
        self._chunk_state_ids: np.ndarray | None = None
        self._chunk_grain_ids: np.ndarray | None = None
        self._chunk_local_atom_ids: np.ndarray | None = None
        self._chunk_metadata: list[dict[str, Any]] = []

    def prepare(self) -> None:
        if self.output_dir.exists():
            if not self.config.output.overwrite:
                raise FileExistsError(
                    f"Temporal dataset output directory already exists: {self.output_dir}. "
                    "Set output.overwrite=true or choose a different output path."
                )
            shutil.rmtree(self.output_dir)
        if self.frame_storage == "frame_dirs":
            self.frames_dir.mkdir(parents=True, exist_ok=False)
        elif self.frame_storage != "single_chunk_npz":
            raise ValueError(
                "output.frame_storage must be either 'frame_dirs' or 'single_chunk_npz', "
                f"got {self.frame_storage!r}."
            )
        self.latent_dir.mkdir(parents=True, exist_ok=False)
        self.neighborhoods_dir.mkdir(parents=True, exist_ok=False)

    def write_static_artifacts(self, layout: SiteLayout) -> None:
        dump_temporal_config(self.config, self.output_dir / "config_snapshot.yaml")
        with (self.output_dir / "transition_graph.json").open("w", encoding="utf-8") as handle:
            json.dump(self.graph.serializable(), handle, indent=2)
        np.savez_compressed(
            self.output_dir / "site_layout.npz",
            centers=layout.centers,
            neighbor_indices=layout.neighbor_indices,
            phase_centers=layout.phase_centers,
            phase_neighbor_indices=layout.phase_neighbor_indices,
            tracked_to_phase_index=layout.tracked_to_phase_index,
        )

    def write_frame(self, frame: RenderedFrame) -> None:
        if self.frame_storage == "single_chunk_npz":
            self._write_frame_single_chunk(frame)
            return
        frame_dir = self.frames_dir / f"frame_{frame.frame_index:05d}"
        frame_dir.mkdir(parents=False, exist_ok=False)
        np.save(frame_dir / "atoms.npy", frame.atoms)
        if self.config.rendering.save_atom_tables:
            saver = np.savez_compressed if self.config.output.compress else np.savez
            saver(
                frame_dir / "atom_table.npz",
                site_ids=frame.site_ids,
                state_ids=frame.state_ids,
                grain_ids=frame.grain_ids,
                local_atom_ids=frame.local_atom_ids,
            )
        with (frame_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(frame.metadata, handle, indent=2)
        self.frame_dirs.append(frame_dir)

    def _write_frame_single_chunk(self, frame: RenderedFrame) -> None:
        if self._frame_atom_count is None:
            self._initialize_chunk_storage(frame)
        if self._frame_atom_count != int(frame.atoms.shape[0]):
            raise RuntimeError(
                "single_chunk_npz frame storage requires a constant atom count across frames, "
                f"but frame 0 has {self._frame_atom_count} atoms and frame {frame.frame_index} has "
                f"{frame.atoms.shape[0]} atoms."
            )
        if self._chunk_atoms is None:
            raise RuntimeError("Chunk frame storage was not initialized before write_frame call.")
        frame_idx = int(frame.frame_index)
        self._chunk_atoms[frame_idx] = frame.atoms
        self._chunk_site_ids[frame_idx] = frame.site_ids
        self._chunk_state_ids[frame_idx] = frame.state_ids
        self._chunk_grain_ids[frame_idx] = frame.grain_ids
        self._chunk_local_atom_ids[frame_idx] = frame.local_atom_ids
        self._chunk_metadata.append(frame.metadata)
        self._chunk_write_count += 1

    def _initialize_chunk_storage(self, frame: RenderedFrame) -> None:
        num_frames = int(self.config.time.num_frames)
        atom_count = int(frame.atoms.shape[0])
        self._frame_atom_count = atom_count
        self._chunk_atoms = np.empty((num_frames, atom_count, 3), dtype=np.float32)
        self._chunk_site_ids = np.empty((num_frames, atom_count), dtype=np.int32)
        self._chunk_state_ids = np.empty((num_frames, atom_count), dtype=np.int16)
        self._chunk_grain_ids = np.empty((num_frames, atom_count), dtype=np.int32)
        self._chunk_local_atom_ids = np.empty((num_frames, atom_count), dtype=np.int32)
        self._chunk_metadata = []

    def finalize_frames(self) -> None:
        if self.frame_storage != "single_chunk_npz":
            return
        if self._chunk_atoms is None:
            raise RuntimeError(
                "single_chunk_npz frame storage was requested but no frames were written before finalize_frames."
            )
        if self._chunk_write_count != int(self.config.time.num_frames):
            raise RuntimeError(
                "single_chunk_npz frame storage is incomplete. "
                f"Wrote {self._chunk_write_count} frames but expected {self.config.time.num_frames}."
            )
        saver = np.savez_compressed if self.config.output.compress else np.savez
        saver(
            self.frame_chunk_path,
            atoms=self._chunk_atoms,
            site_ids=self._chunk_site_ids,
            state_ids=self._chunk_state_ids,
            grain_ids=self._chunk_grain_ids,
            local_atom_ids=self._chunk_local_atom_ids,
        )
        with self.frame_metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(self._chunk_metadata, handle, indent=2)
        self.frame_dirs = []

    def write_latent(self, latent: LatentTrajectories) -> None:
        saver = np.savez_compressed if self.config.output.compress else np.savez
        saver(
            self.latent_dir / "site_latent_trajectories.npz",
            state_ids=latent.state_ids,
            grain_ids=latent.grain_ids,
            crystal_variant_ids=latent.crystal_variant_ids,
            orientation_quaternions=latent.orientation_quaternions,
            strain=latent.strain,
            thermal_jitter=latent.thermal_jitter,
            defect_amplitude=latent.defect_amplitude,
            dwell_total=latent.dwell_total,
            dwell_remaining=latent.dwell_remaining,
            segment_ids=latent.segment_ids,
            transition_mask=latent.transition_mask,
            metastable_mask=latent.metastable_mask,
            seed_site_mask=latent.seed_site_mask,
        )
        saver(
            self.latent_dir / "phase_latent_trajectories.npz",
            state_ids=latent.phase_state_ids,
            grain_ids=latent.phase_grain_ids,
            crystal_variant_ids=latent.phase_crystal_variant_ids,
            orientation_quaternions=latent.phase_orientation_quaternions,
            strain=latent.phase_strain,
            thermal_jitter=latent.phase_thermal_jitter,
            defect_amplitude=latent.phase_defect_amplitude,
            seed_site_mask=latent.phase_seed_site_mask,
        )
        grain_orientation_records = {
            str(grain_id): quat.astype(float).tolist()
            for grain_id, quat in sorted(latent.grain_orientations.items(), key=lambda item: item[0])
        }
        with (self.latent_dir / "grain_orientations.json").open("w", encoding="utf-8") as handle:
            json.dump(grain_orientation_records, handle, indent=2)
        write_transition_events_csv(latent.transition_events, self.latent_dir / "transition_events.csv")

    def write_neighborhoods(self, pack: NeighborhoodTrajectoryPack) -> None:
        saver = np.savez_compressed if self.config.output.compress else np.savez
        saver(
            self.neighborhoods_dir / "trajectory_pack.npz",
            points=pack.points,
            state_ids=pack.state_ids,
            grain_ids=pack.grain_ids,
            transition_mask=pack.transition_mask,
            metastable_mask=pack.metastable_mask,
            site_centers=pack.site_centers,
            frame_times=pack.frame_times,
        )
        with (self.neighborhoods_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(neighborhood_manifest(pack, self.graph), handle, indent=2)

    def write_validation(self, validation: ValidationSummary) -> Path:
        summary_path = self.output_dir / "validation_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(validation.summary, handle, indent=2)
        return summary_path

    def write_manifest(
        self,
        layout: SiteLayout,
        latent: LatentTrajectories,
        validation: ValidationSummary,
    ) -> Path:
        manifest = {
            "dataset_name": self.config.dataset_name,
            "output_dir": str(self.output_dir),
            "seed": int(self.config.seed),
            "num_frames": int(self.config.time.num_frames),
            "site_count": int(layout.site_count),
            "phase_site_count": int(layout.phase_site_count),
            "points_per_site": int(self.config.domain.atoms_per_site),
            "frame_storage": self.frame_storage,
            "frame_dirs": [str(path) for path in self.frame_dirs],
            "frame_chunk_path": str(self.frame_chunk_path) if self.frame_storage == "single_chunk_npz" else None,
            "frame_metadata_path": (
                str(self.frame_metadata_path) if self.frame_storage == "single_chunk_npz" else None
            ),
            "site_latent_path": str(self.latent_dir / "site_latent_trajectories.npz"),
            "phase_latent_path": str(self.latent_dir / "phase_latent_trajectories.npz"),
            "state_names": self.graph.state_names,
            "transition_event_count": int(len(latent.transition_events)),
            "validation_summary": validation.summary,
        }
        manifest_path = self.output_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        return manifest_path


def result_from_paths(
    output_dir: Path,
    frame_dirs: list[Path],
    frame_chunk_path: Path | None,
    manifest_path: Path,
    validation_summary_path: Path,
    visualization_dir: Path | None = None,
    visualization_manifest_path: Path | None = None,
) -> TemporalDatasetWriteResult:
    return TemporalDatasetWriteResult(
        output_dir=output_dir,
        frame_dirs=frame_dirs,
        frame_chunk_path=frame_chunk_path,
        manifest_path=manifest_path,
        validation_summary_path=validation_summary_path,
        visualization_dir=visualization_dir,
        visualization_manifest_path=visualization_manifest_path,
    )
