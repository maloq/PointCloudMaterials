from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.training_methods.vamp.common import TrajectoryEmbeddings, save_json


@dataclass(frozen=True)
class SuccessorEmbeddingsArtifact:
    invariant_embeddings: np.ndarray
    successor_embeddings: np.ndarray
    center_positions: np.ndarray
    atom_ids: np.ndarray
    frame_indices: np.ndarray
    timesteps: np.ndarray
    metadata: dict[str, Any]

    @property
    def frame_count(self) -> int:
        return int(self.invariant_embeddings.shape[0])

    @property
    def num_atoms(self) -> int:
        return int(self.invariant_embeddings.shape[1])

    @property
    def invariant_dim(self) -> int:
        return int(self.invariant_embeddings.shape[2])

    @property
    def successor_dim(self) -> int:
        return int(self.successor_embeddings.shape[2])

    def to_trajectory_embeddings(self) -> TrajectoryEmbeddings:
        return TrajectoryEmbeddings(
            invariant_embeddings=np.asarray(self.invariant_embeddings, dtype=np.float32),
            center_positions=np.asarray(self.center_positions, dtype=np.float32),
            atom_ids=np.asarray(self.atom_ids, dtype=np.int64),
            frame_indices=np.asarray(self.frame_indices, dtype=np.int64),
            timesteps=np.asarray(self.timesteps, dtype=np.int64),
            metadata=dict(self.metadata),
        )

    def save(self, path: str | Path) -> Path:
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            resolved,
            invariant_embeddings=self.invariant_embeddings.astype(np.float32, copy=False),
            successor_embeddings=self.successor_embeddings.astype(np.float32, copy=False),
            center_positions=self.center_positions.astype(np.float32, copy=False),
            atom_ids=self.atom_ids.astype(np.int64, copy=False),
            frame_indices=self.frame_indices.astype(np.int64, copy=False),
            timesteps=self.timesteps.astype(np.int64, copy=False),
        )
        meta_payload = dict(self.metadata)
        meta_payload.update(
            {
                "artifact_version": 1,
                "npz_path": str(resolved),
                "frame_count": int(self.frame_count),
                "num_atoms": int(self.num_atoms),
                "invariant_dim": int(self.invariant_dim),
                "successor_dim": int(self.successor_dim),
            }
        )
        save_json(meta_payload, resolved.with_suffix(resolved.suffix + ".meta.json"))
        return resolved

    @classmethod
    def load(cls, path: str | Path) -> "SuccessorEmbeddingsArtifact":
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Successor embedding artifact does not exist: {resolved}")
        meta_path = resolved.with_suffix(resolved.suffix + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Successor embedding artifact metadata does not exist: {meta_path}"
            )
        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        with np.load(resolved) as payload:
            invariant_embeddings = np.asarray(payload["invariant_embeddings"], dtype=np.float32)
            successor_embeddings = np.asarray(payload["successor_embeddings"], dtype=np.float32)
            center_positions = np.asarray(payload["center_positions"], dtype=np.float32)
            atom_ids = np.asarray(payload["atom_ids"], dtype=np.int64)
            frame_indices = np.asarray(payload["frame_indices"], dtype=np.int64)
            timesteps = np.asarray(payload["timesteps"], dtype=np.int64)

        if invariant_embeddings.ndim != 3:
            raise ValueError(
                "Expected invariant_embeddings with shape (frames, atoms, dim), "
                f"got {tuple(invariant_embeddings.shape)}."
            )
        if successor_embeddings.ndim != 3:
            raise ValueError(
                "Expected successor_embeddings with shape (frames, atoms, dim), "
                f"got {tuple(successor_embeddings.shape)}."
            )
        if successor_embeddings.shape[:2] != invariant_embeddings.shape[:2]:
            raise ValueError(
                "Successor embedding shape mismatch. "
                f"invariant_embeddings.shape={tuple(invariant_embeddings.shape)}, "
                f"successor_embeddings.shape={tuple(successor_embeddings.shape)}."
            )
        expected_coord_shape = invariant_embeddings.shape[:2] + (3,)
        if center_positions.shape != expected_coord_shape:
            raise ValueError(
                "center_positions shape mismatch in successor embedding artifact. "
                f"expected={expected_coord_shape}, got={tuple(center_positions.shape)}."
            )
        if atom_ids.shape != (invariant_embeddings.shape[1],):
            raise ValueError(
                "atom_ids shape mismatch in successor embedding artifact. "
                f"expected={(invariant_embeddings.shape[1],)}, got={tuple(atom_ids.shape)}."
            )
        if frame_indices.shape != (invariant_embeddings.shape[0],):
            raise ValueError(
                "frame_indices shape mismatch in successor embedding artifact. "
                f"expected={(invariant_embeddings.shape[0],)}, got={tuple(frame_indices.shape)}."
            )
        if timesteps.shape != (invariant_embeddings.shape[0],):
            raise ValueError(
                "timesteps shape mismatch in successor embedding artifact. "
                f"expected={(invariant_embeddings.shape[0],)}, got={tuple(timesteps.shape)}."
            )
        return cls(
            invariant_embeddings=invariant_embeddings,
            successor_embeddings=successor_embeddings,
            center_positions=center_positions,
            atom_ids=atom_ids,
            frame_indices=frame_indices,
            timesteps=timesteps,
            metadata=metadata,
        )
