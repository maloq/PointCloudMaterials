from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.training_methods.vamp.common import (
    TrajectoryEmbeddings,
    build_full_trajectory_dataset,
    build_temporal_dataloader,
    load_contrastive_checkpoint,
    log_progress,
    resolve_num_points,
    resolve_radius,
)
from src.training_methods.vamp.config import load_vamp_config, resolve_path


def _default_output_path(checkpoint_path: str | Path, dump_file: str | Path) -> Path:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    dump_path = Path(dump_file).expanduser().resolve()
    return checkpoint.parent / "vamp" / f"{dump_path.stem}_embeddings.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed every atom-centered local neighborhood in a LAMMPS trajectory using a VAMP config."
        )
    )
    parser.add_argument("config", help="Config name inside configs/vamp/ or a YAML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, config_path, base_dir = load_vamp_config(args.config)
    embed_cfg = cfg.embed

    checkpoint_path_text = resolve_path(cfg.paths.checkpoint_path, base_dir=base_dir)
    dump_file_text = resolve_path(cfg.paths.dump_file, base_dir=base_dir)
    if checkpoint_path_text is None:
        raise ValueError("paths.checkpoint_path must be set in the VAMP config.")
    if dump_file_text is None:
        raise ValueError("paths.dump_file must be set in the VAMP config.")

    checkpoint_path = Path(checkpoint_path_text).expanduser().resolve()
    dump_file = Path(dump_file_text).expanduser().resolve()
    output_text = resolve_path(getattr(embed_cfg, "output", None), base_dir=base_dir)
    output_path = (
        _default_output_path(checkpoint_path, dump_file)
        if output_text is None
        else Path(output_text).expanduser().resolve()
    )
    cuda_device = int(getattr(cfg.runtime, "cuda_device", 0))
    radius_override = getattr(embed_cfg, "radius", None)
    num_points_override = getattr(embed_cfg, "num_points", None)
    batch_size = int(getattr(embed_cfg, "batch_size", 256))
    num_workers = int(getattr(embed_cfg, "num_workers", 4))
    cache_dir = resolve_path(getattr(embed_cfg, "cache_dir", None), base_dir=base_dir)
    frame_start = getattr(embed_cfg, "frame_start", None)
    frame_stop = getattr(embed_cfg, "frame_stop", None)
    center_selection_mode = getattr(embed_cfg, "center_selection_mode", None)
    center_atom_stride = getattr(embed_cfg, "center_atom_stride", None)
    max_center_atoms = getattr(embed_cfg, "max_center_atoms", None)
    center_selection_seed = int(getattr(embed_cfg, "center_selection_seed", 0))
    selection_method = str(getattr(embed_cfg, "selection_method", "closest"))
    disable_normalize = bool(getattr(embed_cfg, "disable_normalize", False))
    disable_center_neighborhoods = bool(getattr(embed_cfg, "disable_center_neighborhoods", False))
    precompute_neighbor_indices = bool(getattr(embed_cfg, "precompute_neighbor_indices", False))
    tree_cache_size = int(getattr(embed_cfg, "tree_cache_size", 4))

    log_progress(
        "embed_trajectory",
        f"loading checkpoint from {checkpoint_path} using config={config_path.name}",
    )
    model, cfg, device = load_contrastive_checkpoint(
        checkpoint_path,
        cuda_device=cuda_device,
    )
    log_progress(
        "embed_trajectory",
        f"checkpoint ready on device={device}",
    )
    num_points = resolve_num_points(cfg, num_points_override)
    log_progress(
        "embed_trajectory",
        f"resolved neighborhood size num_points={num_points}; resolving cutoff radius",
    )
    radius, radius_source, radius_estimation = resolve_radius(
        dump_file,
        cfg,
        num_points=num_points,
        radius_override=radius_override,
    )
    log_progress(
        "embed_trajectory",
        f"using radius={radius:.6f} (source={radius_source})",
    )

    log_progress(
        "embed_trajectory",
        (
            f"building trajectory dataset for dump={dump_file} "
            f"frame_start={frame_start} frame_stop={frame_stop} "
            f"center_selection_mode={center_selection_mode} "
            f"center_atom_stride={center_atom_stride} "
            f"max_center_atoms={max_center_atoms}"
        ),
    )
    dataset = build_full_trajectory_dataset(
        dump_file,
        radius=radius,
        num_points=num_points,
        cache_dir=cache_dir,
        frame_start=frame_start,
        frame_stop=frame_stop,
        center_selection_mode=center_selection_mode,
        center_atom_stride=center_atom_stride,
        max_center_atoms=max_center_atoms,
        center_selection_seed=center_selection_seed,
        normalize=not disable_normalize,
        center_neighborhoods=not disable_center_neighborhoods,
        selection_method=selection_method,
        precompute_neighbor_indices=precompute_neighbor_indices,
        tree_cache_size=tree_cache_size,
    )
    dataloader = build_temporal_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    total_batches = len(dataloader)

    atom_ids = np.asarray(dataset.center_atom_ids, dtype=np.int64)
    frame_indices = np.asarray(dataset.window_start_frames, dtype=np.int64)
    timesteps = np.asarray(dataset.timesteps[frame_indices], dtype=np.int64)
    total_samples = int(frame_indices.size * atom_ids.size)
    log_progress(
        "embed_trajectory",
        (
            f"dataset ready: selected_frames={frame_indices.size}, tracked_atoms={atom_ids.size}, "
            f"total_samples={total_samples}, batches={total_batches}, batch_size={batch_size}"
        ),
    )

    embeddings = None
    center_positions = np.empty((frame_indices.size, atom_ids.size, 3), dtype=np.float32)
    filled = np.zeros((frame_indices.size, atom_ids.size), dtype=bool)

    model.eval()
    embed_start = time.perf_counter()
    progress_every = max(1, total_batches // 20)
    log_progress("embed_trajectory", "starting frozen-encoder embedding pass")
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            points = batch["points"]
            if not torch.is_tensor(points):
                points = torch.as_tensor(points)
            if points.ndim != 4 or points.shape[1] != 1 or points.shape[-1] != 3:
                raise ValueError(
                    "Expected TemporalLAMMPSDumpDataset batches with points shaped (B, 1, N, 3), "
                    f"got {tuple(points.shape)}."
                )
            inputs = points[:, 0].to(device=device, dtype=torch.float32, non_blocking=True)
            if hasattr(model, "_prepare_model_input"):
                inputs = model._prepare_model_input(inputs)
            z_inv, _, _ = model(inputs)
            if z_inv is None:
                raise RuntimeError(
                    "The frozen checkpoint did not return invariant embeddings. "
                    f"checkpoint_path={checkpoint_path}."
                )
            z_np = z_inv.detach().cpu().numpy().astype(np.float32, copy=False)

            if embeddings is None:
                embeddings = np.empty(
                    (frame_indices.size, atom_ids.size, z_np.shape[1]),
                    dtype=np.float32,
                )

            batch_frame_indices = np.asarray(batch["frame_indices"][:, 0].cpu(), dtype=np.int64)
            batch_atom_ids = np.asarray(batch["center_atom_id"].cpu(), dtype=np.int64)
            batch_centers = np.asarray(batch["center_positions"][:, 0].cpu(), dtype=np.float32)

            frame_slots = np.searchsorted(frame_indices, batch_frame_indices)
            frame_slot_valid = (frame_slots >= 0) & (frame_slots < frame_indices.size)
            if not np.all(frame_slot_valid):
                missing = batch_frame_indices[~frame_slot_valid]
                raise ValueError(
                    "Encountered batch frame indices outside the selected embedding window. "
                    f"missing_frame_indices={missing.tolist()}."
                )
            if not np.array_equal(frame_indices[frame_slots], batch_frame_indices):
                raise ValueError(
                    "Encountered batch frame indices that do not match the selected embedding frame list. "
                    f"selected_frames={frame_indices.tolist()}, batch_frames={batch_frame_indices.tolist()}."
                )

            atom_slots = np.searchsorted(atom_ids, batch_atom_ids)
            atom_slot_valid = (atom_slots >= 0) & (atom_slots < atom_ids.size)
            if not np.all(atom_slot_valid):
                missing = batch_atom_ids[~atom_slot_valid]
                raise ValueError(
                    "Encountered batch atom ids outside the tracked-center atom list. "
                    f"missing_atom_ids={missing.tolist()}."
                )
            if not np.array_equal(atom_ids[atom_slots], batch_atom_ids):
                raise ValueError(
                    "Encountered batch atom ids that do not match the selected tracked atom ordering. "
                    "Tracked time-lagged pairs would be invalid."
                )

            if np.any(filled[frame_slots, atom_slots]):
                duplicate_pairs = np.argwhere(filled[frame_slots, atom_slots])
                raise RuntimeError(
                    "Encountered duplicate frame/atom assignments while embedding the trajectory. "
                    f"batch_idx={batch_idx}, duplicate_count={int(duplicate_pairs.shape[0])}."
                )

            embeddings[frame_slots, atom_slots] = z_np
            center_positions[frame_slots, atom_slots] = batch_centers
            filled[frame_slots, atom_slots] = True

            if (batch_idx + 1) == 1 or (batch_idx + 1) % progress_every == 0 or (batch_idx + 1) == total_batches:
                sample_count = int(np.count_nonzero(filled))
                elapsed = time.perf_counter() - embed_start
                log_progress(
                    "embed_trajectory",
                    (
                        f"batch={batch_idx + 1}/{total_batches} "
                        f"embedded_samples={sample_count}/{total_samples} "
                        f"elapsed={elapsed:.1f}s"
                    ),
                )

    if embeddings is None:
        raise RuntimeError("No embeddings were produced from the trajectory.")
    if not np.all(filled):
        missing = np.argwhere(~filled)
        raise RuntimeError(
            "Embedding pass finished with missing frame/atom embeddings. "
            f"missing_count={int(missing.shape[0])}, first_missing={missing[:5].tolist()}."
        )

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "dump_file": str(dump_file),
        "config_path": str(config_path),
        "cache_dir": None if cache_dir is None else str(Path(cache_dir).expanduser().resolve()),
        "encoder_name": str(getattr(cfg.encoder, "name", "unknown")),
        "radius": float(radius),
        "radius_source": str(radius_source),
        "radius_estimation": radius_estimation,
        "num_points": int(num_points),
        "device": str(device),
        "selection_method": selection_method,
        "normalize": not disable_normalize,
        "center_neighborhoods": not disable_center_neighborhoods,
        "precompute_neighbor_indices": precompute_neighbor_indices,
        "tree_cache_size": tree_cache_size,
        "source_frame_count": int(dataset.frame_count),
        "selected_frame_start": None if frame_start is None else int(frame_start),
        "selected_frame_stop": None if frame_stop is None else int(frame_stop),
        "center_selection_mode": (
            "random_subset"
            if center_selection_mode is None and max_center_atoms is not None
            else ("atom_stride" if center_selection_mode is None else str(center_selection_mode))
        ),
        "center_atom_stride": (
            None if center_atom_stride is None else int(center_atom_stride)
        ),
        "max_center_atoms": (
            None if max_center_atoms is None else int(max_center_atoms)
        ),
        "center_selection_seed": center_selection_seed,
        "tracked_center_count": int(atom_ids.size),
        "model_points": (
            None if getattr(model, "model_points", None) is None else int(model.model_points)
        ),
    }
    artifact = TrajectoryEmbeddings(
        invariant_embeddings=embeddings,
        center_positions=center_positions,
        atom_ids=atom_ids,
        frame_indices=frame_indices,
        timesteps=timesteps,
        metadata=metadata,
    )
    saved_path = artifact.save(output_path)
    total_elapsed = time.perf_counter() - embed_start
    log_progress(
        "embed_trajectory",
        (
            f"saved frames={artifact.frame_count}, atoms={artifact.num_atoms}, "
            f"latent_dim={artifact.latent_dim} to {saved_path} in {total_elapsed:.1f}s"
        ),
    )


if __name__ == "__main__":
    main()
