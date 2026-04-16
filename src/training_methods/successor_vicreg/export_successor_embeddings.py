from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.analysis.config import load_checkpoint_training_config
from src.training_methods.successor_vicreg.artifact import SuccessorEmbeddingsArtifact
from src.training_methods.successor_vicreg.module import SuccessorVICRegModule
from src.training_methods.vamp.common import (
    build_full_trajectory_dataset,
    build_temporal_dataloader,
    log_progress,
    resolve_device,
    resolve_num_points,
    resolve_radius,
)
from src.utils.model_utils import load_model_from_checkpoint


def _default_output_path(checkpoint_path: str | Path, dump_file: str | Path) -> Path:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    dump_path = Path(dump_file).expanduser().resolve()
    return checkpoint.parent / "successor_vicreg" / f"{dump_path.stem}_successor_embeddings.npz"


def load_successor_checkpoint(
    checkpoint_path: str | Path,
    *,
    cuda_device: int | None = 0,
) -> tuple[SuccessorVICRegModule, object, str]:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    cfg = load_checkpoint_training_config(str(checkpoint))
    device = resolve_device(cuda_device)
    model: SuccessorVICRegModule = load_model_from_checkpoint(
        str(checkpoint),
        cfg,
        device=device,
        module=SuccessorVICRegModule,
    )
    model.to(device).eval()
    return model, cfg, device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export invariant and successor embeddings for every selected atom-centered "
            "local neighborhood in a LAMMPS trajectory."
        )
    )
    parser.add_argument("checkpoint", help="Path to a Successor-VICReg checkpoint.")
    parser.add_argument("dump_file", help="LAMMPS dump file to embed.")
    parser.add_argument("--output", default=None, help="Output NPZ path.")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--radius", type=float, default=None, help="Override cutoff radius.")
    parser.add_argument("--num-points", type=int, default=None, help="Override neighborhood size.")
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--cache-dir", default=None, help="Optional trajectory cache directory.")
    parser.add_argument("--frame-start", type=int, default=None, help="Optional first frame index.")
    parser.add_argument("--frame-stop", type=int, default=None, help="Optional exclusive frame stop.")
    parser.add_argument(
        "--center-selection-mode",
        default=None,
        help="Optional center selection mode override.",
    )
    parser.add_argument(
        "--center-atom-stride",
        type=int,
        default=None,
        help="Optional atom-stride override.",
    )
    parser.add_argument(
        "--max-center-atoms",
        type=int,
        default=None,
        help="Optional cap on tracked center atoms.",
    )
    parser.add_argument(
        "--center-selection-seed",
        type=int,
        default=0,
        help="Center selection seed override.",
    )
    parser.add_argument(
        "--selection-method",
        default="closest",
        help="Neighborhood selection method override.",
    )
    parser.add_argument(
        "--disable-normalize",
        action="store_true",
        help="Disable local-neighborhood normalization during export.",
    )
    parser.add_argument(
        "--disable-center-neighborhoods",
        action="store_true",
        help="Disable local recentering during export.",
    )
    parser.add_argument(
        "--precompute-neighbor-indices",
        action="store_true",
        help="Precompute neighbor indices for the export dataset.",
    )
    parser.add_argument(
        "--tree-cache-size",
        type=int,
        default=4,
        help="KDTree cache size for the export dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    dump_file = Path(args.dump_file).expanduser().resolve()
    output_path = (
        _default_output_path(checkpoint_path, dump_file)
        if args.output is None
        else Path(args.output).expanduser().resolve()
    )

    log_progress(
        "export_successor_embeddings",
        f"loading checkpoint from {checkpoint_path}",
    )
    model, cfg, device = load_successor_checkpoint(
        checkpoint_path,
        cuda_device=int(args.cuda_device),
    )
    num_points = resolve_num_points(cfg, args.num_points)
    radius, radius_source, radius_estimation = resolve_radius(
        dump_file,
        cfg,
        num_points=num_points,
        radius_override=args.radius,
    )

    log_progress(
        "export_successor_embeddings",
        f"using radius={radius:.6f} (source={radius_source}) and num_points={num_points}",
    )
    dataset = build_full_trajectory_dataset(
        dump_file,
        radius=radius,
        num_points=num_points,
        cache_dir=args.cache_dir,
        frame_start=args.frame_start,
        frame_stop=args.frame_stop,
        center_selection_mode=args.center_selection_mode,
        center_atom_stride=args.center_atom_stride,
        max_center_atoms=args.max_center_atoms,
        center_selection_seed=args.center_selection_seed,
        normalize=not args.disable_normalize,
        center_neighborhoods=not args.disable_center_neighborhoods,
        selection_method=str(args.selection_method),
        precompute_neighbor_indices=bool(args.precompute_neighbor_indices),
        tree_cache_size=int(args.tree_cache_size),
    )
    dataloader = build_temporal_dataloader(
        dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )

    atom_ids = np.asarray(dataset.center_atom_ids, dtype=np.int64)
    frame_indices = np.asarray(dataset.window_start_frames, dtype=np.int64)
    timesteps = np.asarray(dataset.timesteps[frame_indices], dtype=np.int64)
    total_samples = int(frame_indices.size * atom_ids.size)
    total_batches = len(dataloader)

    invariant_embeddings = None
    successor_embeddings = None
    center_positions = np.empty((frame_indices.size, atom_ids.size, 3), dtype=np.float32)
    filled = np.zeros((frame_indices.size, atom_ids.size), dtype=bool)

    log_progress(
        "export_successor_embeddings",
        (
            f"dataset ready: selected_frames={frame_indices.size}, tracked_atoms={atom_ids.size}, "
            f"total_samples={total_samples}, batches={total_batches}"
        ),
    )

    model.eval()
    export_start = time.perf_counter()
    progress_every = max(1, total_batches // 20)
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
            z_inv, hat_s = model.predict_successor_from_points(inputs)
            z_np = z_inv.detach().cpu().numpy().astype(np.float32, copy=False)
            hat_s_np = hat_s.detach().cpu().numpy().astype(np.float32, copy=False)

            if invariant_embeddings is None:
                invariant_embeddings = np.empty(
                    (frame_indices.size, atom_ids.size, z_np.shape[1]),
                    dtype=np.float32,
                )
                successor_embeddings = np.empty(
                    (frame_indices.size, atom_ids.size, hat_s_np.shape[1]),
                    dtype=np.float32,
                )

            batch_frame_indices = np.asarray(batch["frame_indices"][:, 0].cpu(), dtype=np.int64)
            batch_atom_ids = np.asarray(batch["center_atom_id"].cpu(), dtype=np.int64)
            batch_centers = np.asarray(batch["center_positions"][:, 0].cpu(), dtype=np.float32)

            frame_slots = np.searchsorted(frame_indices, batch_frame_indices)
            valid_frame_slots = (frame_slots >= 0) & (frame_slots < frame_indices.size)
            if not np.all(valid_frame_slots):
                missing = batch_frame_indices[~valid_frame_slots]
                raise ValueError(
                    "Encountered batch frame indices outside the selected export frame list. "
                    f"missing_frame_indices={missing.tolist()}."
                )
            if not np.array_equal(frame_indices[frame_slots], batch_frame_indices):
                raise ValueError(
                    "Encountered batch frame indices that do not match the selected export frame list. "
                    f"selected_frames={frame_indices.tolist()}, batch_frames={batch_frame_indices.tolist()}."
                )
            atom_slots = np.searchsorted(atom_ids, batch_atom_ids)
            valid_atom_slots = (atom_slots >= 0) & (atom_slots < atom_ids.size)
            if not np.all(valid_atom_slots):
                missing = batch_atom_ids[~valid_atom_slots]
                raise ValueError(
                    "Encountered batch atom ids outside the selected export atom ordering. "
                    f"missing_atom_ids={missing.tolist()}."
                )
            if not np.array_equal(atom_ids[atom_slots], batch_atom_ids):
                raise ValueError(
                    "Encountered batch atom ids that do not match the selected export atom ordering."
                )
            if np.any(filled[frame_slots, atom_slots]):
                raise RuntimeError(
                    "Encountered duplicate frame/atom assignments while exporting successor embeddings. "
                    f"batch_idx={batch_idx}."
                )

            invariant_embeddings[frame_slots, atom_slots] = z_np
            successor_embeddings[frame_slots, atom_slots] = hat_s_np
            center_positions[frame_slots, atom_slots] = batch_centers
            filled[frame_slots, atom_slots] = True

            if (batch_idx + 1) == 1 or (batch_idx + 1) % progress_every == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.perf_counter() - export_start
                log_progress(
                    "export_successor_embeddings",
                    (
                        f"batch={batch_idx + 1}/{total_batches} "
                        f"exported_samples={int(np.count_nonzero(filled))}/{total_samples} "
                        f"elapsed={elapsed:.1f}s"
                    ),
                )

    if invariant_embeddings is None or successor_embeddings is None:
        raise RuntimeError("No successor embeddings were produced from the trajectory.")
    if not np.all(filled):
        missing = np.argwhere(~filled)
        raise RuntimeError(
            "Successor embedding export finished with missing frame/atom entries. "
            f"missing_count={int(missing.shape[0])}, first_missing={missing[:5].tolist()}."
        )

    artifact = SuccessorEmbeddingsArtifact(
        invariant_embeddings=invariant_embeddings,
        successor_embeddings=successor_embeddings,
        center_positions=center_positions,
        atom_ids=atom_ids,
        frame_indices=frame_indices,
        timesteps=timesteps,
        metadata={
            "checkpoint_path": str(checkpoint_path),
            "dump_file": str(dump_file),
            "radius": float(radius),
            "radius_source": str(radius_source),
            "radius_estimation": radius_estimation,
            "num_points": int(num_points),
            "device": str(device),
            "selection_method": str(args.selection_method),
            "normalize": not args.disable_normalize,
            "center_neighborhoods": not args.disable_center_neighborhoods,
            "precompute_neighbor_indices": bool(args.precompute_neighbor_indices),
            "tree_cache_size": int(args.tree_cache_size),
            "source_frame_count": int(dataset.frame_count),
            "training_frame_stride": int(getattr(cfg.data, "frame_stride", 1)),
            "training_window_stride": int(getattr(cfg.data, "window_stride", 1)),
            "training_sequence_length": int(getattr(cfg.data, "sequence_length", 1)),
            "selected_frame_start": None if args.frame_start is None else int(args.frame_start),
            "selected_frame_stop": None if args.frame_stop is None else int(args.frame_stop),
            "center_selection_mode": (
                "random_subset"
                if args.center_selection_mode is None and args.max_center_atoms is not None
                else ("atom_stride" if args.center_selection_mode is None else str(args.center_selection_mode))
            ),
            "center_atom_stride": (
                None if args.center_atom_stride is None else int(args.center_atom_stride)
            ),
            "max_center_atoms": (
                None if args.max_center_atoms is None else int(args.max_center_atoms)
            ),
            "center_selection_seed": int(args.center_selection_seed),
            "tracked_center_count": int(atom_ids.size),
            "successor_enabled": bool(model.enable_successor),
            "successor_horizon_H": int(model.successor_horizon_H),
            "successor_gamma": float(model.successor_gamma),
            "successor_lambda": float(model.successor_lambda),
            "successor_use_ema_teacher": bool(model.successor_use_ema_teacher),
            "successor_teacher_ema_decay": float(model.successor_teacher_ema_decay),
            "successor_use_concat_z_and_successor_for_clustering": bool(
                model.successor_use_concat_z_and_successor_for_clustering
            ),
        },
    )
    saved_path = artifact.save(output_path)
    log_progress(
        "export_successor_embeddings",
        f"saved successor embedding artifact to {saved_path}",
    )


if __name__ == "__main__":
    main()
