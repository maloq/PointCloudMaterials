from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset
from src.temporal_vamp.data import (
    TemporalPairDataset,
    TrajectorySpec,
    all_pair_anchors,
    build_temporal_pair_dataset,
    contiguous_temporal_split,
    event_aligned_frame_interval,
    resolve_lag_frame_offset,
)
from src.temporal_vamp.embeddings import (
    EmbeddingCache,
    FrozenEncoder,
    extract_embedding_cache,
    load_frozen_encoder,
)
from src.temporal_vamp.evaluation import (
    CovariancePCA,
    FutureNeighborCandidateFilter,
    encoder_sanity_checks,
    fit_future_state_labels,
    future_neighbor_consistency,
    future_prediction_probes,
    plot_future_neighbor_comparison,
    plot_kinetic_coordinates,
    plot_singular_spectrum,
    plot_temporal_trajectories,
    regularization_sensitivity,
    save_coordinate_archive,
    write_json,
)
from src.temporal_vamp.linear_vamp import LinearVAMP
from src.temporal_vamp.simulation_catalog import (
    discover_simulation_catalog,
    validate_dump_scan,
)


@dataclass(frozen=True)
class ResolvedLag:
    label: str
    requested_kind: str
    requested_value: int | float
    offsets_by_run: dict[str, int]


@dataclass(frozen=True)
class LagDatasets:
    train: tuple[TemporalPairDataset, ...]
    validation: tuple[TemporalPairDataset, ...]
    split_summary: dict[str, Any]


def _required(cfg: Any, path: str) -> Any:
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise KeyError(f"Temporal VAMP config requires {path!r}.")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def _resolve_device(raw: str) -> str:
    requested = str(raw).strip().lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"embedding.device={raw!r} requests CUDA, but torch.cuda.is_available() is false."
        )
    return str(raw)


def _resolve_trajectories(cfg: DictConfig) -> tuple[TrajectorySpec, ...]:
    catalog_cfg = OmegaConf.select(cfg, "data.catalog", default=None)
    explicit_entries = OmegaConf.select(cfg, "data.trajectories", default=None)
    if catalog_cfg is not None and explicit_entries is not None:
        raise ValueError("Configure exactly one of data.catalog or data.trajectories, not both.")
    if catalog_cfg is not None:
        entries = discover_simulation_catalog(
            _resolve_path(_required(catalog_cfg, "root")),
            campaign_globs=[str(value) for value in _required(catalog_cfg, "campaign_globs")],
            cache_root=_resolve_path(_required(catalog_cfg, "cache_root")),
            required_atom_count=int(_required(catalog_cfg, "required_atom_count")),
            required_potential_parameter_sha256=str(
                _required(catalog_cfg, "required_potential_parameter_sha256")
            ),
            required_crystal_seed=OmegaConf.select(
                catalog_cfg, "required_crystal_seed", default=None
            ),
            require_periodic=bool(_required(catalog_cfg, "require_periodic")),
        )
        return tuple(
            TrajectorySpec(
                path=entry.trajectory_path,
                run_id=entry.run_id,
                cache_dir=entry.cache_dir,
                metadata=entry.metadata,
            )
            for entry in entries
        )
    if explicit_entries is None:
        raise KeyError("Temporal VAMP config requires data.catalog or data.trajectories.")
    entries = list(explicit_entries)
    trajectories: list[TrajectorySpec] = []
    for entry in entries:
        path = _resolve_path(_required(entry, "path"))
        if not path.is_file():
            raise FileNotFoundError(f"Configured trajectory does not exist: {path}")
        run_id = str(_required(entry, "run_id"))
        cache_raw = OmegaConf.select(entry, "cache_dir", default=None)
        cache_dir = None if cache_raw in {None, ""} else _resolve_path(str(cache_raw))
        trajectories.append(TrajectorySpec(path=path, run_id=run_id, cache_dir=cache_dir))
    run_ids = [trajectory.run_id for trajectory in trajectories]
    if not trajectories or len(set(run_ids)) != len(run_ids):
        raise ValueError(
            f"data.trajectories must be non-empty with unique run_id values, got {run_ids}."
        )
    return tuple(trajectories)


def _resolve_lags(
    cfg: DictConfig,
    trajectories: Sequence[TrajectorySpec],
) -> tuple[ResolvedLag, ...]:
    frame_values = OmegaConf.select(cfg, "lags.frames", default=None)
    timestep_values = OmegaConf.select(cfg, "lags.timesteps", default=None)
    picosecond_values = OmegaConf.select(cfg, "lags.picoseconds", default=None)
    configured = [
        value is not None for value in (frame_values, timestep_values, picosecond_values)
    ]
    if sum(configured) != 1:
        raise ValueError(
            "Set exactly one of lags.frames, lags.timesteps, or lags.picoseconds."
        )
    if frame_values is not None:
        requested_kind = "frames"
        raw_values = frame_values
    elif timestep_values is not None:
        requested_kind = "timesteps"
        raw_values = timestep_values
    else:
        requested_kind = "picoseconds"
        raw_values = picosecond_values
    values = [
        float(value) if requested_kind == "picoseconds" else int(value)
        for value in list(raw_values)
    ]
    if not values or len(set(values)) != len(values):
        raise ValueError(f"Configured lag values must be non-empty and unique, got {values}.")

    scans = {
        trajectory.run_id: TemporalLAMMPSDumpDataset.scan_dump_file(
            trajectory.path, cache_dir=trajectory.cache_dir
        )
        for trajectory in trajectories
    }
    for trajectory in trajectories:
        if trajectory.metadata is not None:
            validate_dump_scan(
                trajectory.metadata,
                scans[trajectory.run_id],
                trajectory.path,
            )
    resolved: list[ResolvedLag] = []
    for value in values:
        offsets: dict[str, int] = {}
        for trajectory in trajectories:
            if requested_kind == "picoseconds":
                if trajectory.metadata is None:
                    raise ValueError(
                        "lags.picoseconds requires catalog-backed simulation metadata; "
                        f"run {trajectory.run_id!r} has none."
                    )
                ratio = float(value) / trajectory.metadata.sample_interval_ps
                offset = int(round(ratio))
                if offset <= 0 or not np.isclose(ratio, offset, rtol=0.0, atol=1.0e-9):
                    raise ValueError(
                        f"Physical lag {value} ps is not an exact positive multiple of "
                        f"sample_interval_ps={trajectory.metadata.sample_interval_ps} for "
                        f"run {trajectory.run_id}."
                    )
                if offset >= int(scans[trajectory.run_id].frame_count):
                    raise ValueError(
                        f"Physical lag {value} ps resolves to {offset} frames and leaves no "
                        f"pairs for run {trajectory.run_id}."
                    )
            else:
                offset = resolve_lag_frame_offset(
                    scans[trajectory.run_id].timesteps,
                    lag_frames=int(value) if requested_kind == "frames" else None,
                    lag_timesteps=int(value) if requested_kind == "timesteps" else None,
                )
            offsets[trajectory.run_id] = offset
        label_value = str(value).replace(".", "p")
        resolved.append(
            ResolvedLag(
                label=f"lag_{requested_kind}_{label_value}",
                requested_kind=requested_kind,
                requested_value=value,
                offsets_by_run=offsets,
            )
        )
    return tuple(resolved)


def _dataset_kwargs(cfg: DictConfig) -> dict[str, Any]:
    center_ids_raw = OmegaConf.select(cfg, "data.center_selection.atom_ids", default=None)
    context_enabled = bool(
        OmegaConf.select(cfg, "data.spatial_context.enabled", default=False)
    )
    context_center_count = int(
        OmegaConf.select(cfg, "data.spatial_context.center_count", default=0)
    )
    if context_enabled and context_center_count <= 0:
        raise ValueError(
            "data.spatial_context.enabled=true requires spatial_context.center_count > 0."
        )
    if not context_enabled and context_center_count != 0:
        raise ValueError(
            "data.spatial_context.center_count must be 0 when spatial context is disabled."
        )
    return {
        "num_points": int(_required(cfg, "data.num_points")),
        "radius": float(_required(cfg, "data.radius")),
        "center_selection_mode": str(_required(cfg, "data.center_selection.mode")),
        "center_atom_ids": None if center_ids_raw is None else [int(v) for v in center_ids_raw],
        "center_atom_stride": OmegaConf.select(
            cfg, "data.center_selection.atom_stride", default=None
        ),
        "max_center_atoms": OmegaConf.select(
            cfg, "data.center_selection.max_atoms", default=None
        ),
        "center_selection_seed": int(
            OmegaConf.select(cfg, "data.center_selection.seed", default=0)
        ),
        "center_grid_overlap": OmegaConf.select(
            cfg, "data.center_selection.grid_overlap", default=None
        ),
        "center_grid_reference_frame_index": OmegaConf.select(
            cfg, "data.center_selection.grid_reference_frame", default=None
        ),
        "normalize": bool(_required(cfg, "data.normalize")),
        "center_neighborhoods": bool(_required(cfg, "data.center_neighborhoods")),
        "selection_method": str(_required(cfg, "data.selection_method")),
        "rebuild_cache": bool(OmegaConf.select(cfg, "data.rebuild_cache", default=False)),
        "tree_cache_size": int(OmegaConf.select(cfg, "data.tree_cache_size", default=4)),
        "precompute_neighbor_indices": bool(
            OmegaConf.select(cfg, "data.precompute_neighbor_indices", default=False)
        ),
        "spatial_context_center_count": context_center_count,
    }


def _build_lag_datasets(
    cfg: DictConfig,
    trajectories: Sequence[TrajectorySpec],
    lag: ResolvedLag,
) -> LagDatasets:
    split_mode = str(_required(cfg, "split.mode")).strip().lower()
    if split_mode == "auto":
        split_mode = "run" if len(trajectories) > 1 else "contiguous"
    if split_mode not in {"run", "contiguous"}:
        raise ValueError(f"split.mode must be auto, run, or contiguous; got {split_mode!r}.")
    ratio = float(_required(cfg, "split.train_ratio"))
    frame_start = int(OmegaConf.select(cfg, "data.frame_start", default=0))
    frame_stop_raw = OmegaConf.select(cfg, "data.frame_stop", default=None)
    window_stride = int(OmegaConf.select(cfg, "data.window_stride", default=1))
    gap = int(OmegaConf.select(cfg, "split.boundary_gap_frames", default=0))
    scans = {
        trajectory.run_id: TemporalLAMMPSDumpDataset.scan_dump_file(
            trajectory.path, cache_dir=trajectory.cache_dir
        )
        for trajectory in trajectories
    }
    train_anchors: dict[str, np.ndarray] = {}
    validation_anchors: dict[str, np.ndarray] = {}
    split_summary: dict[str, Any] = {"mode": split_mode, "runs": {}}

    event_window_enabled = bool(
        OmegaConf.select(cfg, "data.event_window.enabled", default=False)
    )
    frame_intervals: dict[str, tuple[int, int | None]] = {}
    if event_window_enabled:
        if frame_start != 0 or frame_stop_raw is not None:
            raise ValueError(
                "data.event_window.enabled=true cannot be combined with non-default "
                "data.frame_start or data.frame_stop."
            )
        reference = str(_required(cfg, "data.event_window.reference")).strip().lower()
        if reference != "nucleation":
            raise ValueError(
                "data.event_window.reference currently supports exactly 'nucleation'; "
                f"got {reference!r}."
            )
        start_offset_ps = float(_required(cfg, "data.event_window.start_offset_ps"))
        stop_offset_ps = float(_required(cfg, "data.event_window.stop_offset_ps"))
        clip_to_trajectory = bool(
            _required(cfg, "data.event_window.clip_to_trajectory")
        )
        split_summary["event_window"] = {
            "reference": reference,
            "start_offset_ps": start_offset_ps,
            "stop_offset_ps": stop_offset_ps,
            "clip_to_trajectory": clip_to_trajectory,
            "label_used_for_sampling_only": True,
        }
        for trajectory in trajectories:
            metadata = trajectory.metadata
            if (
                metadata is None
                or not metadata.nucleation_observed
                or metadata.nucleation_time_ps is None
            ):
                raise ValueError(
                    "Nucleation-aligned sampling requires a detected nucleation time for "
                    f"every trajectory; missing for run {trajectory.run_id!r}."
                )
            frame_intervals[trajectory.run_id] = event_aligned_frame_interval(
                np.asarray(metadata.progress_times_ps, dtype=np.float64),
                event_time_ps=metadata.nucleation_time_ps,
                start_offset_ps=start_offset_ps,
                stop_offset_ps=stop_offset_ps,
                clip_to_trajectory=clip_to_trajectory,
            )
    else:
        frame_intervals = {
            trajectory.run_id: (
                frame_start,
                None if frame_stop_raw is None else int(frame_stop_raw),
            )
            for trajectory in trajectories
        }

    if split_mode == "run":
        if len(trajectories) < 2:
            raise ValueError("split.mode='run' requires at least two independent trajectories.")
        group_by_raw = OmegaConf.select(cfg, "split.group_by", default=None)
        group_by = None if group_by_raw in {None, ""} else str(group_by_raw)
        if group_by is None:
            train_run_count = int(np.floor(len(trajectories) * ratio))
            if train_run_count <= 0 or train_run_count >= len(trajectories):
                raise ValueError(
                    f"Run split with {len(trajectories)} trajectories and train_ratio={ratio} "
                    "produces an empty partition."
                )
            rng = np.random.default_rng(int(OmegaConf.select(cfg, "split.seed", default=0)))
            order = rng.permutation(len(trajectories))
            train_ids = {
                trajectories[int(index)].run_id for index in order[:train_run_count]
            }
            split_summary["group_by"] = None
        else:
            if group_by != "velocity_seed":
                raise ValueError(
                    "Catalog-backed run splitting currently supports group_by='velocity_seed'; "
                    f"got {group_by!r}."
                )
            if any(trajectory.metadata is None for trajectory in trajectories):
                raise ValueError(
                    "split.group_by='velocity_seed' requires catalog metadata on every trajectory."
                )
            groups_by_run = {
                trajectory.run_id: trajectory.metadata.velocity_seed
                for trajectory in trajectories
                if trajectory.metadata is not None
            }
            available_groups = sorted(set(groups_by_run.values()))
            validation_groups_raw = OmegaConf.select(
                cfg, "split.validation_groups", default=None
            )
            if validation_groups_raw is None:
                raise KeyError(
                    "split.group_by='velocity_seed' requires explicit split.validation_groups."
                )
            validation_groups = {int(value) for value in validation_groups_raw}
            missing_groups = sorted(validation_groups.difference(available_groups))
            if missing_groups:
                raise ValueError(
                    f"split.validation_groups contains absent velocity seeds {missing_groups}; "
                    f"available={available_groups}."
                )
            train_ids = {
                run_id
                for run_id, group in groups_by_run.items()
                if group not in validation_groups
            }
            if not train_ids or len(train_ids) == len(trajectories):
                raise ValueError(
                    f"Velocity-seed split produced an empty partition: "
                    f"validation_groups={sorted(validation_groups)}."
                )
            split_summary.update(
                {
                    "group_by": group_by,
                    "train_groups": sorted(set(available_groups).difference(validation_groups)),
                    "validation_groups": sorted(validation_groups),
                }
            )
        for trajectory in trajectories:
            scan = scans[trajectory.run_id]
            run_frame_start, run_frame_stop = frame_intervals[trajectory.run_id]
            anchors = all_pair_anchors(
                frame_count=int(scan.frame_count),
                lag_frames=lag.offsets_by_run[trajectory.run_id],
                frame_start=run_frame_start,
                frame_stop=run_frame_stop,
                window_stride=window_stride,
            )
            if anchors.size == 0:
                raise ValueError(
                    f"Resolved frame interval [{run_frame_start}, {run_frame_stop}) leaves "
                    f"no pairs at lag={lag.requested_value} for run {trajectory.run_id}."
                )
            partition = "train" if trajectory.run_id in train_ids else "validation"
            (train_anchors if partition == "train" else validation_anchors)[trajectory.run_id] = anchors
            split_summary["runs"][trajectory.run_id] = {
                "partition": partition,
                "pair_anchor_count": int(anchors.size),
                "frame_start": run_frame_start,
                "frame_stop_exclusive": (
                    int(scan.frame_count) if run_frame_stop is None else run_frame_stop
                ),
            }
            if trajectory.metadata is not None:
                split_summary["runs"][trajectory.run_id].update(
                    {
                        "temperature_K": trajectory.metadata.temperature_K,
                        "velocity_seed": trajectory.metadata.velocity_seed,
                        "first_anchor_time_ps": float(
                            trajectory.metadata.progress_times_ps[int(anchors[0])]
                        ),
                        "last_future_time_ps": float(
                            trajectory.metadata.progress_times_ps[
                                int(anchors[-1]) + lag.offsets_by_run[trajectory.run_id]
                            ]
                        ),
                    }
                )
    else:
        for trajectory in trajectories:
            scan = scans[trajectory.run_id]
            run_frame_start, run_frame_stop = frame_intervals[trajectory.run_id]
            split = contiguous_temporal_split(
                frame_count=int(scan.frame_count),
                lag_frames=lag.offsets_by_run[trajectory.run_id],
                train_ratio=ratio,
                frame_start=run_frame_start,
                frame_stop=run_frame_stop,
                window_stride=window_stride,
                boundary_gap_frames=gap,
            )
            train_anchors[trajectory.run_id] = split.train
            validation_anchors[trajectory.run_id] = split.validation
            all_anchors = all_pair_anchors(
                frame_count=int(scan.frame_count),
                lag_frames=lag.offsets_by_run[trajectory.run_id],
                frame_start=run_frame_start,
                frame_stop=run_frame_stop,
                window_stride=window_stride,
            )
            split_summary["runs"][trajectory.run_id] = {
                "partition": "contiguous",
                "boundary_frame": split.boundary_frame,
                "train_anchor_count": int(split.train.size),
                "validation_anchor_count": int(split.validation.size),
                "omitted_cross_boundary_or_gap_anchors": int(
                    all_anchors.size - split.train.size - split.validation.size
                ),
                "frame_start": run_frame_start,
                "frame_stop_exclusive": (
                    int(scan.frame_count) if run_frame_stop is None else run_frame_stop
                ),
            }

    kwargs = _dataset_kwargs(cfg)
    train_datasets = tuple(
        build_temporal_pair_dataset(
            trajectory=trajectory,
            anchor_frames=train_anchors[trajectory.run_id],
            lag_frames=lag.offsets_by_run[trajectory.run_id],
            **kwargs,
        )
        for trajectory in trajectories
        if trajectory.run_id in train_anchors
    )
    validation_datasets = tuple(
        build_temporal_pair_dataset(
            trajectory=trajectory,
            anchor_frames=validation_anchors[trajectory.run_id],
            lag_frames=lag.offsets_by_run[trajectory.run_id],
            **kwargs,
        )
        for trajectory in trajectories
        if trajectory.run_id in validation_anchors
    )
    return LagDatasets(
        train=train_datasets,
        validation=validation_datasets,
        split_summary=split_summary,
    )


def _cache_spec(
    cfg: DictConfig,
    *,
    trajectories: Sequence[TrajectorySpec],
    lag: ResolvedLag,
    split: str,
    checkpoint: Path,
) -> dict[str, Any]:
    checkpoint_stat = checkpoint.stat()
    return {
        "checkpoint": {
            "path": str(checkpoint),
            "size": int(checkpoint_stat.st_size),
            "mtime_ns": int(checkpoint_stat.st_mtime_ns),
        },
        "lag": {
            "kind": lag.requested_kind,
            "value": lag.requested_value,
            "offsets_by_run": lag.offsets_by_run,
        },
        "split": split,
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "embedding": {
            "device": str(cfg.embedding.device),
            "repeats": int(cfg.embedding.repeats),
            "seed": int(cfg.embedding.seed),
            "representation_source": str(
                OmegaConf.select(
                    cfg, "encoder.representation_source", default="checkpoint"
                )
            ),
        },
        "trajectories": [
            {
                "path": str(trajectory.path),
                "run_id": trajectory.run_id,
                "size": int(trajectory.path.stat().st_size),
                "mtime_ns": int(trajectory.path.stat().st_mtime_ns),
                "metadata": (
                    None if trajectory.metadata is None else trajectory.metadata.to_dict()
                ),
            }
            for trajectory in trajectories
        ],
    }


def _extract_or_load_caches(
    cfg: DictConfig,
    *,
    lag_dir: Path,
    lag: ResolvedLag,
    datasets: LagDatasets,
    trajectories: Sequence[TrajectorySpec],
    encoder: FrozenEncoder | None,
    extract: bool,
) -> tuple[EmbeddingCache, EmbeddingCache]:
    paths = {
        "train": lag_dir / "embeddings" / "train",
        "validation": lag_dir / "embeddings" / "validation",
    }
    checkpoint = _resolve_path(_required(cfg, "encoder.checkpoint"))
    if extract:
        assert encoder is not None
        batch_size = int(_required(cfg, "embedding.batch_size"))
        point_cloud_batch_size = int(
            OmegaConf.select(
                cfg,
                "embedding.point_cloud_batch_size",
                default=batch_size,
            )
        )
        context_aggregation = str(
            OmegaConf.select(
                cfg,
                "data.spatial_context.aggregation",
                default="mean_std",
            )
        )
        workers = int(_required(cfg, "embedding.num_workers"))
        force = bool(_required(cfg, "cache.force_recompute"))
        train = extract_embedding_cache(
            datasets.train,
            encoder=encoder,
            cache_path=paths["train"],
            cache_spec=_cache_spec(
                cfg,
                trajectories=trajectories,
                lag=lag,
                split="train",
                checkpoint=checkpoint,
            ),
            batch_size=batch_size,
            point_cloud_batch_size=point_cloud_batch_size,
            num_workers=workers,
            force_recompute=force,
            spatial_context_aggregation=context_aggregation,
        )
        validation = extract_embedding_cache(
            datasets.validation,
            encoder=encoder,
            cache_path=paths["validation"],
            cache_spec=_cache_spec(
                cfg,
                trajectories=trajectories,
                lag=lag,
                split="validation",
                checkpoint=checkpoint,
            ),
            batch_size=batch_size,
            point_cloud_batch_size=point_cloud_batch_size,
            num_workers=workers,
            force_recompute=force,
            spatial_context_aggregation=context_aggregation,
        )
        return train, validation
    return EmbeddingCache.load(paths["train"]), EmbeddingCache.load(paths["validation"])


def _fit_models(
    cfg: DictConfig,
    train: EmbeddingCache,
    lag_dir: Path,
) -> tuple[LinearVAMP, CovariancePCA]:
    dimensions = [int(value) for value in _required(cfg, "vamp.dimensions")]
    max_dimension = max(dimensions)
    vamp = LinearVAMP(
        regularization=float(_required(cfg, "vamp.regularization")),
        eigenvalue_cutoff=float(_required(cfg, "vamp.eigenvalue_cutoff")),
        covariance_batch_size=int(_required(cfg, "vamp.covariance_batch_size")),
    ).fit(train.z0, train.z1)
    if max_dimension > vamp.rank:
        raise ValueError(
            f"Requested VAMP dimension {max_dimension} exceeds retained rank {vamp.rank}."
        )
    pca = CovariancePCA.fit(
        train.z0,
        dimension=max_dimension,
        batch_size=int(_required(cfg, "vamp.covariance_batch_size")),
    )
    vamp.save(lag_dir / "vamp_model.npz")
    pca.save(lag_dir / "pca_model.npz")
    return vamp, pca


def _evaluate_lag(
    cfg: DictConfig,
    *,
    lag: ResolvedLag,
    lag_dir: Path,
    train: EmbeddingCache,
    validation: EmbeddingCache,
    datasets: LagDatasets,
    vamp: LinearVAMP,
    pca: CovariancePCA,
    encoder: FrozenEncoder | None,
) -> dict[str, Any]:
    dimensions = [int(value) for value in _required(cfg, "vamp.dimensions")]
    max_dimension = max(dimensions)
    train_kinetic = vamp.transform(train.z0, max_dimension)
    validation_kinetic = vamp.transform(validation.z0, max_dimension)
    train_pca = pca.transform(train.z0, max_dimension)
    validation_pca = pca.transform(validation.z0, max_dimension)
    train_future_reference = train.z1 if train.local_z1 is None else train.local_z1
    validation_future_reference = (
        validation.z1 if validation.local_z1 is None else validation.local_z1
    )

    future_state_cfg = cfg.evaluation.future_state
    train_future_state: np.ndarray | None = None
    validation_future_state: np.ndarray | None = None
    if bool(future_state_cfg.enabled):
        train_future_state, validation_future_state, clusterer = fit_future_state_labels(
            train_future_reference,
            validation_future_reference,
            clusters=int(future_state_cfg.clusters),
            max_fit_samples=int(future_state_cfg.max_fit_samples),
            seed=int(cfg.evaluation.seed),
        )
        np.savez(
            lag_dir / "future_state_kmeans.npz",
            cluster_centers=clusterer.cluster_centers_,
        )

    save_coordinate_archive(
        lag_dir / "coordinates_train.npz",
        train,
        kinetic=train_kinetic,
        pca=train_pca,
        future_state=train_future_state,
    )
    save_coordinate_archive(
        lag_dir / "coordinates_validation.npz",
        validation,
        kinetic=validation_kinetic,
        pca=validation_pca,
        future_state=validation_future_state,
    )

    neighbor_cfg = cfg.evaluation.future_neighbors
    spaces: dict[str, np.ndarray]
    if validation.local_z0 is None:
        spaces = {"encoder": np.asarray(validation.z0)}
    else:
        spaces = {
            "encoder": np.asarray(validation.local_z0),
            "context_encoder": np.asarray(validation.z0),
        }
    for dimension in dimensions:
        spaces[f"pca_{dimension}d"] = validation_pca[:, :dimension]
        spaces[f"vamp_{dimension}d"] = validation_kinetic[:, :dimension]
    neighbor_metrics = {
        name: future_neighbor_consistency(
            values,
            validation_future_reference,
            validation,
            neighbors=int(neighbor_cfg.k),
            max_queries=int(neighbor_cfg.max_queries),
            exclude_same_atom=bool(neighbor_cfg.exclude_same_atom),
            seed=int(cfg.evaluation.seed),
            future_labels=validation_future_state,
        )
        for name, values in spaces.items()
    }
    matched_neighbor_metrics: dict[str, Any] = {}
    matched_profiles = list(
        OmegaConf.select(neighbor_cfg, "matched_profiles", default=[])
    )
    for profile_cfg in matched_profiles:
        profile_name = str(_required(profile_cfg, "name"))
        if profile_name in matched_neighbor_metrics:
            raise ValueError(
                f"evaluation.future_neighbors.matched_profiles contains duplicate name "
                f"{profile_name!r}."
            )
        time_tolerance_raw = OmegaConf.select(
            profile_cfg, "relative_time_tolerance_ps", default=None
        )
        crystallinity_tolerance_raw = OmegaConf.select(
            profile_cfg, "crystalline_fraction_tolerance", default=None
        )
        candidate_filter = FutureNeighborCandidateFilter(
            exclude_same_run=bool(_required(profile_cfg, "exclude_same_run")),
            match_temperature=bool(_required(profile_cfg, "match_temperature")),
            relative_time_tolerance_ps=(
                None if time_tolerance_raw is None else float(time_tolerance_raw)
            ),
            crystalline_fraction_tolerance=(
                None
                if crystallinity_tolerance_raw is None
                else float(crystallinity_tolerance_raw)
            ),
        )
        print(
            f"[temporal-vamp] {lag.label}: matched future-neighbor profile "
            f"{profile_name} {candidate_filter.to_dict()}"
        )
        matched_neighbor_metrics[profile_name] = {
            "candidate_filter": candidate_filter.to_dict(),
            "spaces": {
                name: future_neighbor_consistency(
                    values,
                    validation_future_reference,
                    validation,
                    neighbors=int(neighbor_cfg.k),
                    max_queries=int(neighbor_cfg.max_queries),
                    exclude_same_atom=bool(neighbor_cfg.exclude_same_atom),
                    seed=int(cfg.evaluation.seed),
                    future_labels=validation_future_state,
                    candidate_filter=candidate_filter,
                )
                for name, values in spaces.items()
            },
        }

    probe_metrics = None
    if bool(cfg.evaluation.future_probe.enabled):
        if train_future_state is None or validation_future_state is None:
            raise ValueError(
                "evaluation.future_probe.enabled=true requires evaluation.future_state.enabled=true."
            )
        if train.local_z0 is None or validation.local_z0 is None:
            train_spaces = {"encoder": np.asarray(train.z0)}
            validation_spaces = {"encoder": np.asarray(validation.z0)}
        else:
            train_spaces = {
                "encoder": np.asarray(train.local_z0),
                "context_encoder": np.asarray(train.z0),
            }
            validation_spaces = {
                "encoder": np.asarray(validation.local_z0),
                "context_encoder": np.asarray(validation.z0),
            }
        for dimension in dimensions:
            train_spaces[f"pca_{dimension}d"] = train_pca[:, :dimension]
            train_spaces[f"vamp_{dimension}d"] = train_kinetic[:, :dimension]
            validation_spaces[f"pca_{dimension}d"] = validation_pca[:, :dimension]
            validation_spaces[f"vamp_{dimension}d"] = validation_kinetic[:, :dimension]
        probe_metrics = future_prediction_probes(
            train_spaces,
            validation_spaces,
            train_future_state,
            validation_future_state,
            max_train_samples=int(cfg.evaluation.future_probe.max_train_samples),
            seed=int(cfg.evaluation.seed),
        )

    plots_dir = lag_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_singular_spectrum(
        {lag.label: vamp.singular_values_},
        plots_dir / "singular_spectrum.png",
        max_modes=int(cfg.evaluation.spectrum_modes),
    )
    plot_future_neighbor_comparison(
        neighbor_metrics,
        plots_dir / "future_neighbor_consistency.png",
    )
    for profile_name, profile_metrics in matched_neighbor_metrics.items():
        plot_future_neighbor_comparison(
            profile_metrics["spaces"],
            plots_dir / f"future_neighbor_consistency_{profile_name}.png",
        )
    for color in ("time", "temperature", "crystallinity", "run", "future_state"):
        plot_kinetic_coordinates(
            validation_kinetic,
            validation,
            plots_dir / f"kinetic_xi1_xi2_by_{color}.png",
            color=color,
            future_state=validation_future_state,
            max_points=int(cfg.evaluation.plot_max_points),
            seed=int(cfg.evaluation.seed),
        )
    trajectory_atoms_raw = OmegaConf.select(
        cfg, "evaluation.trajectories.atom_ids", default=None
    )
    plotted_trajectories = plot_temporal_trajectories(
        validation_kinetic,
        validation,
        plots_dir / "selected_atom_trajectories.png",
        atom_ids=None if trajectory_atoms_raw is None else [int(v) for v in trajectory_atoms_raw],
        count=int(cfg.evaluation.trajectories.count),
    )

    sanity: dict[str, Any] = {}
    if bool(cfg.evaluation.sanity.enabled):
        if encoder is None:
            raise RuntimeError("Encoder sanity checks require a loaded frozen encoder.")
        sanity["encoder"] = encoder_sanity_checks(
            encoder,
            vamp,
            datasets.validation[0],
            samples=int(cfg.evaluation.sanity.samples),
            dimension=min(max_dimension, int(cfg.evaluation.sanity.dimension)),
            spatial_context_aggregation=str(
                OmegaConf.select(
                    cfg,
                    "data.spatial_context.aggregation",
                    default="mean_std",
                )
            ),
            point_cloud_batch_size=int(
                OmegaConf.select(
                    cfg,
                    "embedding.point_cloud_batch_size",
                    default=int(cfg.embedding.batch_size),
                )
            ),
        )
        regularizations = [
            float(value) for value in cfg.evaluation.sanity.regularizations
        ]
        sanity["regularization"] = regularization_sensitivity(
            train,
            validation,
            regularizations=regularizations,
            dimension=min(max_dimension, int(cfg.evaluation.sanity.dimension)),
            eigenvalue_cutoff=float(cfg.vamp.eigenvalue_cutoff),
            covariance_batch_size=int(cfg.vamp.covariance_batch_size),
        )

    metrics: dict[str, Any] = {
        "lag": {
            "kind": lag.requested_kind,
            "value": lag.requested_value,
            "frame_offsets_by_run": lag.offsets_by_run,
        },
        "split": datasets.split_summary,
        "sample_counts": {
            "train": int(train.z0.shape[0]),
            "validation": int(validation.z0.shape[0]),
        },
        "embedding_features": {
            "vamp_input_dimension": int(train.z0.shape[1]),
            "local_encoder_dimension": int(
                train.z0.shape[1] if train.local_z0 is None else train.local_z0.shape[1]
            ),
            "has_spatial_context": train.local_z0 is not None,
        },
        "vamp": {
            "rank": vamp.rank,
            "singular_values": vamp.singular_values_.tolist(),
            "ridge0": vamp.ridge0_,
            "ridge1": vamp.ridge1_,
            "regularization": vamp.regularization,
            "eigenvalue_cutoff": vamp.eigenvalue_cutoff,
        },
        "future_neighbor_consistency": neighbor_metrics,
        "future_neighbor_consistency_matched": matched_neighbor_metrics,
        "future_prediction_probe": probe_metrics,
        "sanity": sanity,
        "plotted_trajectories": plotted_trajectories,
    }
    write_json(lag_dir / "metrics.json", metrics)
    return metrics


def run_temporal_vamp(config_path: str | Path, *, stage: str = "all") -> dict[str, Any]:
    config_file = _resolve_path(config_path)
    cfg = OmegaConf.load(config_file)
    OmegaConf.resolve(cfg)
    resolved_stage = str(stage).strip().lower()
    if resolved_stage not in {"all", "extract", "fit", "evaluate"}:
        raise ValueError(f"stage must be all, extract, fit, or evaluate; got {stage!r}.")

    output_dir = _resolve_path(_required(cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = _resolve_trajectories(cfg)
    lags = _resolve_lags(cfg, trajectories)
    checkpoint = _resolve_path(_required(cfg, "encoder.checkpoint"))
    device = _resolve_device(str(_required(cfg, "embedding.device")))
    needs_encoder = resolved_stage in {"all", "extract"} or (
        resolved_stage == "evaluate" and bool(cfg.evaluation.sanity.enabled)
    )
    encoder = (
        load_frozen_encoder(
            checkpoint,
            device=device,
            repeats=int(_required(cfg, "embedding.repeats")),
            seed=int(_required(cfg, "embedding.seed")),
            representation_source=str(
                OmegaConf.select(
                    cfg, "encoder.representation_source", default="checkpoint"
                )
            ),
        )
        if needs_encoder
        else None
    )
    if encoder is not None:
        requested_points = int(cfg.data.num_points)
        model_points = int(encoder.model.model_points)
        if requested_points < model_points:
            raise ValueError(
                f"data.num_points={requested_points} is smaller than checkpoint model_points={model_points}."
            )
        print(
            f"[temporal-vamp] encoder=GeoFrameTransformer checkpoint={checkpoint} "
            f"representation={encoder.representation_source} device={device} "
            f"deterministic={encoder.deterministic} repeats={encoder.repeats}"
        )

    catalog_payload = {
        "trajectory_count": len(trajectories),
        "trajectories": [
            {
                "path": str(trajectory.path),
                "run_id": trajectory.run_id,
                "cache_dir": (
                    None if trajectory.cache_dir is None else str(trajectory.cache_dir)
                ),
                "metadata": (
                    None if trajectory.metadata is None else trajectory.metadata.to_dict()
                ),
            }
            for trajectory in trajectories
        ],
    }
    write_json(output_dir / "dataset_catalog.json", catalog_payload)

    summary: dict[str, Any] = {
        "config": str(config_file),
        "checkpoint": str(checkpoint),
        "representation_source": (
            None if encoder is None else encoder.representation_source
        ),
        "trajectory_count": len(trajectories),
        "stage": resolved_stage,
        "lags": {},
    }
    spectra: dict[str, np.ndarray] = {}
    for lag in lags:
        print(
            f"[temporal-vamp] {lag.label}: frame offsets {lag.offsets_by_run}"
        )
        lag_dir = output_dir / lag.label
        lag_dir.mkdir(parents=True, exist_ok=True)
        datasets = _build_lag_datasets(cfg, trajectories, lag)
        train_cache, validation_cache = _extract_or_load_caches(
            cfg,
            lag_dir=lag_dir,
            lag=lag,
            datasets=datasets,
            trajectories=trajectories,
            encoder=encoder,
            extract=resolved_stage in {"all", "extract"},
        )
        if resolved_stage == "extract":
            summary["lags"][lag.label] = {
                "train_samples": int(train_cache.z0.shape[0]),
                "validation_samples": int(validation_cache.z0.shape[0]),
            }
            continue

        if resolved_stage in {"all", "fit"}:
            vamp, pca = _fit_models(cfg, train_cache, lag_dir)
        else:
            vamp = LinearVAMP.load(lag_dir / "vamp_model.npz")
            pca = CovariancePCA.load(lag_dir / "pca_model.npz")
        spectra[lag.label] = np.asarray(vamp.singular_values_)

        if resolved_stage == "fit":
            summary["lags"][lag.label] = {
                "rank": vamp.rank,
                "singular_values": vamp.singular_values_.tolist(),
            }
            continue
        summary["lags"][lag.label] = _evaluate_lag(
            cfg,
            lag=lag,
            lag_dir=lag_dir,
            train=train_cache,
            validation=validation_cache,
            datasets=datasets,
            vamp=vamp,
            pca=pca,
            encoder=encoder,
        )

    if spectra:
        plot_singular_spectrum(
            spectra,
            output_dir / "singular_spectra_all_lags.png",
            max_modes=int(cfg.evaluation.spectrum_modes),
        )
    write_json(output_dir / "run_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate linear VAMP coordinates on frozen GeoFrameTransformer embeddings."
    )
    parser.add_argument("--config", required=True, help="Temporal VAMP YAML configuration.")
    parser.add_argument(
        "--stage",
        choices=("all", "extract", "fit", "evaluate"),
        default="all",
        help="Run the full experiment or one cache-aware stage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_temporal_vamp(args.config, stage=args.stage)


if __name__ == "__main__":
    main()


__all__ = ["run_temporal_vamp"]
