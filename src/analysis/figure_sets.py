import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .config import FigureSetSettings
from .cluster_figures import (
    _build_cluster_color_map,
    _save_fixed_k_cluster_figure_set,
)
from .cluster_gallery import _save_horizontal_image_gallery
from .output_layout import (
    snapshot_figure_set_dir,
    snapshot_outputs_root,
    write_json,
)


@dataclass(frozen=True)
class SnapshotFigureLayout:
    source_groups: list[tuple[str, np.ndarray]]
    output_names: dict[str, str]
    multi_snapshot_real: bool


def filter_snapshot_figure_layout(
    snapshot_layout: SnapshotFigureLayout,
    *,
    allowed_source_names: list[str] | None,
) -> SnapshotFigureLayout:
    if not allowed_source_names:
        return snapshot_layout
    allowed = [str(v) for v in allowed_source_names]
    allowed_set = set(allowed)
    filtered_groups = [
        (str(source_name), np.asarray(indices, dtype=int))
        for source_name, indices in snapshot_layout.source_groups
        if str(source_name) in allowed_set
    ]
    missing = [name for name in allowed if name not in {str(source_name) for source_name, _ in filtered_groups}]
    if missing:
        raise RuntimeError(
            "Requested snapshot names were not found in the resolved snapshot layout. "
            f"missing={missing}, available={[str(name) for name, _ in snapshot_layout.source_groups]}."
        )
    ordered_groups = [
        next(
            (group_name, group_indices)
            for group_name, group_indices in filtered_groups
            if str(group_name) == name
        )
        for name in allowed
    ]
    filtered_output_names = {
        str(name): str(snapshot_layout.output_names[str(name)])
        for name in allowed
    }
    return SnapshotFigureLayout(
        source_groups=ordered_groups,
        output_names=filtered_output_names,
        multi_snapshot_real=len(ordered_groups) > 1,
    )


def _unwrap_dataset_with_subset_indices(
    dataset: Any,
) -> tuple[Any, list[int] | None]:
    indices: list[int] | None = None
    while isinstance(dataset, torch.utils.data.Subset):
        current_indices = [int(v) for v in list(dataset.indices)]
        if indices is None:
            indices = current_indices
        else:
            indices = [indices[i] for i in current_indices]
        dataset = dataset.dataset
    while hasattr(dataset, "dataset") and not isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    return dataset, indices


def _resolve_sample_source_groups(
    dataset: Any,
    *,
    n_samples: int,
) -> list[tuple[str, np.ndarray]]:
    if n_samples < 0:
        raise ValueError(f"n_samples must be >= 0, got {n_samples}.")
    if n_samples == 0:
        return []

    base_dataset, subset_indices = _unwrap_dataset_with_subset_indices(dataset)
    sample_source_names_raw = getattr(base_dataset, "sample_source_names", None)
    if sample_source_names_raw is None:
        return []

    sample_source_names = [str(v) for v in list(sample_source_names_raw)]
    if subset_indices is not None:
        if any(int(i) < 0 or int(i) >= len(sample_source_names) for i in subset_indices):
            raise IndexError(
                "Subset indices reference sample_source_names out of bounds: "
                f"len(sample_source_names)={len(sample_source_names)}, "
                f"max_index={max(subset_indices) if subset_indices else 'N/A'}."
            )
        sample_source_names = [sample_source_names[int(i)] for i in subset_indices]

    if len(sample_source_names) < int(n_samples):
        raise ValueError(
            "Not enough sample_source_names to map collected analysis samples: "
            f"have {len(sample_source_names)}, need {n_samples}."
        )
    sample_source_names = sample_source_names[: int(n_samples)]

    grouped_indices: dict[str, list[int]] = {}
    for sample_idx, source_name in enumerate(sample_source_names):
        grouped_indices.setdefault(str(source_name), []).append(int(sample_idx))

    return [
        (source_name, np.asarray(indices, dtype=int))
        for source_name, indices in grouped_indices.items()
    ]


def _sanitize_snapshot_output_name(name: str) -> str:
    source_name = Path(str(name)).name or str(name)
    known_data_suffixes = {
        ".dump",
        ".extxyz",
        ".lammpstrj",
        ".npy",
        ".npz",
        ".pos",
        ".traj",
        ".xyz",
    }
    source_path = Path(source_name)
    stem = (
        source_path.stem
        if source_path.suffix.lower() in known_data_suffixes
        else source_name
    )
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return sanitized or "snapshot"


def _build_unique_snapshot_output_names(source_names: list[str]) -> dict[str, str]:
    used: set[str] = set()
    output_names: dict[str, str] = {}
    for source_name in source_names:
        base = _sanitize_snapshot_output_name(source_name)
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        output_names[str(source_name)] = candidate
    return output_names


def _save_snapshot_raytrace_galleries_by_view(
    snapshot_figure_sets: dict[str, Any],
) -> dict[str, Any]:
    snapshots = list(snapshot_figure_sets.get("snapshots") or [])
    if not snapshots:
        raise ValueError("snapshot_figure_sets['snapshots'] must be non-empty.")
    k_value = int(snapshot_figure_sets.get("k_value", -1))
    if k_value < 2:
        raise ValueError(
            f"Invalid snapshot_figure_sets k_value={k_value}; expected an integer >= 2."
        )
    root_dir_raw = snapshot_figure_sets.get("root_dir")
    if not root_dir_raw:
        raise ValueError("snapshot_figure_sets is missing 'root_dir'.")
    gallery_root = Path(str(root_dir_raw)) / "_galleries_by_view" / f"figure_set_k{k_value}"
    gallery_root.mkdir(parents=True, exist_ok=True)
    stale_paths: set[Path] = set()
    for pattern in (
        "01_md_clusters_all_k*_view*_raytrace_gallery.png",
        "02_md_clusters_*_k*_view*_raytrace_gallery.png",
    ):
        stale_paths.update(gallery_root.glob(pattern))
    for stale_path in stale_paths:
        stale_path.unlink()

    def _build_view_lookup(panel_views: Any, *, context: str) -> dict[str, dict[str, Any]]:
        if not isinstance(panel_views, list) or not panel_views:
            raise RuntimeError(f"{context}: expected a non-empty list of panel views.")
        lookup: dict[str, dict[str, Any]] = {}
        for panel_idx, panel in enumerate(panel_views):
            if not isinstance(panel, dict):
                raise RuntimeError(
                    f"{context}: panel view #{panel_idx} must be a dict, got {type(panel)!r}."
                )
            view_name = str(panel.get("view_name", "")).strip()
            if not view_name:
                raise RuntimeError(
                    f"{context}: panel view #{panel_idx} is missing a non-empty 'view_name'."
                )
            if view_name in lookup:
                raise RuntimeError(f"{context}: duplicate view_name={view_name!r}.")
            lookup[view_name] = panel
        return lookup

    def _extract_raytrace_path(panel_view: dict[str, Any], *, context: str) -> Path:
        raytrace_info = panel_view.get("raytrace_render")
        if not isinstance(raytrace_info, dict):
            raise RuntimeError(
                f"{context}: missing raytrace_render metadata. "
                "Re-run with figure_set.raytrace.enabled=true in the analysis config."
            )
        out_file = raytrace_info.get("out_file")
        if not out_file:
            raise RuntimeError(f"{context}: raytrace_render metadata is missing 'out_file'.")
        path = Path(str(out_file))
        if not path.exists():
            raise FileNotFoundError(
                f"{context}: expected raytraced image at {path}, but it is missing."
            )
        return path

    def _snapshot_identity(snapshot_entry: dict[str, Any]) -> dict[str, str]:
        source_name = str(snapshot_entry.get("source_name", "")).strip()
        output_name = str(snapshot_entry.get("output_name", "")).strip()
        if not source_name or not output_name:
            raise RuntimeError(
                "Each snapshot entry must contain non-empty 'source_name' and 'output_name' fields."
            )
        return {
            "source_name": source_name,
            "output_name": output_name,
        }

    first_identity = _snapshot_identity(snapshots[0])
    first_figure_set = snapshots[0].get("figure_set")
    if not isinstance(first_figure_set, dict):
        raise RuntimeError(
            f"Snapshot {first_identity['source_name']} is missing figure_set metadata."
        )
    first_all_cluster_views = first_figure_set.get("panel_all_clusters_views")
    ordered_view_names: list[str] = []
    if isinstance(first_all_cluster_views, list) and first_all_cluster_views:
        all_view_lookup = _build_view_lookup(
            first_all_cluster_views,
            context=f"snapshot={first_identity['source_name']} panel_all_clusters_views",
        )
        ordered_view_names = list(all_view_lookup.keys())

    summary: dict[str, Any] = {
        "root_dir": str(Path(str(root_dir_raw))),
        "gallery_root": str(gallery_root),
        "k_value": k_value,
        "all_clusters": [],
        "crystal_like": {
            "ptm_structure_types": ["FCC", "HCP", "BCC"],
            "snapshots": [],
            "views": [],
        },
    }

    for view_name in ordered_view_names:
        panel_paths: list[Path] = []
        panel_titles: list[str] = []
        for snapshot_entry in snapshots:
            identity = _snapshot_identity(snapshot_entry)
            figure_set_info = snapshot_entry.get("figure_set")
            if not isinstance(figure_set_info, dict):
                raise RuntimeError(
                    f"Snapshot {identity['source_name']} is missing figure_set metadata."
                )
            view_lookup = _build_view_lookup(
                figure_set_info.get("panel_all_clusters_views"),
                context=f"snapshot={identity['source_name']} panel_all_clusters_views",
            )
            if view_name not in view_lookup:
                raise RuntimeError(
                    f"snapshot={identity['source_name']}: missing panel_all_clusters view "
                    f"{view_name!r}. Available views={list(view_lookup)}."
                )
            panel_paths.append(
                _extract_raytrace_path(
                    view_lookup[view_name],
                    context=f"snapshot={identity['source_name']} panel_all_clusters view={view_name}",
                )
            )
            panel_titles.append(identity["output_name"])

        out_file = gallery_root / f"01_md_clusters_all_k{k_value}_{view_name}_raytrace_gallery.png"
        _save_horizontal_image_gallery(
            panel_paths,
            out_file=out_file,
            panel_titles=panel_titles,
        )
        summary["all_clusters"].append(
            {
                "view_name": view_name,
                "out_file": str(out_file),
                "panel_titles": panel_titles,
            }
        )

    rendered_crystal_snapshots: list[dict[str, Any]] = []
    for snapshot_entry in snapshots:
        identity = _snapshot_identity(snapshot_entry)
        figure_set_info = snapshot_entry["figure_set"]
        crystal_ids = [int(v) for v in figure_set_info["crystal_like_cluster_ids"]]
        snapshot_record = {
            **identity,
            "cluster_ids": crystal_ids,
            "rendered": bool(crystal_ids),
        }
        summary["crystal_like"]["snapshots"].append(snapshot_record)
        if crystal_ids:
            rendered_crystal_snapshots.append(
                {
                    "identity": identity,
                    "cluster_ids": crystal_ids,
                    "view_lookup": _build_view_lookup(
                        figure_set_info["panel_crystal_like_views"],
                        context=(
                            f"snapshot={identity['source_name']} "
                            "panel_crystal_like_views"
                        ),
                    ),
                }
            )

    for view_name in ordered_view_names:
        if not rendered_crystal_snapshots:
            break
        panel_paths: list[Path] = []
        panel_titles: list[str] = []
        cluster_ids_by_snapshot: list[dict[str, Any]] = []
        for crystal_snapshot in rendered_crystal_snapshots:
            identity = crystal_snapshot["identity"]
            view_lookup = crystal_snapshot["view_lookup"]
            if view_name not in view_lookup:
                raise RuntimeError(
                    f"snapshot={identity['source_name']}: missing crystal-like view "
                    f"{view_name!r}. Available views={list(view_lookup)}."
                )
            panel_paths.append(
                _extract_raytrace_path(
                    view_lookup[view_name],
                    context=(
                        f"snapshot={identity['source_name']} "
                        f"crystal_like_clusters={crystal_snapshot['cluster_ids']} "
                        f"view={view_name}"
                    ),
                )
            )
            panel_titles.append(identity["output_name"])
            cluster_ids_by_snapshot.append(
                {
                    **identity,
                    "cluster_ids": crystal_snapshot["cluster_ids"],
                }
            )

        out_file = gallery_root / (
            f"02_md_clusters_crystal_like_k{k_value}_{view_name}_raytrace_gallery.png"
        )
        _save_horizontal_image_gallery(
            panel_paths,
            out_file=out_file,
            panel_titles=panel_titles,
        )
        summary["crystal_like"]["views"].append(
            {
                "view_name": view_name,
                "out_file": str(out_file),
                "panel_titles": panel_titles,
                "cluster_ids_by_snapshot": cluster_ids_by_snapshot,
            }
        )
    return summary


@contextmanager
def _temporary_disable_dataset_aug(dataloader: torch.utils.data.DataLoader):
    changes: list[tuple[Any, str, Any]] = []
    ds = getattr(dataloader, "dataset", None)
    while ds is not None:
        for attr in ("random_rotate", "random_jitter"):
            if hasattr(ds, attr):
                prev = getattr(ds, attr)
                if prev != 0.0:
                    changes.append((ds, attr, prev))
                    setattr(ds, attr, 0.0)
        ds = getattr(ds, "dataset", None)
    try:
        yield
    finally:
        for target, attr, prev in changes:
            setattr(target, attr, float(prev))


def resolve_snapshot_figure_layout(
    dataset: Any,
    *,
    is_synthetic: bool,
    n_samples: int,
    analysis_source_names: list[str] | None,
) -> SnapshotFigureLayout:
    if is_synthetic or dataset is None:
        return SnapshotFigureLayout(
            source_groups=[],
            output_names={},
            multi_snapshot_real=False,
        )

    source_groups = _resolve_sample_source_groups(dataset, n_samples=n_samples)
    encountered_source_names = [str(name) for name, _ in source_groups]
    if analysis_source_names is not None and len(analysis_source_names) > 1:
        missing_sources = [
            str(name) for name in analysis_source_names if str(name) not in encountered_source_names
        ]
        if missing_sources:
            raise RuntimeError(
                "Per-snapshot plotting requires collected samples from every requested "
                "analysis snapshot, but some snapshots are missing from the collected prefix. "
                f"missing={missing_sources}, encountered={encountered_source_names}, "
                f"n_samples_collected={n_samples}. Increase analysis_max_samples_total / "
                "max_batches_latent, or disable sampling limits for this analysis run."
            )

    multi_snapshot_real = len(source_groups) > 1
    output_names = (
        _build_unique_snapshot_output_names(encountered_source_names)
        if multi_snapshot_real
        else {}
    )
    if multi_snapshot_real:
        print(f"Per-snapshot plotting enabled for sources: {encountered_source_names}")
    return SnapshotFigureLayout(
        source_groups=source_groups,
        output_names=output_names,
        multi_snapshot_real=multi_snapshot_real,
    )


def build_shared_cluster_color_map(
    labels_for_k: np.ndarray,
    *,
    cluster_color_assignment: dict[int, int | str] | None,
) -> dict[int, str]:
    color_map = _build_cluster_color_map(
        labels_for_k,
        cluster_color_assignment=cluster_color_assignment,
    )
    return {int(cluster_id): str(color) for cluster_id, color in color_map.items()}


def render_cluster_figure_outputs(
    *,
    out_dir: Path,
    dataloader: torch.utils.data.DataLoader,
    figure_settings: FigureSetSettings,
    figure_set_run_kwargs: dict[str, Any],
    labels_for_k: np.ndarray,
    latents: np.ndarray,
    coords: np.ndarray,
    dataset_obj: Any,
    snapshot_layout: SnapshotFigureLayout,
    analysis_source_names: list[str] | None,
    step: Callable[[str], None],
    representative_render_cache: dict[str, Any] | None = None,
    representative_selection_features: np.ndarray | None = None,
    representative_selection_info: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    def _run_figure_set(
        labels_subset: np.ndarray,
        *,
        figure_out_dir: Path | None = None,
        dataset_override: Any | None = None,
        latents_override: np.ndarray | None = None,
        coords_override: np.ndarray | None = None,
        cluster_color_assignment_override: dict[int, int | str] | None = None,
        representative_render_cache_override: dict[str, Any] | None = None,
        representative_selection_features_override: np.ndarray | None = None,
        include_all_cluster_panels_override: bool | None = None,
    ) -> dict[str, Any]:
        step("Generating fixed-k cluster figure set")
        figure_set_dir = (
            out_dir / f"cluster_figure_set_k{figure_settings.k}"
            if figure_out_dir is None
            else Path(figure_out_dir)
        )
        run_kwargs = dict(figure_set_run_kwargs)
        if dataset_override is not None:
            run_kwargs["dataset"] = dataset_override
        if latents_override is not None:
            run_kwargs["latents"] = latents_override
        if coords_override is not None:
            run_kwargs["coords"] = coords_override
        if cluster_color_assignment_override is not None:
            run_kwargs["cluster_color_assignment"] = cluster_color_assignment_override
        if representative_render_cache_override is not None:
            run_kwargs["representative_render_cache"] = representative_render_cache_override
        if representative_selection_features_override is not None:
            run_kwargs["representative_selection_features"] = (
                representative_selection_features_override
            )
            run_kwargs["representative_selection_info"] = representative_selection_info
        if include_all_cluster_panels_override is not None:
            run_kwargs["include_all_cluster_panels"] = bool(include_all_cluster_panels_override)
        with _temporary_disable_dataset_aug(dataloader):
            return _save_fixed_k_cluster_figure_set(
                out_dir=figure_set_dir,
                cluster_labels=labels_subset,
                **run_kwargs,
            )

    if not figure_settings.enabled:
        return None, None

    if not snapshot_layout.multi_snapshot_real:
        return _run_figure_set(
            labels_for_k,
            representative_render_cache_override=representative_render_cache,
            representative_selection_features_override=representative_selection_features,
        ), None

    if dataset_obj is None:
        raise RuntimeError(
            "Cannot generate per-snapshot cluster figure sets: dataloader dataset is missing."
        )

    min_required_samples = int(figure_settings.k) + 1
    too_small = [
        (str(source_name), int(indices.size))
        for source_name, indices in snapshot_layout.source_groups
        if int(indices.size) < min_required_samples
    ]
    if too_small:
        details = ", ".join(f"{name}: {count}" for name, count in too_small)
        raise RuntimeError(
            "Cannot generate per-snapshot cluster figure sets because at least one "
            "snapshot has too few collected samples for the requested fixed-k analysis. "
            f"Need at least {min_required_samples} samples per snapshot for "
            f"k={figure_settings.k}, got {details}. "
            "Increase inputs.max_samples_total / inputs.max_batches_latent, or lower "
            "clustering.primary_k."
        )

    global_color_map = build_shared_cluster_color_map(
        labels_for_k,
        cluster_color_assignment=figure_settings.cluster_color_assignment,
    )
    snapshot_root = snapshot_outputs_root(out_dir)
    snapshot_summary: dict[str, Any] = {
        "root_dir": str(snapshot_root),
        "k_value": int(figure_settings.k),
        "crystal_like_detection": "representative_center_ptm_fcc_hcp_bcc",
        "snapshots": [],
    }
    ordered_snapshot_groups = list(snapshot_layout.source_groups)
    if analysis_source_names is not None and len(analysis_source_names) > 1:
        groups_by_name = {
            str(source_name): np.asarray(indices, dtype=int)
            for source_name, indices in snapshot_layout.source_groups
        }
        ordered_snapshot_groups = [
            (str(source_name), groups_by_name[str(source_name)])
            for source_name in analysis_source_names
            if str(source_name) in groups_by_name
        ]
        ordered_snapshot_groups.extend(
            [
                (str(source_name), np.asarray(indices, dtype=int))
                for source_name, indices in snapshot_layout.source_groups
                if str(source_name) not in {name for name, _ in ordered_snapshot_groups}
            ]
        )

    for source_name, indices in ordered_snapshot_groups:
        snapshot_dirname = snapshot_layout.output_names[str(source_name)]
        snapshot_dir = snapshot_figure_set_dir(
            out_dir,
            snapshot_dirname,
            k_value=int(figure_settings.k),
        )
        subset_dataset = torch.utils.data.Subset(
            dataset_obj,
            [int(v) for v in indices.tolist()],
        )
        figure_info = _run_figure_set(
            labels_for_k[indices],
            figure_out_dir=snapshot_dir,
            dataset_override=subset_dataset,
            latents_override=latents[indices],
            coords_override=coords[indices],
            cluster_color_assignment_override=global_color_map,
            representative_selection_features_override=(
                None
                if representative_selection_features is None
                else representative_selection_features[indices]
            ),
            include_all_cluster_panels_override=True,
        )
        snapshot_summary["snapshots"].append(
            {
                "source_name": str(source_name),
                "output_name": str(snapshot_dirname),
                "sample_count": int(indices.size),
                "figure_set": figure_info,
            }
        )
    if bool(figure_settings.raytrace_enabled):
        snapshot_summary["raytrace_galleries_by_view"] = _save_snapshot_raytrace_galleries_by_view(
            snapshot_summary,
        )
    return None, snapshot_summary


def write_figure_only_metrics(
    *,
    metrics_path: Path,
    all_metrics: dict[str, Any],
    multi_snapshot_real: bool,
) -> dict[str, Any]:
    merged_metrics = {}
    if metrics_path.exists():
        with metrics_path.open("r") as handle:
            merged_metrics = json.load(handle)
    existing_clustering = merged_metrics.get("clustering", {})
    if isinstance(existing_clustering, dict):
        existing_clustering.update(all_metrics["clustering"])
        merged_metrics["clustering"] = existing_clustering
    else:
        merged_metrics["clustering"] = all_metrics["clustering"]
    if "clustering_model_fit" in all_metrics:
        merged_metrics["clustering_model_fit"] = all_metrics["clustering_model_fit"]
    if "runtime_profile" in all_metrics:
        merged_metrics["runtime_profile"] = all_metrics["runtime_profile"]
    merged_metrics["inference_cache"] = all_metrics["inference_cache"]
    if "cluster_figure_set" in all_metrics:
        merged_metrics["cluster_figure_set"] = all_metrics["cluster_figure_set"]
    elif multi_snapshot_real:
        merged_metrics.pop("cluster_figure_set", None)
    if "cluster_figure_sets_by_snapshot" in all_metrics:
        merged_metrics["cluster_figure_sets_by_snapshot"] = all_metrics[
            "cluster_figure_sets_by_snapshot"
        ]
    if "cluster_figure_sets_by_k" in all_metrics:
        merged_metrics["cluster_figure_sets_by_k"] = all_metrics["cluster_figure_sets_by_k"]
    write_json(metrics_path, merged_metrics)
    return merged_metrics


def print_figure_set_summary(
    all_metrics: dict[str, Any],
    *,
    n_samples: int,
    out_dir: Path,
    elapsed: float,
) -> None:
    snapshot_sets = all_metrics.get("cluster_figure_sets_by_snapshot", {})
    has_snapshot_sets = isinstance(snapshot_sets, dict) and bool(snapshot_sets.get("snapshots"))
    if "cluster_figure_set" not in all_metrics and not has_snapshot_sets:
        return
    print(f"\nTotal samples analyzed: {n_samples}")
    print(f"Saved outputs to {out_dir}, runtime: {elapsed:.1f}s")
    if "cluster_figure_set" in all_metrics:
        fs = all_metrics["cluster_figure_set"]
        k_fig = fs.get("k_value", "N/A")
        raytrace_on = bool(fs.get("raytrace_render_settings", {}).get("enabled", False))
        print(f"  - cluster_figure_set_k{k_fig}/cluster_color_assignment_k{k_fig}.json")
        if fs.get("panel_all_clusters_views"):
            print(f"  - cluster_figure_set_k{k_fig}/01_md_clusters_all_k{k_fig}[_view*].png")
        if raytrace_on:
            if fs.get("panel_all_clusters_views"):
                print(f"  - cluster_figure_set_k{k_fig}/01_md_clusters_all_k{k_fig}[_view*]_raytrace.png")
                print(f"  - cluster_figure_set_k{k_fig}/01_md_clusters_all_k{k_fig}_raytrace_gallery.png")
        if fs.get("crystal_like_cluster_ids"):
            print(
                f"  - cluster_figure_set_k{k_fig}/"
                f"02_md_clusters_crystal_like_k{k_fig}[_view*].png"
            )
            if raytrace_on:
                print(f"  - ..._crystal_like_k{k_fig}[_view*]_raytrace.png")
        print(f"  - cluster_figure_set_k{k_fig}/04_cluster_representatives_k{k_fig}*.png")
        rep_analysis = fs.get("panel_representatives_structure_analysis")
        if isinstance(rep_analysis, dict):
            print(
                "  - cluster_figure_set_k"
                f"{k_fig}/10_cluster_representatives_structure_analysis_k{k_fig}.json"
            )
            print(
                "  - cluster_figure_set_k"
                f"{k_fig}/10_cluster_representatives_structure_analysis_k{k_fig}.csv"
            )
    if has_snapshot_sets:
        print("  - snapshots/<snapshot>/figure_set_k*/...")
        snapshot_gallery_sets = snapshot_sets.get("raytrace_galleries_by_view", {})
        if isinstance(snapshot_gallery_sets, dict) and snapshot_gallery_sets.get("all_clusters"):
            k_fig = snapshot_gallery_sets.get("k_value", "N/A")
            print(
                "  - snapshots/_galleries_by_view/"
                f"figure_set_k{k_fig}/01_md_clusters_all_k{k_fig}_view*_raytrace_gallery.png"
            )
            crystal_gallery = snapshot_gallery_sets.get("crystal_like", {})
            if isinstance(crystal_gallery, dict) and crystal_gallery.get("views"):
                print(
                    "  - snapshots/_galleries_by_view/"
                    f"figure_set_k{k_fig}/"
                    f"02_md_clusters_crystal_like_k{k_fig}_view*_raytrace_gallery.png"
                )
