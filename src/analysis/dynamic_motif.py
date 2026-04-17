from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from .dynamic_motif_metrics import (
    build_bridge_event_windows,
    build_transition_matrix,
    build_usage_table,
    compute_bridge_tables,
    compute_dwell_tables,
    compute_field_gain,
    compute_frame_motif_proportions,
    compute_hazard_calibration,
    compute_prediction_metrics,
    compute_recurrence_tables,
    compute_transition_tables,
)
from .dynamic_motif_plots import (
    plot_bridge_event_gallery,
    plot_dwell_histograms,
    plot_hazard_calibration_curve,
    plot_motif_latent_2d,
    plot_motif_proportion_area,
    plot_neighbor_influence_heatmap,
    plot_recurrence_heatmap,
    plot_representative_panel,
    plot_survival_curves,
    plot_transition_heatmap,
)
from src.vis_tools.real_md_analysis_vis import save_transition_flow_plot


@dataclass(frozen=True)
class AssignmentBundle:
    stable_ids: np.ndarray
    stable_confidence: np.ndarray
    stable_probs: np.ndarray | None
    bridge_ids: np.ndarray | None
    bridge_confidence: np.ndarray | None
    bridge_probs: np.ndarray | None
    bridge_gate: np.ndarray | None
    bridge_active: np.ndarray
    source_path: np.ndarray
    center_atom_id: np.ndarray | None
    frame_index: np.ndarray | None
    timestep: np.ndarray | None
    residual_norm: np.ndarray | None
    inv_latents: np.ndarray | None
    stable_k: int
    bridge_k: int


def _resolve_dynamic_settings(analysis_cfg: Any) -> Any:
    if hasattr(analysis_cfg, "dynamic_motif"):
        return analysis_cfg.dynamic_motif
    raw_cfg = OmegaConf.select(analysis_cfg, "dynamic_motif", default=None)
    if raw_cfg is None:
        class _DisabledSettings:
            enabled = False
        return _DisabledSettings()
    return raw_cfg


def _per_sample_strings(cache: dict[str, np.ndarray], key: str, *, default: str, num_samples: int) -> np.ndarray:
    if key not in cache:
        return np.asarray([default] * int(num_samples), dtype=str)
    values = np.asarray(cache[key]).reshape(-1)
    if values.shape[0] != int(num_samples):
        raise ValueError(
            f"Cache key {key!r} must have {num_samples} entries, got shape={tuple(values.shape)}."
        )
    return values.astype(str)


def _per_sample_ints(cache: dict[str, np.ndarray], key: str, *, num_samples: int) -> np.ndarray | None:
    if key not in cache:
        return None
    values = np.asarray(cache[key]).reshape(-1)
    if values.shape[0] != int(num_samples):
        raise ValueError(
            f"Cache key {key!r} must have {num_samples} entries, got shape={tuple(values.shape)}."
        )
    return values.astype(np.int64, copy=False)


def _per_sample_floats(cache: dict[str, np.ndarray], key: str, *, num_samples: int) -> np.ndarray | None:
    if key not in cache:
        return None
    values = np.asarray(cache[key]).reshape(-1)
    if values.shape[0] != int(num_samples):
        raise ValueError(
            f"Cache key {key!r} must have {num_samples} entries, got shape={tuple(values.shape)}."
        )
    return values.astype(np.float32, copy=False)


def resolve_motif_assignments(
    cache: dict[str, np.ndarray],
    cluster_labels_fallback: np.ndarray | None = None,
) -> AssignmentBundle:
    if "inv_latents" not in cache:
        raise KeyError("Dynamic motif analysis requires cache['inv_latents'].")
    num_samples = int(np.asarray(cache["inv_latents"]).shape[0])
    if num_samples <= 0:
        raise ValueError("Dynamic motif analysis requires at least one cached sample.")

    stable_probs = None
    stable_ids: np.ndarray | None = None
    if "stable_ids" in cache:
        stable_ids = np.asarray(cache["stable_ids"], dtype=np.int64).reshape(-1)
    elif "stable_probs" in cache:
        stable_probs = np.asarray(cache["stable_probs"], dtype=np.float32)
        if stable_probs.ndim != 2 or stable_probs.shape[0] != int(num_samples):
            raise ValueError(
                "cache['stable_probs'] must have shape (N, K), "
                f"got {tuple(stable_probs.shape)} for N={num_samples}."
            )
        stable_ids = stable_probs.argmax(axis=1).astype(np.int64, copy=False)
    elif cluster_labels_fallback is not None:
        stable_ids = np.asarray(cluster_labels_fallback, dtype=np.int64).reshape(-1)
    else:
        raise ValueError(
            "Could not resolve stable motif assignments. "
            "Cache is missing 'stable_ids'/'stable_probs' and no fallback cluster labels were provided."
        )

    if stable_ids.shape[0] != int(num_samples):
        raise ValueError(
            "Resolved stable motif assignments must align to the sample count. "
            f"num_samples={num_samples}, stable_ids.shape={tuple(stable_ids.shape)}."
        )
    if stable_probs is None and "stable_probs" in cache:
        stable_probs = np.asarray(cache["stable_probs"], dtype=np.float32)
    if stable_probs is not None and (stable_probs.ndim != 2 or stable_probs.shape[0] != int(num_samples)):
        raise ValueError(
            "cache['stable_probs'] must have shape (N, K), "
            f"got {tuple(stable_probs.shape)} for N={num_samples}."
        )
    if stable_probs is not None:
        stable_confidence = stable_probs.max(axis=1).astype(np.float32, copy=False)
        stable_k = int(stable_probs.shape[1])
    else:
        stable_confidence = np.ones((num_samples,), dtype=np.float32)
        stable_k = int(np.max(stable_ids)) + 1 if stable_ids.size > 0 else 0

    bridge_probs = None
    bridge_ids = None
    if "bridge_ids" in cache:
        bridge_ids = np.asarray(cache["bridge_ids"], dtype=np.int64).reshape(-1)
    elif "bridge_probs" in cache:
        bridge_probs = np.asarray(cache["bridge_probs"], dtype=np.float32)
        if bridge_probs.ndim != 2 or bridge_probs.shape[0] != int(num_samples):
            raise ValueError(
                "cache['bridge_probs'] must have shape (N, K_bridge), "
                f"got {tuple(bridge_probs.shape)} for N={num_samples}."
            )
        bridge_ids = bridge_probs.argmax(axis=1).astype(np.int64, copy=False)
    if bridge_probs is None and "bridge_probs" in cache:
        bridge_probs = np.asarray(cache["bridge_probs"], dtype=np.float32)
    if bridge_probs is not None:
        bridge_confidence = bridge_probs.max(axis=1).astype(np.float32, copy=False)
        bridge_k = int(bridge_probs.shape[1])
    elif bridge_ids is not None:
        bridge_confidence = np.ones((num_samples,), dtype=np.float32)
        bridge_k = int(np.max(bridge_ids)) + 1 if bridge_ids.size > 0 else 0
    else:
        bridge_confidence = None
        bridge_k = 0

    bridge_gate = _per_sample_floats(cache, "bridge_gate", num_samples=num_samples)
    if bridge_ids is None:
        bridge_active = np.zeros((num_samples,), dtype=bool)
    elif bridge_gate is not None:
        bridge_active = bridge_gate >= 0.5
    elif bridge_confidence is not None:
        bridge_active = bridge_confidence >= 0.5
    else:
        bridge_active = np.ones((num_samples,), dtype=bool)

    center_atom_id = _per_sample_ints(cache, "center_atom_id", num_samples=num_samples)
    if center_atom_id is None:
        center_atom_id = _per_sample_ints(cache, "instance_ids", num_samples=num_samples)

    return AssignmentBundle(
        stable_ids=stable_ids,
        stable_confidence=stable_confidence,
        stable_probs=stable_probs,
        bridge_ids=bridge_ids,
        bridge_confidence=bridge_confidence,
        bridge_probs=bridge_probs,
        bridge_gate=bridge_gate,
        bridge_active=bridge_active,
        source_path=_per_sample_strings(cache, "source_path", default="<unknown>", num_samples=num_samples),
        center_atom_id=center_atom_id,
        frame_index=_per_sample_ints(cache, "frame_index", num_samples=num_samples),
        timestep=_per_sample_ints(cache, "timestep", num_samples=num_samples),
        residual_norm=_per_sample_floats(cache, "residual_norm", num_samples=num_samples),
        inv_latents=np.asarray(cache["inv_latents"], dtype=np.float32),
        stable_k=int(stable_k),
        bridge_k=int(bridge_k),
    )


def _build_sample_assignments_dataframe(
    cache: dict[str, np.ndarray],
    bundle: AssignmentBundle,
) -> pd.DataFrame:
    num_samples = int(bundle.stable_ids.shape[0])
    sample_index = (
        np.asarray(cache["sample_index"], dtype=np.int64).reshape(-1)
        if "sample_index" in cache
        else np.arange(num_samples, dtype=np.int64)
    )
    if sample_index.shape[0] != int(num_samples):
        raise ValueError(
            "cache['sample_index'] must align to the sample count. "
            f"num_samples={num_samples}, sample_index.shape={tuple(sample_index.shape)}."
        )
    resolved_family = np.where(bundle.bridge_active, "bridge", "stable")
    resolved_motif_id = np.where(
        bundle.bridge_active,
        -1 if bundle.bridge_ids is None else bundle.bridge_ids,
        bundle.stable_ids,
    ).astype(np.int64, copy=False)
    resolved_state_label = np.asarray(
        [
            ("B" if family == "bridge" else "S") + str(int(motif_id))
            for family, motif_id in zip(resolved_family.tolist(), resolved_motif_id.tolist(), strict=True)
        ],
        dtype=str,
    )

    data: dict[str, Any] = {
        "sample_index": sample_index,
        "source_path": bundle.source_path,
        "center_atom_id": (
            np.full((num_samples,), -1, dtype=np.int64)
            if bundle.center_atom_id is None
            else bundle.center_atom_id
        ),
        "frame_index": (
            np.full((num_samples,), -1, dtype=np.int64)
            if bundle.frame_index is None
            else bundle.frame_index
        ),
        "timestep": (
            np.full((num_samples,), -1, dtype=np.int64)
            if bundle.timestep is None
            else bundle.timestep
        ),
        "stable_id": bundle.stable_ids,
        "stable_confidence": bundle.stable_confidence,
        "bridge_id": (
            np.full((num_samples,), -1, dtype=np.int64)
            if bundle.bridge_ids is None
            else bundle.bridge_ids
        ),
        "bridge_confidence": (
            np.full((num_samples,), np.nan, dtype=np.float32)
            if bundle.bridge_confidence is None
            else bundle.bridge_confidence
        ),
        "bridge_gate": (
            np.full((num_samples,), np.nan, dtype=np.float32)
            if bundle.bridge_gate is None
            else bundle.bridge_gate
        ),
        "bridge_active": bundle.bridge_active.astype(bool, copy=False),
        "resolved_family": resolved_family,
        "resolved_motif_id": resolved_motif_id,
        "resolved_state_label": resolved_state_label,
        "residual_norm": (
            np.full((num_samples,), np.nan, dtype=np.float32)
            if bundle.residual_norm is None
            else bundle.residual_norm
        ),
    }
    coords = np.asarray(cache.get("coords", np.empty((0, 3), dtype=np.float32)))
    if coords.ndim == 2 and coords.shape == (num_samples, 3):
        data["coord_x"] = coords[:, 0]
        data["coord_y"] = coords[:, 1]
        data["coord_z"] = coords[:, 2]

    for key, value in cache.items():
        if key in data or key in {"inv_latents", "eq_latents", "stable_probs", "bridge_probs", "coords"}:
            continue
        array = np.asarray(value)
        if array.ndim == 1 and array.shape[0] == int(num_samples):
            data[key] = array
    return pd.DataFrame(data)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _path_relative_to(base_dir: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.relative_to(base_dir))


def _equidistant_selection_indices(*, num_items: int, num_selected: int) -> np.ndarray:
    if num_items <= 0:
        raise ValueError(f"num_items must be > 0, got {num_items}.")
    if num_selected <= 0:
        raise ValueError(f"num_selected must be > 0, got {num_selected}.")
    if num_selected > num_items:
        raise ValueError(
            f"num_selected ({num_selected}) cannot exceed num_items ({num_items})."
        )
    if num_selected == 1:
        return np.asarray([0], dtype=np.int64)
    raw = np.linspace(0, num_items - 1, num=num_selected, dtype=np.float64)
    rounded = np.rint(raw).astype(np.int64)
    unique = np.unique(rounded)
    if unique.size != num_selected:
        raise RuntimeError(
            "Equidistant selection produced duplicate indices. "
            f"num_items={num_items}, num_selected={num_selected}, raw={raw.tolist()}, "
            f"rounded={rounded.tolist()}."
        )
    return unique.astype(np.int64, copy=False)


def _format_transition_snapshot_label(*, frame_index: int, timestep: int | None) -> str:
    if timestep is None:
        return f"frame {int(frame_index)}"
    return f"frame {int(frame_index)} (t={int(timestep)})"


def _remove_existing_transition_snapshot_flow_artifacts(transitions_dir: Path) -> None:
    for pattern in ("transition_snapshot_flow_*.png", "transition_snapshot_flow_index.csv"):
        for path in transitions_dir.glob(pattern):
            path.unlink()


def _resolve_transition_snapshot_anchors(
    sample_df: pd.DataFrame,
    *,
    num_flow_charts: int,
) -> list[dict[str, Any]]:
    if int(num_flow_charts) <= 0:
        raise ValueError(
            "num_flow_charts must be > 0 for snapshot transition flows, "
            f"got {num_flow_charts}."
        )
    required = {"frame_index", "timestep"}
    missing = sorted(required.difference(sample_df.columns))
    if missing:
        raise KeyError(
            "Transition snapshot anchors require sample_df columns "
            f"{sorted(required)}, missing {missing}."
        )

    frame_table = (
        sample_df.loc[:, ["frame_index", "timestep"]]
        .drop_duplicates()
        .sort_values(["frame_index", "timestep"])
        .reset_index(drop=True)
    )
    duplicate_frame_counts = frame_table["frame_index"].value_counts()
    conflicting_frames = sorted(
        int(frame_index)
        for frame_index, count in duplicate_frame_counts.items()
        if int(count) > 1
    )
    if conflicting_frames:
        raise ValueError(
            "Transition snapshot flow anchors found multiple timestep values for the same frame index. "
            f"Conflicting frame indices: {conflicting_frames[:10]}."
        )

    anchor_count = int(num_flow_charts) + 1
    if frame_table.shape[0] < anchor_count:
        raise ValueError(
            "Not enough unique frames to render the requested number of transition snapshot flows. "
            f"unique_frames={int(frame_table.shape[0])}, requested_flows={int(num_flow_charts)}, "
            f"required_anchor_frames={anchor_count}."
        )

    anchor_positions = _equidistant_selection_indices(
        num_items=int(frame_table.shape[0]),
        num_selected=anchor_count,
    )
    anchors = frame_table.iloc[anchor_positions].reset_index(drop=True)
    return [
        {
            "frame_index": int(row.frame_index),
            "timestep": None if pd.isna(row.timestep) else int(row.timestep),
            "label": _format_transition_snapshot_label(
                frame_index=int(row.frame_index),
                timestep=None if pd.isna(row.timestep) else int(row.timestep),
            ),
        }
        for row in anchors.itertuples(index=False)
    ]


def _build_transition_snapshot_counts(
    sample_df: pd.DataFrame,
    *,
    frame_index_from: int,
    frame_index_to: int,
) -> pd.DataFrame:
    required = {
        "source_path",
        "center_atom_id",
        "frame_index",
        "resolved_family",
        "resolved_motif_id",
        "resolved_state_label",
    }
    missing = sorted(required.difference(sample_df.columns))
    if missing:
        raise KeyError(
            "Transition snapshot flow counts require sample_df columns "
            f"{sorted(required)}, missing {missing}."
        )

    pair_columns = [
        "source_path",
        "center_atom_id",
        "resolved_family",
        "resolved_motif_id",
        "resolved_state_label",
    ]
    left = sample_df.loc[
        sample_df["frame_index"] == int(frame_index_from),
        pair_columns,
    ].copy()
    right = sample_df.loc[
        sample_df["frame_index"] == int(frame_index_to),
        pair_columns,
    ].copy()
    if left.empty or right.empty:
        raise ValueError(
            "Transition snapshot flow pair is missing one of the requested anchor frames. "
            f"frame_index_from={int(frame_index_from)}, left_rows={int(left.shape[0])}, "
            f"frame_index_to={int(frame_index_to)}, right_rows={int(right.shape[0])}."
        )

    merged = left.merge(
        right,
        on=["source_path", "center_atom_id"],
        how="inner",
        suffixes=("_from", "_to"),
    )
    if merged.empty:
        raise ValueError(
            "Transition snapshot flow pair has no overlapping tracked centers between anchor frames. "
            f"frame_index_from={int(frame_index_from)}, frame_index_to={int(frame_index_to)}."
        )

    counts = (
        merged.groupby(
            [
                "resolved_family_from",
                "resolved_motif_id_from",
                "resolved_family_to",
                "resolved_motif_id_to",
                "resolved_state_label_from",
                "resolved_state_label_to",
            ],
            sort=True,
        )
        .size()
        .reset_index(name="count")
        .rename(
            columns={
                "resolved_family_from": "from_family",
                "resolved_motif_id_from": "from_id",
                "resolved_family_to": "to_family",
                "resolved_motif_id_to": "to_id",
                "resolved_state_label_from": "from_label",
                "resolved_state_label_to": "to_label",
            }
        )
        .sort_values(["count", "from_label", "to_label"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    counts["total_transitions"] = int(counts["count"].sum())
    counts["fraction"] = counts["count"].astype(np.float64) / float(counts["count"].sum())
    return counts


def _select_representatives(
    sample_df: pd.DataFrame,
    *,
    motif_family: str,
    motif_id: int,
    max_samples: int,
) -> list[int]:
    motif_df = sample_df.loc[
        (sample_df["resolved_family"] == str(motif_family))
        & (sample_df["resolved_motif_id"] == int(motif_id))
    ].copy()
    if motif_df.empty:
        return []
    confidence_col = "stable_confidence" if motif_family == "stable" else "bridge_confidence"
    if confidence_col not in motif_df.columns:
        confidence_col = "stable_confidence"
    motif_df = motif_df.sort_values(
        [confidence_col, "residual_norm", "frame_index", "sample_index"],
        ascending=[False, False, True, True],
    )
    if motif_df.shape[0] <= int(max_samples):
        return motif_df["sample_index"].astype(int).tolist()

    selected: list[int] = []
    for fraction in np.linspace(0.0, 1.0, int(max_samples)):
        row_idx = int(round(fraction * float(motif_df.shape[0] - 1)))
        sample_index = int(motif_df.iloc[row_idx]["sample_index"])
        if sample_index not in selected:
            selected.append(sample_index)
        if len(selected) >= int(max_samples):
            break
    if len(selected) < int(max_samples):
        for sample_index in motif_df["sample_index"].astype(int).tolist():
            if sample_index in selected:
                continue
            selected.append(sample_index)
            if len(selected) >= int(max_samples):
                break
    return selected


def _build_summary_markdown(
    *,
    out_file: Path,
    dynamic_dir: Path,
    metrics: dict[str, Any],
    artifact_paths: dict[str, str],
    warnings_list: list[str],
) -> None:
    lines = [
        "# Dynamic Motif Analysis",
        "",
        f"- Stable motifs configured: `{int(metrics.get('num_stable_motifs', 0))}`",
        f"- Stable motifs active: `{int(metrics.get('stable_active_count', 0))}`",
        f"- Bridge motifs configured: `{int(metrics.get('num_bridge_motifs', 0))}`",
        f"- Bridge motifs active: `{int(metrics.get('bridge_active_count', 0))}`",
        f"- Self-transition fraction: `{metrics.get('self_transition_fraction', float('nan')):.4f}`",
        f"- Mean dwell: `{metrics.get('mean_dwell_overall', float('nan')):.4f}`",
        f"- Revisit rate: `{metrics.get('revisit_rate', float('nan')):.4f}`",
    ]
    if warnings_list:
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {warning}" for warning in warnings_list])
    if artifact_paths:
        lines.extend(["", "## Key artifacts"])
        for key, relative_path in sorted(artifact_paths.items()):
            lines.append(f"- `{key}`: `{relative_path}`")
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dynamic_motif_analysis(
    *,
    cache: dict[str, np.ndarray],
    out_dir: Path,
    model_cfg: Any,
    analysis_cfg: Any,
    cluster_labels_primary: np.ndarray | None,
    step: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    settings = _resolve_dynamic_settings(analysis_cfg)
    if not bool(getattr(settings, "enabled", False)):
        return {"enabled": False}

    if step is not None:
        step("Running dynamic motif analysis")

    dynamic_dir = Path(out_dir) / "dynamic_motif"
    assignments_dir = dynamic_dir / "assignments"
    transitions_dir = dynamic_dir / "transitions"
    dwell_dir = dynamic_dir / "dwell"
    recurrence_dir = dynamic_dir / "recurrence"
    representatives_dir = dynamic_dir / "representatives"
    temporal_dir = dynamic_dir / "temporal"
    field_dir = dynamic_dir / "field"
    for directory in (
        assignments_dir,
        transitions_dir,
        dwell_dir,
        recurrence_dir,
        representatives_dir,
        temporal_dir,
        field_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    assignment_cache = dict(cache)
    if not bool(getattr(settings, "use_model_outputs", True)):
        for key in ("stable_ids", "stable_probs", "bridge_ids", "bridge_probs", "bridge_gate"):
            assignment_cache.pop(key, None)

    warnings_list: list[str] = []
    try:
        bundle = resolve_motif_assignments(
            assignment_cache,
            cluster_labels_fallback=cluster_labels_primary,
        )
    except ValueError as exc:
        return {
            "enabled": True,
            "skipped": True,
            "warnings": [str(exc)],
        }

    sample_df = _build_sample_assignments_dataframe(cache, bundle)
    sample_assignments_path: Path | None = None
    if bool(getattr(settings, "export_per_sample_arrays", True)):
        sample_assignments_path = assignments_dir / "sample_assignments.csv"
        _write_csv(sample_df, sample_assignments_path)

    stable_usage_df, stable_usage_metrics = build_usage_table(
        bundle.stable_ids,
        confidences=bundle.stable_confidence,
        motif_family="stable",
        total_motif_count=(
            getattr(settings, "stable_k", None)
            if getattr(settings, "stable_k", None) is not None
            else bundle.stable_k
        ),
    )
    stable_usage_path = assignments_dir / "stable_usage.csv"
    _write_csv(stable_usage_df, stable_usage_path)

    bridge_usage_metrics: dict[str, Any] = {
        "bridge_active_count": 0,
        "bridge_usage_entropy": float("nan"),
        "bridge_dead_fraction": float("nan"),
    }
    bridge_usage_path: Path | None = None
    if bundle.bridge_ids is not None:
        active_bridge_ids = bundle.bridge_ids[bundle.bridge_active]
        active_bridge_conf = (
            None
            if bundle.bridge_confidence is None
            else bundle.bridge_confidence[bundle.bridge_active]
        )
        bridge_usage_df_assign, bridge_usage_metrics = build_usage_table(
            active_bridge_ids,
            confidences=active_bridge_conf,
            motif_family="bridge",
            total_motif_count=(
                getattr(settings, "bridge_k", None)
                if getattr(settings, "bridge_k", None) is not None
                else bundle.bridge_k
            ),
        )
        bridge_usage_path = assignments_dir / "bridge_usage.csv"
        _write_csv(bridge_usage_df_assign, bridge_usage_path)
    else:
        warnings_list.append(
            "Cache does not contain bridge outputs; bridge-specific sections were skipped."
        )

    frame_proportions_df = compute_frame_motif_proportions(sample_df)
    frame_proportions_path = assignments_dir / "frame_motif_proportions.csv"
    _write_csv(frame_proportions_df, frame_proportions_path)

    track_columns_available = bundle.center_atom_id is not None and bundle.frame_index is not None
    if not track_columns_available:
        warnings_list.append(
            "Cache is missing center_atom_id/frame_index metadata, so transition/dwell/recurrence analyses were skipped."
        )

    transition_counts = pd.DataFrame()
    transition_probs = pd.DataFrame()
    bridge_triplets = pd.DataFrame()
    dwell_df = pd.DataFrame()
    survival_df = pd.DataFrame()
    hazard_df = pd.DataFrame()
    hazard_calibration_df = pd.DataFrame()
    recurrence_scores_df = pd.DataFrame()
    revisit_df = pd.DataFrame()
    bridge_stats_df = pd.DataFrame()
    field_gain_df = pd.DataFrame()
    dynamic_metrics: dict[str, Any] = {
        "enabled": True,
        "warnings": warnings_list,
        "num_stable_motifs": int(
            getattr(settings, "stable_k", None)
            if getattr(settings, "stable_k", None) is not None
            else bundle.stable_k
        ),
        "num_bridge_motifs": int(
            getattr(settings, "bridge_k", None)
            if getattr(settings, "bridge_k", None) is not None
            else bundle.bridge_k
        ),
        "stable_active_count": int(stable_usage_metrics["stable_active_count"]),
        "bridge_active_count": int(bridge_usage_metrics["bridge_active_count"]),
        "stable_usage_entropy": float(stable_usage_metrics["stable_usage_entropy"]),
        "bridge_usage_entropy": float(bridge_usage_metrics["bridge_usage_entropy"]),
        "artifacts": {},
    }

    prediction_metrics = compute_prediction_metrics(sample_df)
    dynamic_metrics.update(prediction_metrics)
    hazard_calibration_df, hazard_metrics = compute_hazard_calibration(sample_df)
    dynamic_metrics.update(hazard_metrics)

    if track_columns_available:
        transition_counts, transition_probs, bridge_triplets, transition_metrics = compute_transition_tables(sample_df)
        dynamic_metrics.update(transition_metrics)
        dwell_df, survival_df, hazard_df, dwell_metrics = compute_dwell_tables(
            sample_df,
            dwell_min_length=int(getattr(settings, "dwell_min_length", 1)),
        )
        dynamic_metrics.update(dwell_metrics)
        bridge_stats_df, bridge_triplets, bridge_metrics = compute_bridge_tables(
            sample_df,
            dwell_df=dwell_df,
            transition_counts=transition_counts,
            bridge_triplets=bridge_triplets,
            bridge_min_support=int(getattr(settings, "bridge_min_support", 50)),
        )
        dynamic_metrics.update(bridge_metrics)
        recurrence_scores_df, revisit_df, recurrence_metrics = compute_recurrence_tables(
            dwell_df,
            recurrence_max_gap=int(getattr(settings, "recurrence_max_gap", 64)),
        )
        dynamic_metrics.update(recurrence_metrics)

        if not transition_counts.empty:
            transition_counts_path = transitions_dir / "transition_counts.csv"
            transition_probs_path = transitions_dir / "transition_probs.csv"
            triplets_path = transitions_dir / "stable_to_bridge_to_stable.csv"
            _write_csv(transition_counts, transition_counts_path)
            _write_csv(transition_probs, transition_probs_path)
            _write_csv(bridge_triplets, triplets_path)
            dynamic_metrics["artifacts"]["transition_counts_csv"] = _path_relative_to(out_dir, transition_counts_path)
            dynamic_metrics["artifacts"]["transition_probs_csv"] = _path_relative_to(out_dir, transition_probs_path)
            dynamic_metrics["artifacts"]["stable_to_bridge_to_stable_csv"] = _path_relative_to(out_dir, triplets_path)

        if not dwell_df.empty:
            dwell_times_path = dwell_dir / "dwell_times.csv"
            survival_path = dwell_dir / "survival_curves.csv"
            hazard_by_motif_path = dwell_dir / "hazard_by_motif.csv"
            _write_csv(dwell_df, dwell_times_path)
            _write_csv(survival_df, survival_path)
            _write_csv(hazard_df, hazard_by_motif_path)
            dynamic_metrics["artifacts"]["dwell_times_csv"] = _path_relative_to(out_dir, dwell_times_path)
            dynamic_metrics["artifacts"]["survival_curves_csv"] = _path_relative_to(out_dir, survival_path)
            dynamic_metrics["artifacts"]["hazard_by_motif_csv"] = _path_relative_to(out_dir, hazard_by_motif_path)

        if not bridge_stats_df.empty:
            bridge_stats_path = assignments_dir / "bridge_usage.csv"
            _write_csv(bridge_stats_df, bridge_stats_path)
            bridge_usage_path = bridge_stats_path
            dynamic_metrics["artifacts"]["bridge_usage_csv"] = _path_relative_to(out_dir, bridge_stats_path)

        if not recurrence_scores_df.empty:
            recurrence_scores_path = recurrence_dir / "recurrence_scores.csv"
            revisit_counts_path = recurrence_dir / "motif_revisit_counts.csv"
            _write_csv(recurrence_scores_df, recurrence_scores_path)
            _write_csv(revisit_df, revisit_counts_path)
            dynamic_metrics["artifacts"]["recurrence_scores_csv"] = _path_relative_to(out_dir, recurrence_scores_path)
            dynamic_metrics["artifacts"]["motif_revisit_counts_csv"] = _path_relative_to(out_dir, revisit_counts_path)

    if not hazard_calibration_df.empty:
        hazard_calibration_path = dwell_dir / "hazard_calibration.csv"
        _write_csv(hazard_calibration_df, hazard_calibration_path)
        dynamic_metrics["artifacts"]["hazard_calibration_csv"] = _path_relative_to(out_dir, hazard_calibration_path)

    if bool(getattr(getattr(settings, "field", None), "enabled", False)):
        field_gain_df, field_metrics = compute_field_gain(sample_df)
        dynamic_metrics.update(field_metrics)
        if not field_gain_df.empty:
            neighbor_gain_path = field_dir / "neighbor_gain.csv"
            _write_csv(field_gain_df, neighbor_gain_path)
            dynamic_metrics["artifacts"]["neighbor_gain_csv"] = _path_relative_to(out_dir, neighbor_gain_path)
        else:
            warnings_list.append(
                "Field analysis was enabled, but cache does not contain field predictions aligned with future targets."
            )

    timelines_path = temporal_dir / "motif_timelines.csv"
    atlas_path = temporal_dir / "per_atom_event_atlas.csv"
    _write_csv(sample_df.sort_values(["source_path", "center_atom_id", "frame_index", "timestep"]), timelines_path)
    _write_csv(sample_df.sort_values(["source_path", "center_atom_id", "frame_index", "timestep"]), atlas_path)
    dynamic_metrics["artifacts"]["motif_timelines_csv"] = _path_relative_to(out_dir, timelines_path)
    dynamic_metrics["artifacts"]["per_atom_event_atlas_csv"] = _path_relative_to(out_dir, atlas_path)
    if sample_assignments_path is not None:
        dynamic_metrics["artifacts"]["sample_assignments_csv"] = _path_relative_to(out_dir, sample_assignments_path)
    dynamic_metrics["artifacts"]["frame_motif_proportions_csv"] = _path_relative_to(out_dir, frame_proportions_path)
    dynamic_metrics["artifacts"]["stable_usage_csv"] = _path_relative_to(out_dir, stable_usage_path)
    if bridge_usage_path is not None:
        dynamic_metrics["artifacts"]["bridge_usage_csv"] = _path_relative_to(out_dir, bridge_usage_path)

    transition_matrix_path: Path | None = None
    transition_snapshot_flow_index_path: Path | None = None
    transition_snapshot_flow_records: list[dict[str, Any]] = []
    dwell_hist_path: Path | None = None
    survival_plot_path: Path | None = None
    recurrence_plot_path: Path | None = None
    bridge_gallery_path: Path | None = None
    motif_projection_path: Path | None = None
    hazard_curve_path: Path | None = None
    neighbor_heatmap_path: Path | None = None
    representative_index_path: Path | None = None

    if not transition_counts.empty and bool(getattr(getattr(settings, "render", None), "heatmaps", True)):
        state_ids = sorted(
            set(zip(transition_counts["from_family"], transition_counts["from_id"], strict=False)).union(
                set(zip(transition_counts["to_family"], transition_counts["to_id"], strict=False))
            ),
            key=lambda item: (str(item[0]), int(item[1])),
        )
        matrix, labels = build_transition_matrix(
            transition_counts,
            state_ids=[(str(family), int(motif_id)) for family, motif_id in state_ids],
        )
        transition_matrix_path = transitions_dir / "transition_matrix.png"
        plot_transition_heatmap(matrix, labels, transition_matrix_path)
    if not transition_counts.empty and bool(getattr(getattr(settings, "render", None), "sankey", True)):
        _remove_existing_transition_snapshot_flow_artifacts(transitions_dir)
        legacy_transition_flow_path = transitions_dir / "transition_flow.png"
        if legacy_transition_flow_path.exists():
            legacy_transition_flow_path.unlink()
        snapshot_flow_count = int(getattr(settings, "transition_snapshot_flow_count", 0))
        if snapshot_flow_count > 0:
            anchors = _resolve_transition_snapshot_anchors(
                sample_df,
                num_flow_charts=int(snapshot_flow_count),
            )
            for pair_idx in range(len(anchors) - 1):
                anchor_from = anchors[pair_idx]
                anchor_to = anchors[pair_idx + 1]
                pair_counts = _build_transition_snapshot_counts(
                    sample_df,
                    frame_index_from=int(anchor_from["frame_index"]),
                    frame_index_to=int(anchor_to["frame_index"]),
                )
                pair_state_ids = sorted(
                    set(
                        zip(
                            pair_counts["from_family"],
                            pair_counts["from_id"],
                            strict=False,
                        )
                    ).union(
                        set(
                            zip(
                                pair_counts["to_family"],
                                pair_counts["to_id"],
                                strict=False,
                            )
                        )
                    ),
                    key=lambda item: (str(item[0]), int(item[1])),
                )
                pair_matrix, pair_labels = build_transition_matrix(
                    pair_counts,
                    state_ids=[
                        (str(family), int(motif_id))
                        for family, motif_id in pair_state_ids
                    ],
                )
                flow_path = transitions_dir / f"transition_snapshot_flow_{pair_idx:02d}.png"
                save_transition_flow_plot(
                    pair_matrix,
                    flow_path,
                    title=(
                        "Motif flow | "
                        f"{anchor_from['label']} -> {anchor_to['label']}"
                    ),
                    row_labels=pair_labels,
                    col_labels=pair_labels,
                    mute_diagonal=True,
                    min_draw_fraction=0.001,
                )
                transition_snapshot_flow_records.append(
                    {
                        "pair_index": int(pair_idx),
                        "frame_index_from": int(anchor_from["frame_index"]),
                        "frame_index_to": int(anchor_to["frame_index"]),
                        "timestep_from": (
                            None
                            if anchor_from["timestep"] is None
                            else int(anchor_from["timestep"])
                        ),
                        "timestep_to": (
                            None
                            if anchor_to["timestep"] is None
                            else int(anchor_to["timestep"])
                        ),
                        "label_from": str(anchor_from["label"]),
                        "label_to": str(anchor_to["label"]),
                        "count_total": int(pair_counts["count"].sum()),
                        "artifact": str(flow_path.relative_to(dynamic_dir)),
                    }
                )
            transition_snapshot_flow_index_path = (
                transitions_dir / "transition_snapshot_flow_index.csv"
            )
            _write_csv(
                pd.DataFrame(transition_snapshot_flow_records),
                transition_snapshot_flow_index_path,
            )

    proportion_plot_path = temporal_dir / "motif_proportion_area.png"
    plot_motif_proportion_area(frame_proportions_df, proportion_plot_path)

    if not dwell_df.empty and bool(getattr(getattr(settings, "render", None), "heatmaps", True)):
        dwell_hist_path = dwell_dir / "dwell_histograms.png"
        survival_plot_path = dwell_dir / "survival_curves.png"
        plot_dwell_histograms(dwell_df, dwell_hist_path)
        plot_survival_curves(survival_df, survival_plot_path)

    if not revisit_df.empty and bool(getattr(getattr(settings, "render", None), "heatmaps", True)):
        recurrence_plot_path = recurrence_dir / "recurrence_heatmap.png"
        plot_recurrence_heatmap(revisit_df, recurrence_plot_path)

    if bool(getattr(getattr(settings, "render", None), "event_gallery", True)):
        bridge_events = build_bridge_event_windows(
            sample_df,
            window_radius=2,
            max_events=12,
        )
        if bridge_events:
            bridge_gallery_path = temporal_dir / "bridge_event_gallery.png"
            plot_bridge_event_gallery(bridge_events, bridge_gallery_path)

    motif_projection_path = temporal_dir / "motif_umap.png"
    hazard_column_candidates = sorted(
        [column for column in sample_df.columns if column.startswith("hazard_probs_lag_")]
    )
    projection_2d = plot_motif_latent_2d(
        bundle.inv_latents if bundle.inv_latents is not None else np.empty((0, 2), dtype=np.float32),
        stable_ids=bundle.stable_ids,
        bridge_active=bundle.bridge_active,
        hazard_probs=(
            sample_df[hazard_column_candidates[0]].to_numpy(dtype=np.float32)
            if hazard_column_candidates
            else None
        ),
        out_file=motif_projection_path,
    )

    if not hazard_calibration_df.empty:
        hazard_curve_path = dwell_dir / "hazard_calibration.png"
        plot_hazard_calibration_curve(hazard_calibration_df, hazard_curve_path)

    if not field_gain_df.empty:
        neighbor_heatmap_path = field_dir / "neighbor_influence_heatmap.png"
        plot_neighbor_influence_heatmap(field_gain_df, neighbor_heatmap_path)

    representative_rows: list[dict[str, Any]] = []
    if bool(getattr(getattr(settings, "render", None), "representatives", True)) and projection_2d.size > 0:
        stable_ids_to_render = sorted(int(v) for v in np.unique(bundle.stable_ids[bundle.stable_ids >= 0]))
        for stable_id in stable_ids_to_render:
            selected = _select_representatives(
                sample_df,
                motif_family="stable",
                motif_id=int(stable_id),
                max_samples=int(getattr(settings, "representative_samples_per_motif", 12)),
            )
            if not selected:
                continue
            out_path = representatives_dir / f"stable_motif_{int(stable_id)}.png"
            plot_representative_panel(
                projection_2d,
                sample_df,
                motif_family="stable",
                motif_id=int(stable_id),
                representative_sample_indices=selected,
                out_file=out_path,
            )
            for sample_index in selected:
                row = sample_df.loc[sample_df["sample_index"] == int(sample_index)].iloc[0]
                representative_rows.append(
                    {
                        "motif_family": "stable",
                        "motif_id": int(stable_id),
                        "sample_index": int(sample_index),
                        "source_path": str(row["source_path"]),
                        "center_atom_id": int(row["center_atom_id"]),
                        "frame_index": int(row["frame_index"]),
                        "timestep": int(row["timestep"]),
                        "confidence": float(row["stable_confidence"]),
                        "artifact": str(out_path.relative_to(dynamic_dir)),
                    }
                )
        if bundle.bridge_ids is not None:
            bridge_ids_to_render = sorted(
                int(v) for v in np.unique(bundle.bridge_ids[bundle.bridge_active & (bundle.bridge_ids >= 0)])
            )
            for bridge_id in bridge_ids_to_render:
                selected = _select_representatives(
                    sample_df,
                    motif_family="bridge",
                    motif_id=int(bridge_id),
                    max_samples=int(getattr(settings, "representative_samples_per_motif", 12)),
                )
                if not selected:
                    continue
                out_path = representatives_dir / f"bridge_motif_{int(bridge_id)}.png"
                plot_representative_panel(
                    projection_2d,
                    sample_df,
                    motif_family="bridge",
                    motif_id=int(bridge_id),
                    representative_sample_indices=selected,
                    out_file=out_path,
                )
                for sample_index in selected:
                    row = sample_df.loc[sample_df["sample_index"] == int(sample_index)].iloc[0]
                    representative_rows.append(
                        {
                            "motif_family": "bridge",
                            "motif_id": int(bridge_id),
                            "sample_index": int(sample_index),
                            "source_path": str(row["source_path"]),
                            "center_atom_id": int(row["center_atom_id"]),
                            "frame_index": int(row["frame_index"]),
                            "timestep": int(row["timestep"]),
                            "confidence": float(row["bridge_confidence"]),
                            "artifact": str(out_path.relative_to(dynamic_dir)),
                        }
                    )
        if representative_rows:
            representative_index_path = representatives_dir / "representative_index.csv"
            _write_csv(pd.DataFrame(representative_rows), representative_index_path)
            dynamic_metrics["artifacts"]["representative_index_csv"] = _path_relative_to(out_dir, representative_index_path)

    if transition_matrix_path is not None:
        dynamic_metrics["artifacts"]["transition_matrix_png"] = _path_relative_to(out_dir, transition_matrix_path)
    if transition_snapshot_flow_index_path is not None:
        dynamic_metrics["artifacts"]["transition_snapshot_flow_index_csv"] = _path_relative_to(
            out_dir,
            transition_snapshot_flow_index_path,
        )
    for flow_record in transition_snapshot_flow_records:
        pair_idx = int(flow_record["pair_index"])
        dynamic_metrics["artifacts"][
            f"transition_snapshot_flow_{pair_idx:02d}_png"
        ] = str(flow_record["artifact"])
    if transition_snapshot_flow_records:
        dynamic_metrics["transition_snapshot_flows"] = transition_snapshot_flow_records
    if dwell_hist_path is not None:
        dynamic_metrics["artifacts"]["dwell_histograms_png"] = _path_relative_to(out_dir, dwell_hist_path)
    if survival_plot_path is not None:
        dynamic_metrics["artifacts"]["survival_curves_png"] = _path_relative_to(out_dir, survival_plot_path)
    if recurrence_plot_path is not None:
        dynamic_metrics["artifacts"]["recurrence_heatmap_png"] = _path_relative_to(out_dir, recurrence_plot_path)
    if bridge_gallery_path is not None:
        dynamic_metrics["artifacts"]["bridge_event_gallery_png"] = _path_relative_to(out_dir, bridge_gallery_path)
    if motif_projection_path is not None:
        dynamic_metrics["artifacts"]["motif_umap_png"] = _path_relative_to(out_dir, motif_projection_path)
    if hazard_curve_path is not None:
        dynamic_metrics["artifacts"]["hazard_calibration_png"] = _path_relative_to(out_dir, hazard_curve_path)
    if neighbor_heatmap_path is not None:
        dynamic_metrics["artifacts"]["neighbor_influence_heatmap_png"] = _path_relative_to(out_dir, neighbor_heatmap_path)

    summary_path = dynamic_dir / "summary.md"
    _build_summary_markdown(
        out_file=summary_path,
        dynamic_dir=dynamic_dir,
        metrics=dynamic_metrics,
        artifact_paths={key: value for key, value in dynamic_metrics["artifacts"].items() if value is not None},
        warnings_list=warnings_list,
    )
    dynamic_metrics["artifacts"]["summary_markdown"] = _path_relative_to(out_dir, summary_path)
    return dynamic_metrics


__all__ = [
    "AssignmentBundle",
    "resolve_motif_assignments",
    "run_dynamic_motif_analysis",
]
