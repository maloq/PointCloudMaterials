"""Clear per-snapshot views for directional Line-JEPA prediction analysis."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np


_DISCOVERY_METRICS = {
    "mean_prediction_cosine_error": (
        "Mean prediction cosine error",
        "Inferno",
    ),
    "relative_directional_variation": (
        "Relative directional variation",
        "Viridis",
    ),
}

_GENERATED_FILENAMES = (
    "prediction_error_space.html",
    "directional_sensitivity_space.html",
    "directional_profiles.html",
)

_LEGACY_FILENAMES = (
    "index.html",
    "directional_discovery_map_3d.html",
    "directional_signature_atlas.html",
    "directional_metrics_3d.html",
    "directional_residual_sweep_3d.html",
    "directional_max_error_vectors_3d.html",
    "directional_dense_atoms_3d.html",
    "directional_volume_3d.html",
)

_LEGACY_ARTIFACT_KEYS = (
    "interactive_3d_index",
    "interactive_3d_discovery_map",
    "interactive_directional_signature_atlas",
    "interactive_3d_metrics",
    "interactive_3d_direction_sweep",
    "interactive_3d_max_error_vectors",
    "interactive_3d_dense_atoms",
    "interactive_3d_volume",
)


def _plotly_modules():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "Directional visualization requires Plotly. Install it with `pip install plotly`."
        ) from exc
    return go, make_subplots


def _safe_output_name(value: str) -> str:
    stem = Path(str(value)).stem
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.")
    return normalized or "snapshot"


def _legacy_source_metadata(
    arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = np.asarray(arrays["source_paths"]).astype(str)
    names = np.asarray(arrays["source_names"]).astype(str)
    unique_paths = list(dict.fromkeys(paths.tolist()))
    path_to_slot = {path: slot for slot, path in enumerate(unique_paths)}
    slots = np.asarray([path_to_slot[path] for path in paths], dtype=np.int64)
    names_by_slot = np.asarray(
        [names[int(np.flatnonzero(paths == path)[0])] for path in unique_paths], dtype=str
    )
    return slots, names_by_slot, np.asarray(unique_paths, dtype=str)


def _ensure_visualization_arrays(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    resolved = {name: np.asarray(value) for name, value in arrays.items()}
    required = {"coords", "atom_ids", "analysis_sample_indices", "directions"}
    missing = sorted(required - set(resolved))
    if missing:
        raise KeyError(f"Directional visualization is missing required fields: {missing}.")

    if "prediction_cosine_error" not in resolved:
        if "reconstruction_cosine_error" not in resolved:
            raise KeyError(
                "Directional visualization requires prediction_cosine_error or the legacy "
                "reconstruction_cosine_error field."
            )
        resolved["prediction_cosine_error"] = np.clip(
            np.asarray(resolved["reconstruction_cosine_error"], dtype=np.float32),
            0.0,
            2.0,
        )

    if "source_names_by_slot" not in resolved:
        legacy_fields = {"source_names", "source_paths"}
        if not legacy_fields.issubset(resolved):
            raise KeyError(
                "Directional visualization needs compact source metadata or legacy "
                "source_names/source_paths arrays."
            )
        slots, names_by_slot, paths_by_slot = _legacy_source_metadata(resolved)
        resolved["source_slots"] = slots
        resolved["source_names_by_slot"] = names_by_slot
        resolved["source_paths_by_slot"] = paths_by_slot

    if "context_coords" not in resolved:
        resolved["context_coords"] = np.asarray(resolved["coords"], dtype=np.float32)
        resolved["context_source_slots"] = np.asarray(
            resolved["source_slots"], dtype=np.int64
        )
        resolved["context_atom_ids"] = np.asarray(resolved["atom_ids"], dtype=np.int64)
        resolved["context_analysis_sample_indices"] = np.asarray(
            resolved["analysis_sample_indices"], dtype=np.int64
        )
        resolved["context_evaluated_mask"] = np.ones(
            (len(resolved["coords"]),), dtype=bool
        )

    summary_fields = {
        "mean_error",
        "max_error",
        "std_error",
        "relative_directional_variation",
        "novelty_percentile",
        "anisotropy_percentile",
    }
    if not summary_fields.issubset(resolved):
        from .directional_line_jepa import compute_directional_error_summaries

        resolved.update(
            compute_directional_error_summaries(
                resolved["prediction_cosine_error"], resolved["directions"]
            )
        )
    resolved["mean_prediction_cosine_error"] = np.asarray(
        resolved.get("mean_prediction_cosine_error", resolved["mean_error"]),
        dtype=np.float32,
    )
    resolved["max_prediction_cosine_error"] = np.asarray(
        resolved.get("max_prediction_cosine_error", resolved["max_error"]),
        dtype=np.float32,
    )
    resolved["std_prediction_cosine_error"] = np.asarray(
        resolved.get("std_prediction_cosine_error", resolved["std_error"]),
        dtype=np.float32,
    )

    atom_count = len(resolved["coords"])
    direction_count = len(resolved["directions"])
    if resolved["prediction_cosine_error"].shape != (atom_count, direction_count):
        raise ValueError(
            "Directional prediction error shape does not match atoms and directions. "
            f"errors={resolved['prediction_cosine_error'].shape}, atoms={atom_count}, "
            f"directions={direction_count}."
        )
    return resolved


def _load_visualization_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return _ensure_visualization_arrays(arrays)


def _source_groups(
    arrays: Mapping[str, np.ndarray], *, context: bool = False
) -> list[tuple[int, str, str, np.ndarray]]:
    slot_field = "context_source_slots" if context else "source_slots"
    slots = np.asarray(arrays[slot_field], dtype=np.int64)
    names = np.asarray(arrays["source_names_by_slot"]).astype(str)
    paths = np.asarray(arrays["source_paths_by_slot"]).astype(str)
    groups = []
    for slot_raw in dict.fromkeys(slots.tolist()):
        slot = int(slot_raw)
        groups.append((slot, str(names[slot]), str(paths[slot]), np.flatnonzero(slots == slot)))
    return groups


def _global_color_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot visualize a metric containing no finite values.")
    low, high = np.quantile(finite, [0.01, 0.99])
    if high <= low:
        high = low + max(abs(float(low)) * 1.0e-6, 1.0e-8)
    return float(low), float(high)


def _render_spatial_map(
    arrays: Mapping[str, np.ndarray],
    *,
    source_slot: int,
    source_label: str,
    evaluated_indices: np.ndarray,
    context_indices: np.ndarray,
    metric_name: str,
    metric_label: str,
    colorscale: str,
    limits: tuple[float, float],
    out_file: Path,
    include_plotlyjs: bool | str,
) -> None:
    go, _ = _plotly_modules()
    coords = np.asarray(arrays["coords"], dtype=np.float32)[evaluated_indices]
    values = np.asarray(arrays[metric_name], dtype=np.float32)[evaluated_indices]
    customdata = np.column_stack(
        (
            np.asarray(arrays["analysis_sample_indices"])[evaluated_indices],
            np.asarray(arrays["atom_ids"])[evaluated_indices],
            values,
        )
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            name="evaluated atoms",
            customdata=customdata,
            marker={
                "size": 3.0,
                "opacity": 0.8,
                "color": values,
                "colorscale": colorscale,
                "cmin": limits[0],
                "cmax": limits[1],
                "colorbar": {"title": metric_label},
            },
            hovertemplate=(
                "atom=%{customdata[1]:.0f}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
                f"<br>{metric_label}=%{{customdata[2]:.5g}}<extra></extra>"
            ),
        )
    )

    context_mask = np.asarray(arrays["context_evaluated_mask"], dtype=bool)[context_indices]
    excluded_indices = context_indices[~context_mask]
    if len(excluded_indices):
        excluded_coords = np.asarray(arrays["context_coords"], dtype=np.float32)[
            excluded_indices
        ]
        fig.add_trace(
            go.Scatter3d(
                x=excluded_coords[:, 0],
                y=excluded_coords[:, 1],
                z=excluded_coords[:, 2],
                mode="markers",
                name=f"boundary excluded ({len(excluded_indices)})",
                marker={"size": 2.5, "opacity": 0.22, "color": "#9e9e9e"},
                hovertemplate=(
                    "Boundary point: no full directional ray"
                    "<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=(
            f"{source_label} — {metric_label} "
            f"(evaluated={len(evaluated_indices)}, excluded={len(excluded_indices)})"
        ),
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "cube",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        template=None,
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_file), include_plotlyjs=include_plotlyjs)


def _rank_percentiles(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return np.zeros(values.shape, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / float(len(values) - 1)


def _representative_rows(
    arrays: Mapping[str, np.ndarray], indices: np.ndarray
) -> list[tuple[str, int]]:
    prediction = np.asarray(arrays["mean_prediction_cosine_error"])[indices]
    variation = np.asarray(arrays["relative_directional_variation"])[indices]
    prediction_rank = _rank_percentiles(prediction)
    variation_rank = _rank_percentiles(variation)
    candidate_orders = (
        (
            "typical",
            np.argsort((prediction_rank - 0.5) ** 2 + (variation_rank - 0.5) ** 2),
        ),
        ("highest prediction error", np.argsort(prediction, kind="mergesort")[::-1]),
        ("strongest directional sensitivity", np.argsort(variation, kind="mergesort")[::-1]),
        (
            "high prediction + directionality",
            np.argsort(prediction_rank + variation_rank, kind="mergesort")[::-1],
        ),
    )
    representatives: list[tuple[str, int]] = []
    used: set[int] = set()
    for label, order in candidate_orders:
        for local_row_raw in order:
            row = int(indices[int(local_row_raw)])
            if row not in used:
                representatives.append((label, row))
                used.add(row)
                break
    if not representatives:
        raise ValueError("Directional profile viewer found no representative atoms.")
    return representatives


def _unit_sphere_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth = np.linspace(0.0, 2.0 * np.pi, 36)
    elevation = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 19)
    azimuth_grid, elevation_grid = np.meshgrid(azimuth, elevation)
    x = np.cos(elevation_grid) * np.cos(azimuth_grid)
    y = np.cos(elevation_grid) * np.sin(azimuth_grid)
    z = np.sin(elevation_grid)
    return x, y, z


def _render_directional_profiles(
    arrays: Mapping[str, np.ndarray],
    *,
    source_label: str,
    evaluated_indices: np.ndarray,
    prediction_limits: tuple[float, float],
    out_file: Path,
    include_plotlyjs: bool | str,
) -> None:
    go, make_subplots = _plotly_modules()
    directions = np.asarray(arrays["directions"], dtype=np.float64)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    azimuth = np.degrees(np.arctan2(directions[:, 1], directions[:, 0]))
    elevation = np.degrees(np.arcsin(np.clip(directions[:, 2], -1.0, 1.0)))
    profiles = np.asarray(arrays["prediction_cosine_error"], dtype=np.float32)
    representatives = _representative_rows(arrays, evaluated_indices)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        column_widths=[0.48, 0.52],
        horizontal_spacing=0.06,
        subplot_titles=("Direction angles", "Unit-direction sphere"),
    )
    sphere_x, sphere_y, sphere_z = _unit_sphere_surface()
    fig.add_trace(
        go.Surface(
            x=sphere_x,
            y=sphere_y,
            z=sphere_z,
            surfacecolor=np.zeros_like(sphere_x),
            colorscale=[[0.0, "#d9e2ec"], [1.0, "#d9e2ec"]],
            showscale=False,
            opacity=0.10,
            hoverinfo="skip",
            name="unit sphere",
        ),
        row=1,
        col=2,
    )
    trace_pairs: list[tuple[int, int]] = []
    for representative_index, (category, row) in enumerate(representatives):
        profile = profiles[row]
        visible = representative_index == 0
        custom = np.column_stack((directions, profile))
        angular_trace = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=azimuth,
                y=elevation,
                mode="markers",
                visible=visible,
                name=category,
                customdata=custom,
                marker={
                    "size": 9,
                    "color": profile,
                    "colorscale": "Inferno",
                    "cmin": prediction_limits[0],
                    "cmax": prediction_limits[1],
                    "showscale": False,
                },
                hovertemplate=(
                    "azimuth=%{x:.1f}°<br>elevation=%{y:.1f}°"
                    "<br>d=(%{customdata[0]:+.3f}, %{customdata[1]:+.3f}, "
                    "%{customdata[2]:+.3f})"
                    "<br>cosine error=%{customdata[3]:.5g}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        sphere_trace = len(fig.data)
        fig.add_trace(
            go.Scatter3d(
                x=directions[:, 0],
                y=directions[:, 1],
                z=directions[:, 2],
                mode="markers",
                visible=visible,
                name=category,
                showlegend=False,
                customdata=profile[:, None],
                marker={
                    "size": 5.5,
                    "color": profile,
                    "colorscale": "Inferno",
                    "cmin": prediction_limits[0],
                    "cmax": prediction_limits[1],
                    "colorbar": {"title": "Cosine error"},
                },
                hovertemplate=(
                    "d=(%{x:+.3f}, %{y:+.3f}, %{z:+.3f})"
                    "<br>cosine error=%{customdata[0]:.5g}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
        trace_pairs.append((angular_trace, sphere_trace))

    buttons = []
    for selected, (category, row) in enumerate(representatives):
        visibility = [True] + [False] * (2 * len(representatives))
        for trace_index in trace_pairs[selected]:
            visibility[trace_index] = True
        buttons.append(
            {
                "label": f"{category}: atom {int(arrays['atom_ids'][row])}",
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {
                        "title": (
                            f"{source_label} — 64 evaluated line directions; "
                            f"{category}, atom {int(arrays['atom_ids'][row])}"
                        )
                    },
                ],
            }
        )

    first_category, first_row = representatives[0]
    fig.update_xaxes(title_text="Azimuth (degrees)", range=[-180, 180], row=1, col=1)
    fig.update_yaxes(title_text="Elevation (degrees)", range=[-90, 90], row=1, col=1)
    fig.update_layout(
        title=(
            f"{source_label} — 64 evaluated line directions; {first_category}, "
            f"atom {int(arrays['atom_ids'][first_row])}"
        ),
        height=760,
        margin={"l": 55, "r": 75, "t": 100, "b": 50},
        scene={
            "xaxis_title": "dₓ",
            "yaxis_title": "dᵧ",
            "zaxis_title": "d_z",
            "xaxis": {"range": [-1.05, 1.05]},
            "yaxis": {"range": [-1.05, 1.05]},
            "zaxis": {"range": [-1.05, 1.05]},
            "aspectmode": "cube",
        },
        updatemenus=[
            {
                "buttons": buttons,
                "x": 0.0,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        template=None,
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_file), include_plotlyjs=include_plotlyjs)


def _remove_stale_outputs(out_dir: Path) -> None:
    for filename in _LEGACY_FILENAMES:
        path = out_dir / filename
        if path.exists():
            path.unlink()
    for path in out_dir.glob("*/"):
        for filename in _GENERATED_FILENAMES:
            generated = path / filename
            if generated.exists():
                generated.unlink()

    legacy_dir = out_dir.parent / "directional_line_jepa_3d"
    if legacy_dir != out_dir and legacy_dir.is_dir():
        for filename in _LEGACY_FILENAMES:
            path = legacy_dir / filename
            if path.exists():
                path.unlink()


def _write_index(
    out_file: Path,
    snapshot_records: list[dict[str, str | int]],
    *,
    direction_count: int,
) -> None:
    entries = []
    for record in snapshot_records:
        output_name = escape(str(record["output_name"]))
        label = escape(str(record["label"]))
        entries.append(
            "<li><b>"
            + label
            + "</b> — evaluated "
            + str(record["evaluated_count"])
            + ", boundary excluded "
            + str(record["excluded_count"])
            + "<br>"
            + f'<a href="{output_name}/prediction_error_space.html">Prediction error</a> · '
            + f'<a href="{output_name}/directional_sensitivity_space.html">Directional sensitivity</a> · '
            + f'<a href="{output_name}/directional_profiles.html">Direction profiles</a></li>'
        )
    out_file.write_text(
        """<!doctype html><meta charset="utf-8"><title>Directional Line-JEPA report</title>
<style>body{font:16px sans-serif;max-width:980px;margin:2.5rem auto;line-height:1.5}
li{margin:1rem 0}code{background:#eef2f6;padding:.1rem .25rem}</style>
<h1>Directional Line-JEPA report</h1>
<p>Each snapshot is shown separately in physical <code>(x,y,z)</code> space.</p>
<ul>
<li><b>Prediction error</b>: mean reconstruction cosine error over DIRECTION_COUNT sampled line
directions. Lower is better. This view contains prediction values only; directions are averaged.</li>
<li><b>Directional sensitivity</b>: standard deviation across directions divided by mean error.
Zero means the prediction error is independent of direction; larger values mean stronger dependence.</li>
<li><b>Direction profiles</b>: every colored point is one of the actual sampled unit directions.
The 2D axes are azimuth/elevation; the 3D axes are the unit-vector components. Point radius never
encodes error.</li>
<li><b>Boundary excluded</b>: the checkpoint geometry cannot construct a complete ray there, so
these atoms are gray and never receive fabricated values.</li>
</ul><h2>Snapshots</h2><ol>SNAPSHOT_ENTRIES</ol>""".replace(
            "DIRECTION_COUNT", str(direction_count)
        ).replace("SNAPSHOT_ENTRIES", "".join(entries)),
        encoding="utf-8",
    )


def render_directional_line_jepa_visualizations(
    data: Mapping[str, np.ndarray] | str | Path,
    out_dir: str | Path,
    *,
    include_plotlyjs: bool | str = "cdn",
) -> dict[str, str]:
    """Write simple prediction, sensitivity, and direction-profile views per snapshot."""
    arrays = (
        _load_visualization_arrays(Path(data))
        if isinstance(data, (str, Path))
        else _ensure_visualization_arrays(data)
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_outputs(out_dir)

    prediction_limits = _global_color_limits(arrays["mean_prediction_cosine_error"])
    sensitivity_limits = _global_color_limits(arrays["relative_directional_variation"])
    evaluated_groups = {slot: (name, path, indices) for slot, name, path, indices in _source_groups(arrays)}
    context_groups = {slot: indices for slot, _, _, indices in _source_groups(arrays, context=True)}

    artifacts: dict[str, str] = {}
    snapshot_records: list[dict[str, str | int]] = []
    used_output_names: set[str] = set()
    for slot, (name, source_path, evaluated_indices) in evaluated_groups.items():
        output_name = _safe_output_name(name or Path(source_path).name)
        if output_name in used_output_names:
            output_name = f"{output_name}_{slot:02d}"
        used_output_names.add(output_name)
        snapshot_dir = out_dir / output_name
        context_indices = context_groups.get(slot, np.empty((0,), dtype=np.int64))
        excluded_count = int(
            np.sum(~np.asarray(arrays["context_evaluated_mask"], dtype=bool)[context_indices])
        )

        prediction_path = snapshot_dir / "prediction_error_space.html"
        _render_spatial_map(
            arrays,
            source_slot=slot,
            source_label=name,
            evaluated_indices=evaluated_indices,
            context_indices=context_indices,
            metric_name="mean_prediction_cosine_error",
            metric_label="Mean prediction cosine error",
            colorscale="Inferno",
            limits=prediction_limits,
            out_file=prediction_path,
            include_plotlyjs=include_plotlyjs,
        )
        sensitivity_path = snapshot_dir / "directional_sensitivity_space.html"
        _render_spatial_map(
            arrays,
            source_slot=slot,
            source_label=name,
            evaluated_indices=evaluated_indices,
            context_indices=context_indices,
            metric_name="relative_directional_variation",
            metric_label="Relative directional variation",
            colorscale="Viridis",
            limits=sensitivity_limits,
            out_file=sensitivity_path,
            include_plotlyjs=include_plotlyjs,
        )
        profiles_path = snapshot_dir / "directional_profiles.html"
        _render_directional_profiles(
            arrays,
            source_label=name,
            evaluated_indices=evaluated_indices,
            prediction_limits=prediction_limits,
            out_file=profiles_path,
            include_plotlyjs=include_plotlyjs,
        )
        key_prefix = f"snapshot_{output_name}"
        artifacts[f"{key_prefix}_prediction_error"] = str(prediction_path)
        artifacts[f"{key_prefix}_directional_sensitivity"] = str(sensitivity_path)
        artifacts[f"{key_prefix}_directional_profiles"] = str(profiles_path)
        snapshot_records.append(
            {
                "label": name,
                "output_name": output_name,
                "evaluated_count": int(len(evaluated_indices)),
                "excluded_count": excluded_count,
            }
        )

    index_path = out_dir / "index.html"
    _write_index(
        index_path,
        snapshot_records,
        direction_count=int(len(arrays["directions"])),
    )
    artifacts = {"interactive_report_index": str(index_path), **artifacts}
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Render directional Line-JEPA report")
    parser.add_argument("npz", type=Path, help="Path to directional_line_jepa.npz")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.npz.parent / "directional_line_jepa_report"
    artifacts = render_directional_line_jepa_visualizations(args.npz, output_dir)

    summary_path = args.npz.parent / "directional_line_jepa_summary.json"
    if summary_path.exists():
        artifact_paths = [Path(value).resolve() for value in artifacts.values()]
        summary_parent = args.npz.parent.resolve()
        if all(path.is_relative_to(summary_parent) for path in artifact_paths):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_artifacts = summary.setdefault("artifacts", {})
            for key in _LEGACY_ARTIFACT_KEYS:
                summary_artifacts.pop(key, None)
            summary_artifacts.update(
                {
                    key: str(path.relative_to(summary_parent))
                    for key, path in zip(artifacts, artifact_paths, strict=True)
                }
            )
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        else:
            print(
                "Summary was not updated because --output-dir is outside the directional NPZ "
                f"directory: output_dir={output_dir.resolve()}, npz_dir={summary_parent}."
            )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
