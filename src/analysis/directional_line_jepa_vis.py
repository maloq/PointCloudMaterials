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

_REQUIRED_VISUALIZATION_FIELDS = {
    "coords",
    "source_slots",
    "source_names_by_slot",
    "source_paths_by_slot",
    "analysis_sample_indices",
    "atom_ids",
    "directions",
    "prediction_cosine_error",
    "context_coords",
    "context_source_slots",
    "context_atom_ids",
    "context_analysis_sample_indices",
    "context_evaluated_mask",
    "mean_prediction_cosine_error",
    "relative_directional_variation",
}


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
    stem = Path(value).stem
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.")
    if not normalized:
        raise ValueError(
            "Directional visualization source names must contain at least one filename-safe "
            f"character, got {value!r}."
        )
    return normalized


def _validate_visualization_arrays(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    resolved = dict(arrays)
    missing = sorted(_REQUIRED_VISUALIZATION_FIELDS - set(resolved))
    if missing:
        raise KeyError(
            "Directional visualization archive does not match the current "
            f"directional_line_jepa schema. Missing fields: {missing}."
        )

    coords = resolved["coords"]
    directions = resolved["directions"]
    context_coords = resolved["context_coords"]
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"Directional visualization coords must have shape (N, 3), got {coords.shape}."
        )
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError(
            "Directional visualization directions must have shape (K, 3), "
            f"got {directions.shape}."
        )
    if context_coords.ndim != 2 or context_coords.shape[1] != 3:
        raise ValueError(
            "Directional visualization context_coords must have shape (M, 3), "
            f"got {context_coords.shape}."
        )

    atom_count = len(coords)
    direction_count = len(directions)
    context_count = len(context_coords)
    expected_shapes = {
        "source_slots": (atom_count,),
        "analysis_sample_indices": (atom_count,),
        "atom_ids": (atom_count,),
        "prediction_cosine_error": (atom_count, direction_count),
        "mean_prediction_cosine_error": (atom_count,),
        "relative_directional_variation": (atom_count,),
        "context_source_slots": (context_count,),
        "context_atom_ids": (context_count,),
        "context_analysis_sample_indices": (context_count,),
        "context_evaluated_mask": (context_count,),
    }
    for name, expected_shape in expected_shapes.items():
        if resolved[name].shape != expected_shape:
            raise ValueError(
                f"Directional visualization field {name!r} must have shape "
                f"{expected_shape}, got {resolved[name].shape}."
            )

    names = resolved["source_names_by_slot"]
    paths = resolved["source_paths_by_slot"]
    if names.ndim != 1 or paths.shape != names.shape:
        raise ValueError(
            "Directional visualization source metadata must be parallel one-dimensional "
            f"arrays, got names={names.shape}, paths={paths.shape}."
        )
    source_count = len(names)
    for field in ("source_slots", "context_source_slots"):
        slots = resolved[field]
        if np.any(slots < 0) or np.any(slots >= source_count):
            raise IndexError(
                f"Directional visualization field {field!r} contains source slots outside "
                f"[0, {source_count}), values={np.unique(slots).tolist()}."
            )
    if not np.isfinite(directions).all() or np.any(
        np.linalg.norm(directions, axis=1) == 0.0
    ):
        raise ValueError(
            "Directional visualization directions must be finite and non-zero."
        )
    if not np.isfinite(resolved["prediction_cosine_error"]).all():
        raise ValueError(
            "Directional visualization prediction_cosine_error contains non-finite values."
        )
    return resolved


def _load_visualization_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return _validate_visualization_arrays(arrays)


def _source_groups(
    arrays: Mapping[str, np.ndarray], *, context: bool = False
) -> list[tuple[int, str, str, np.ndarray]]:
    slot_field = "context_source_slots" if context else "source_slots"
    slots = arrays[slot_field]
    names = arrays["source_names_by_slot"]
    paths = arrays["source_paths_by_slot"]
    groups = []
    for slot_raw in dict.fromkeys(slots.tolist()):
        slot = int(slot_raw)
        groups.append((slot, str(names[slot]), str(paths[slot]), np.flatnonzero(slots == slot)))
    return groups


def _global_color_limits(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("Cannot visualize an empty directional metric.")
    if not np.isfinite(values).all():
        invalid_count = int(np.count_nonzero(~np.isfinite(values)))
        raise ValueError(
            "Cannot visualize a directional metric containing non-finite values. "
            f"shape={values.shape}, invalid_count={invalid_count}."
        )
    low, high = np.quantile(values, [0.01, 0.99])
    if high <= low:
        high = low + max(abs(float(low)) * 1.0e-6, 1.0e-8)
    return float(low), float(high)


def _render_spatial_map(
    arrays: Mapping[str, np.ndarray],
    *,
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
    coords = arrays["coords"][evaluated_indices]
    values = arrays[metric_name][evaluated_indices]
    customdata = np.column_stack(
        (
            arrays["analysis_sample_indices"][evaluated_indices],
            arrays["atom_ids"][evaluated_indices],
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

    context_mask = arrays["context_evaluated_mask"][context_indices]
    excluded_indices = context_indices[~context_mask]
    if len(excluded_indices):
        excluded_coords = arrays["context_coords"][excluded_indices]
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
    if len(values) <= 1:
        return np.zeros(values.shape, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / float(len(values) - 1)


def _representative_rows(
    arrays: Mapping[str, np.ndarray], indices: np.ndarray
) -> list[tuple[str, int]]:
    prediction = arrays["mean_prediction_cosine_error"][indices]
    variation = arrays["relative_directional_variation"][indices]
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
    directions = arrays["directions"]
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    azimuth = np.degrees(np.arctan2(directions[:, 1], directions[:, 0]))
    elevation = np.degrees(np.arcsin(np.clip(directions[:, 2], -1.0, 1.0)))
    profiles = arrays["prediction_cosine_error"]
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


def _remove_generated_outputs(out_dir: Path) -> None:
    for path in out_dir.glob("*/"):
        for filename in _GENERATED_FILENAMES:
            generated = path / filename
            if generated.exists():
                generated.unlink()


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
        else _validate_visualization_arrays(data)
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _remove_generated_outputs(out_dir)

    prediction_limits = _global_color_limits(arrays["mean_prediction_cosine_error"])
    sensitivity_limits = _global_color_limits(arrays["relative_directional_variation"])
    evaluated_groups = {slot: (name, path, indices) for slot, name, path, indices in _source_groups(arrays)}
    context_groups = {slot: indices for slot, _, _, indices in _source_groups(arrays, context=True)}

    artifacts: dict[str, str] = {}
    snapshot_records: list[dict[str, str | int]] = []
    used_output_names: set[str] = set()
    for slot, (name, _source_path, evaluated_indices) in evaluated_groups.items():
        output_name = _safe_output_name(name)
        if output_name in used_output_names:
            output_name = f"{output_name}_{slot:02d}"
        used_output_names.add(output_name)
        snapshot_dir = out_dir / output_name
        try:
            context_indices = context_groups[slot]
        except KeyError as exc:
            raise KeyError(
                "Directional visualization evaluated source slot has no context rows. "
                f"source_slot={slot}, source_name={name!r}."
            ) from exc
        excluded_count = int(
            np.sum(~arrays["context_evaluated_mask"][context_indices])
        )

        prediction_path = snapshot_dir / "prediction_error_space.html"
        _render_spatial_map(
            arrays,
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
            if not isinstance(summary, dict) or not isinstance(summary.get("artifacts"), dict):
                raise TypeError(
                    "Directional summary must be a JSON object containing an artifacts object. "
                    f"path={summary_path}."
                )
            summary_artifacts = summary["artifacts"]
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
    else:
        raise FileNotFoundError(
            "Directional visualization expected the summary produced alongside the NPZ, "
            f"but it does not exist: {summary_path}."
        )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
