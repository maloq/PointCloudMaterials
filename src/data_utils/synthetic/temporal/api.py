from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

from .config import TemporalBenchmarkConfig, load_temporal_config
from .dynamics import build_site_layout, simulate_latent_trajectories
from .graph import TransitionGraph
from .neighborhoods import build_neighborhood_trajectory_pack
from .rendering import FrameRenderer
from .templates import TemplateLibrary
from .validation import validate_temporal_dataset
from .visualization import generate_temporal_visualizations
from .writer import TemporalDatasetWriteResult, TemporalDatasetWriter, result_from_paths


def generate_temporal_dataset(
    config_or_path: TemporalBenchmarkConfig | str | Path,
    *,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    validate: bool = True,
    progress: bool = True,
) -> TemporalDatasetWriteResult:
    started_at = time.perf_counter()
    config = _resolve_config(config_or_path, seed=seed, output_dir=output_dir)
    _emit_progress(progress, f"[temporal] Output directory: {config.output.output_dir}")
    if config.output.frame_storage == "frame_dirs" and not config.rendering.save_frame_dirs:
        raise ValueError(
            "rendering.save_frame_dirs=false is not supported in temporal v1 because frame-wise atom files "
            "are part of the benchmark contract."
        )
    if config.output.frame_storage not in {"frame_dirs", "single_chunk_npz"}:
        raise ValueError(
            "output.frame_storage must be either 'frame_dirs' or 'single_chunk_npz', "
            f"got {config.output.frame_storage!r}."
        )
    if not config.trajectories.save_combined_npz:
        raise ValueError(
            "trajectories.save_combined_npz=false is not supported in temporal v1. "
            "The writer currently emits a combined trajectory pack by design."
        )
    _emit_progress(progress, "[temporal] Step 1/6: Building site layout and latent trajectories")
    rng = np.random.default_rng(int(config.seed))
    graph = TransitionGraph(config.states, config.transitions, config.dynamics.primary_path)
    layout = build_site_layout(config)
    latent = simulate_latent_trajectories(config, graph, layout, rng)
    _emit_progress(
        progress,
        f"[temporal] Step 1/6 complete in {_format_seconds(time.perf_counter() - started_at)}",
    )
    _emit_progress(progress, "[temporal] Step 2/6: Initializing renderer and preparing output")
    templates = TemplateLibrary(config.domain, config.rendering, config.states, seed=int(config.seed))
    renderer = FrameRenderer(config, graph, templates, layout, latent)
    writer = TemporalDatasetWriter(config, graph)
    writer.prepare()
    writer.write_static_artifacts(layout)

    local_points_by_frame = np.empty(
        (
            config.time.num_frames,
            config.domain.site_count,
            config.domain.atoms_per_site,
            3,
        ),
        dtype=np.float32,
    )
    frame_atom_counts: list[int] = []
    frame_stage_started_at = time.perf_counter()
    report_interval = _frame_progress_interval(config.time.num_frames)
    _emit_progress(progress, "[temporal] Step 3/6: Rendering and writing frames")
    io_executor = ThreadPoolExecutor(max_workers=1)
    pending_write_future = None
    try:
        for frame_idx in range(config.time.num_frames):
            frame = renderer.render_frame(frame_idx)
            local_points_by_frame[frame_idx] = frame.local_points
            frame_atom_counts.append(int(frame.atoms.shape[0]))
            # Wait for previous frame's write to finish before submitting next
            if pending_write_future is not None:
                pending_write_future.result()
            # Submit write to background thread so next render can overlap
            pending_write_future = io_executor.submit(writer.write_frame, frame)
            if progress and _should_report_frame_progress(frame_idx, config.time.num_frames, report_interval):
                completed = frame_idx + 1
                elapsed = time.perf_counter() - frame_stage_started_at
                avg_time = elapsed / max(completed, 1)
                remaining = max(config.time.num_frames - completed, 0)
                eta = avg_time * remaining
                _emit_progress(
                    True,
                    (
                        f"[temporal]   frames {completed}/{config.time.num_frames} "
                        f"({100.0 * completed / config.time.num_frames:.1f}%) "
                        f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta)}"
                    ),
                )
        # Drain the last write
        if pending_write_future is not None:
            pending_write_future.result()
    finally:
        io_executor.shutdown(wait=True)
        renderer.close()
    writer.finalize_frames()

    _emit_progress(
        progress,
        f"[temporal] Step 3/6 complete in {_format_seconds(time.perf_counter() - frame_stage_started_at)}",
    )
    _emit_progress(progress, "[temporal] Step 4/6: Writing latent trajectories and neighborhoods")
    writer.write_latent(latent)
    neighborhood_pack = build_neighborhood_trajectory_pack(
        config=config,
        graph=graph,
        layout=layout,
        latent=latent,
        local_points_by_frame=local_points_by_frame,
    )
    writer.write_neighborhoods(neighborhood_pack)
    _emit_progress(progress, "[temporal] Step 4/6 complete")

    # Validation stays loud even when callers pass validate=False because this
    # benchmark is intended for research use and should not silently emit
    # inconsistent artifacts.
    _emit_progress(progress, "[temporal] Step 5/6: Validating dataset")
    validation_summary = validate_temporal_dataset(
        config=config,
        graph=graph,
        layout=layout,
        latent=latent,
        local_points_by_frame=local_points_by_frame,
        frame_atom_counts=frame_atom_counts,
    )
    validation_summary_path = writer.write_validation(validation_summary)
    manifest_path = writer.write_manifest(layout=layout, latent=latent, validation=validation_summary)
    _emit_progress(progress, "[temporal] Step 5/6 complete")
    visualization_result = None
    if config.visualization.enabled:
        _emit_progress(progress, "[temporal] Step 6/6: Generating visualizations")
        visualization_result = generate_temporal_visualizations(
            writer.output_dir,
            visualization_config=config.visualization,
        )
        _emit_progress(progress, "[temporal] Step 6/6 complete")
    else:
        _emit_progress(progress, "[temporal] Step 6/6: Visualization disabled")
    _emit_progress(
        progress,
        f"[temporal] Dataset generation finished in {_format_seconds(time.perf_counter() - started_at)}",
    )
    return result_from_paths(
        output_dir=writer.output_dir,
        frame_dirs=writer.frame_dirs,
        frame_chunk_path=writer.frame_chunk_path if config.output.frame_storage == "single_chunk_npz" else None,
        manifest_path=manifest_path,
        validation_summary_path=validation_summary_path,
        visualization_dir=visualization_result.output_dir if visualization_result is not None else None,
        visualization_manifest_path=(
            visualization_result.manifest_path if visualization_result is not None else None
        ),
    )


def _resolve_config(
    config_or_path: TemporalBenchmarkConfig | str | Path,
    *,
    seed: int | None,
    output_dir: str | Path | None,
) -> TemporalBenchmarkConfig:
    config = config_or_path if isinstance(config_or_path, TemporalBenchmarkConfig) else load_temporal_config(config_or_path)
    if seed is not None:
        config = replace(config, seed=int(seed))
    if output_dir is not None:
        config = replace(config, output=replace(config.output, output_dir=Path(output_dir)))
    return config


def _emit_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _frame_progress_interval(num_frames: int) -> int:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")
    return 1 if num_frames <= 20 else max(1, int(np.ceil(num_frames / 20.0)))


def _should_report_frame_progress(frame_idx: int, num_frames: int, report_interval: int) -> bool:
    completed = frame_idx + 1
    return completed == 1 or completed == num_frames or completed % report_interval == 0


def _format_seconds(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    if total_seconds < 60.0:
        return f"{total_seconds:.1f}s"
    minutes, sec = divmod(int(round(total_seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"
