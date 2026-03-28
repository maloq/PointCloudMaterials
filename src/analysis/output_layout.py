from __future__ import annotations

from pathlib import Path


def snapshot_outputs_root(out_dir: Path | str) -> Path:
    return Path(out_dir) / "snapshots"


def snapshot_output_dir(out_dir: Path | str, snapshot_name: str) -> Path:
    return snapshot_outputs_root(out_dir) / str(snapshot_name)


def snapshot_md_space_dir(out_dir: Path | str, snapshot_name: str) -> Path:
    return snapshot_output_dir(out_dir, snapshot_name) / "md_space"


def snapshot_figure_set_dir(
    out_dir: Path | str,
    snapshot_name: str,
    *,
    k_value: int,
) -> Path:
    return snapshot_output_dir(out_dir, snapshot_name) / f"figure_set_k{int(k_value)}"


def snapshot_raytrace_gallery_root(out_dir: Path | str, *, k_value: int) -> Path:
    return snapshot_outputs_root(out_dir) / "_galleries_by_view" / f"figure_set_k{int(k_value)}"


def real_md_outputs_root(out_dir: Path | str) -> Path:
    return Path(out_dir) / "real_md"
