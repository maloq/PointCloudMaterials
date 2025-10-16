#!/usr/bin/env python3
"""Visualize synthetic scenes using matplotlib or open3d if available."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from src.data_utils.synthetic import (
    BASELINE_PRESET,
    SyntheticPointCloudDataset,
    generate_scene,
)

try:  # Matplotlib for static plots
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore

try:  # optional open3d support for interactive viewing
    import open3d as o3d
except ImportError:  # pragma: no cover - optional dependency
    o3d = None  # type: ignore


def _plot_matplotlib(
    points: np.ndarray,
    point_phase_ids: np.ndarray,
    centers: np.ndarray,
    env_phase_labels: np.ndarray,
    phase_name_map: Sequence[str] | dict[int, str],
    *,
    save_path: Path | None = None,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for static visualization")
    fig = plt.figure(figsize=(10, 6))
    ax_pc = fig.add_subplot(1, 2, 1, projection="3d")
    scatter = ax_pc.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=point_phase_ids,
        cmap="tab10",
        s=6,
        alpha=0.7,
    )
    ax_pc.set_title("Environment point clouds")
    ax_pc.set_xlabel("x")
    ax_pc.set_ylabel("y")
    ax_pc.set_zlabel("z")
    handles, labels = scatter.legend_elements()
    lookup = {int(k): v for k, v in phase_name_map.items()} if isinstance(phase_name_map, dict) else {
        idx: name for idx, name in enumerate(phase_name_map)
    }
    phase_labels = []
    for lbl in labels:
        try:
            pid = int(float(lbl))
        except ValueError:
            pid = lbl  # fallback to raw label
        phase_labels.append(lookup.get(pid, f"phase {pid}"))
    legend1 = ax_pc.legend(handles, phase_labels, title="Phase", loc="upper left")
    ax_pc.add_artist(legend1)

    ax_phase = fig.add_subplot(1, 2, 2, projection="3d")
    unique_phases = np.unique(env_phase_labels)
    if isinstance(phase_name_map, dict):
        lookup = {int(k): v for k, v in phase_name_map.items()}
    else:
        lookup = {idx: name for idx, name in enumerate(phase_name_map)}
    for pid in unique_phases:
        env_mask = env_phase_labels == pid
        label = lookup.get(int(pid), f"phase {int(pid)}")
        ax_phase.scatter(
            centers[env_mask, 0],
            centers[env_mask, 1],
            centers[env_mask, 2],
            label=label,
            s=24,
            alpha=0.8,
        )
    ax_phase.legend()
    ax_phase.set_title("Environment centers by phase")
    ax_phase.set_xlabel("x")
    ax_phase.set_ylabel("y")
    ax_phase.set_zlabel("z")
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def _plot_open3d(batches: Iterable[np.ndarray], labels: Sequence[int], phases: Sequence[str]) -> None:
    if o3d is None:
        raise RuntimeError("open3d is not installed")
    geometries = []
    cmap = plt.cm.get_cmap("tab10") if plt is not None else None
    for idx, (points, label) in enumerate(zip(batches, labels)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if cmap is not None:
            color = np.array(cmap(label % 10)[:3])
        else:
            rng = np.random.default_rng(label)
            color = rng.uniform(0.1, 0.9, size=3)
        pcd.paint_uniform_color(color)
        pcd.translate((idx * 0.2, 0, 0))
        geometries.append(pcd)
    o3d.visualization.draw_geometries(geometries)


def visualize_scene(
    scene,
    *,
    limit: int = 16,
    backend: str = "matplotlib",
    save_path: Path | None = None,
) -> None:
    indices = np.arange(scene.num_environments)
    if limit > 0:
        indices = indices[:limit]
    points = scene.points[indices]
    phase_labels = scene.phase_labels[indices]
    centers = scene.centers[indices]

    if backend == "matplotlib":
        merged = points.reshape(-1, 3)
        phase_ids = np.repeat(phase_labels, scene.points_per_environment)
        if save_path is not None and plt is not None and "agg" not in plt.get_backend().lower():
            plt.switch_backend("Agg")
        phase_name_map = {int(v): str(k) for k, v in scene.phase_ids.items()}
        _plot_matplotlib(
            merged,
            phase_ids,
            centers,
            phase_labels,
            phase_name_map,
            save_path=save_path,
        )
    elif backend == "open3d":
        if save_path is not None:
            raise ValueError("Saving PNG is only supported with the matplotlib backend")
        batches = [points[i] for i in range(points.shape[0])]
        _plot_open3d(batches, phase_labels, list(scene.phase_ids.keys()))
    else:
        raise ValueError(f"Unknown backend {backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize synthetic dataset scenes")
    parser.add_argument("--load", type=str, default=None, help="Path to existing .npz scene to load")
    parser.add_argument("--num-env", type=int, default=1024, help="Number of environments to sample if generating")
    parser.add_argument("--limit", type=int, default=1024, help="Number of environments to display")
    parser.add_argument("--backend", choices=["matplotlib", "open3d"], default="matplotlib")
    parser.add_argument("--preset", type=str, default="baseline", help="Preset to use when generating from scratch")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the visualization as a PNG")
    return parser.parse_args()


def load_or_generate(args: argparse.Namespace):
    if args.load:
        from src.data_utils.synthetic import load_scene

        return load_scene(args.load)

    from src.data_utils.synthetic import (
        BASELINE_PRESET,
        imbalanced_phase_preset,
        temporal_growth_sequence,
        voronoi_anisotropic_preset,
    )

    preset_name = args.preset.lower()
    if preset_name == "baseline":
        cfg = BASELINE_PRESET
    elif preset_name == "imbalanced":
        cfg = imbalanced_phase_preset()
    elif preset_name == "voronoi":
        cfg = voronoi_anisotropic_preset()
    else:
        raise ValueError(f"Unknown preset {args.preset}")
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.validate()
    return generate_scene(cfg, args.num_env)


def main() -> None:
    args = parse_args()
    scene = load_or_generate(args)
    print(
        f"Scene contains {scene.num_environments} environments with {scene.points_per_environment} points each."
    )
    save_path = 'test.png'
    visualize_scene(scene, limit=args.limit, backend=args.backend, save_path=save_path)
    print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    main()
