from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.training_methods.successor_vicreg.artifact import SuccessorEmbeddingsArtifact


def _load_label_grid(labels_path: str | Path, *, label_key: str) -> np.ndarray:
    resolved = Path(labels_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Label file does not exist: {resolved}")
    with np.load(resolved) as payload:
        if label_key not in payload:
            raise KeyError(
                f"Label key {label_key!r} is not present in {resolved}. "
                f"Available keys={sorted(payload.files)}."
            )
        labels = np.asarray(payload[label_key], dtype=np.int64)
    if labels.ndim != 2:
        raise ValueError(
            f"Expected a label grid with shape (frames, atoms), got {tuple(labels.shape)}."
        )
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render selected trajectory frames colored by a cluster grid or state-label grid."
        )
    )
    parser.add_argument("artifact", help="Successor embedding artifact NPZ.")
    parser.add_argument("labels", help="NPZ containing a (frames, atoms) label grid.")
    parser.add_argument(
        "--label-key",
        default="successor",
        help="Key inside the label NPZ to render.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults next to the labels file.",
    )
    parser.add_argument(
        "--snapshot-count",
        type=int,
        default=6,
        help="Number of evenly spaced frames to render.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = SuccessorEmbeddingsArtifact.load(args.artifact)
    labels = _load_label_grid(args.labels, label_key=str(args.label_key))
    if labels.shape != artifact.invariant_embeddings.shape[:2]:
        raise ValueError(
            "Label grid shape does not match the artifact frame/atom grid. "
            f"labels.shape={tuple(labels.shape)}, expected={artifact.invariant_embeddings.shape[:2]}."
        )

    output_path = (
        Path(args.labels).expanduser().resolve().with_name(f"{args.label_key}_selected_frames.png")
        if args.output is None
        else Path(args.output).expanduser().resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_count = int(labels.shape[0])
    snapshot_count = min(int(args.snapshot_count), frame_count)
    selected_frames = np.linspace(0, frame_count - 1, num=snapshot_count, dtype=np.int64)
    ncols = min(3, snapshot_count)
    nrows = int(np.ceil(snapshot_count / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.8 * ncols, 4.6 * nrows), dpi=180)
    axes_arr = np.atleast_1d(axes).reshape(-1)
    cmap = plt.get_cmap("tab20")
    for ax, frame_idx in zip(axes_arr, selected_frames.tolist(), strict=False):
        frame_coords = artifact.center_positions[int(frame_idx)]
        frame_labels = labels[int(frame_idx)]
        ax.scatter(
            frame_coords[:, 0],
            frame_coords[:, 1],
            s=8,
            c=cmap(frame_labels % 20),
            alpha=0.85,
            linewidths=0,
        )
        ax.set_title(
            f"{args.label_key} | frame={frame_idx} | timestep={int(artifact.timesteps[int(frame_idx)])}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
    for ax in axes_arr[snapshot_count:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
