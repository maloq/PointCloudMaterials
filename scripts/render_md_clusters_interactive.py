import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

from src.vis_tools.md_cluster_plot import render_interactive_md_clusters


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render interactive MD cluster plot from an analysis directory."
    )
    parser.add_argument(
        "analysis_dir",
        type=str,
        help="Path to analysis directory containing local_structure_coords_clusters.(npz|csv).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output HTML file (default: <analysis_dir>/md_space_clusters.html).",
    )
    parser.add_argument(
        "--palette",
        type=str,
        default="Set3",
        help="Plotly qualitative palette name (default: Set3).",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=None,
        help="Optional cap on points for interactive plot.",
    )
    parser.add_argument(
        "--marker_size",
        type=float,
        default=3.0,
        help="Marker size (default: 3.0).",
    )
    parser.add_argument(
        "--marker_line_width",
        type=float,
        default=0.0,
        help="Marker outline width in pixels (default: 0.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    analysis_dir = Path(args.analysis_dir)
    out_file = Path(args.out) if args.out else None
    output = render_interactive_md_clusters(
        analysis_dir,
        out_file=out_file,
        palette=args.palette,
        max_points=args.max_points,
        marker_size=args.marker_size,
        marker_line_width=args.marker_line_width,
    )
    print(f"Saved interactive plot to {output}")


if __name__ == "__main__":
    main()
