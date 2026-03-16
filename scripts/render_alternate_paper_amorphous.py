#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_utils.synthetic.atomistic_generator import (  # noqa: E402
    _build_global_cfg_from_metadata,
    _build_visualization_atoms_list,
    _load_metadata_json,
    _load_phase_name_map,
)
from src.data_utils.synthetic.visualization import (  # noqa: E402
    _PAPER_FAMILY_DISPLAY_COLORS,
    _build_local_coordination_edges,
    _compute_local_environment_descriptor,
    _compute_paper_display_half_span,
    _ensure_connected_edges,
    _extract_local_neighborhood_points,
    _paper_xyz_filename,
    _prepare_local_structure_geometry,
    _render_local_base_paper_variant,
    _save_structure_xyz,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a local-base paper figure variant with a different amorphous sample "
            "but the same paper color/style."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("output/synthetic_data/polycrystalline_balanced_geometries_v2"),
        help="Dataset directory containing atoms_full.npy, metadata.json, and paper XYZ exports.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="figure_local_base_paper_alt_amorphous.png",
        help="Output file name for the rendered paper figure variant.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Which distinct alternate amorphous candidate to use after sorting by similarity.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=4096,
        help=(
            "Number of amorphous atoms to evaluate. Use 0 or a negative value to evaluate all "
            "amorphous atoms."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed used when subsampling amorphous candidates.",
    )
    parser.add_argument(
        "--min-geometry-distance",
        type=float,
        default=0.40,
        help="Minimum symmetric nearest-neighbor distance from the current amorphous panel.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=80,
        help="Number of points per local environment. Must match the paper figure export.",
    )
    return parser.parse_args()


def _load_xyz_points(xyz_path: pathlib.Path) -> np.ndarray:
    if not xyz_path.exists():
        raise FileNotFoundError(f"Missing XYZ file required for paper figure reuse: {xyz_path}.")
    lines = xyz_path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"XYZ file is too short to contain a valid structure: {xyz_path}.")
    try:
        expected_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(
            f"First line of {xyz_path} must be an integer atom count, got {lines[0]!r}."
        ) from exc
    coords: List[List[float]] = []
    for line_idx, raw_line in enumerate(lines[2:], start=3):
        parts = raw_line.split()
        if len(parts) != 4:
            raise ValueError(
                f"Expected 4 whitespace-separated fields in {xyz_path} line {line_idx}, got {parts}."
            )
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    points = np.asarray(coords, dtype=np.float32)
    if points.shape != (expected_count, 3):
        raise ValueError(
            f"XYZ atom count mismatch for {xyz_path}: header={expected_count}, parsed_shape={points.shape}."
        )
    return points


def _geometry_from_saved_points(points: np.ndarray) -> Dict[str, Any]:
    pts = np.asarray(points, dtype=np.float32)
    edges, _ = _build_local_coordination_edges(
        pts,
        min_shell_neighbors=2,
        max_shell_neighbors=5,
        shell_gap_ratio=1.22,
        edge_mode="coordination_shell_mutual",
    )
    return {
        "points": pts,
        "edges": _ensure_connected_edges(pts, edges),
    }


def _symmetric_nearest_neighbor_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    pts_a = np.asarray(points_a, dtype=np.float32)
    pts_b = np.asarray(points_b, dtype=np.float32)
    if pts_a.ndim != 2 or pts_a.shape[1] != 3:
        raise ValueError(f"points_a must have shape (N, 3), got {pts_a.shape}.")
    if pts_b.ndim != 2 or pts_b.shape[1] != 3:
        raise ValueError(f"points_b must have shape (N, 3), got {pts_b.shape}.")
    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        raise ValueError(
            "Cannot compute nearest-neighbor distance for empty point clouds: "
            f"points_a.shape={pts_a.shape}, points_b.shape={pts_b.shape}."
        )
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    dist_ab = tree_b.query(pts_a, k=1)[0]
    dist_ba = tree_a.query(pts_b, k=1)[0]
    return 0.5 * (float(np.mean(dist_ab)) + float(np.mean(dist_ba)))


def _choose_candidate_offsets(
    total_count: int,
    *,
    candidate_limit: int,
    seed: int,
) -> np.ndarray:
    if total_count <= 0:
        raise ValueError(f"total_count must be positive, got {total_count}.")
    if candidate_limit <= 0 or candidate_limit >= total_count:
        return np.arange(total_count, dtype=int)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(total_count, size=int(candidate_limit), replace=False))


def _select_alternate_amorphous_record(
    phase_atoms: Sequence[Dict[str, Any]],
    atom_positions: np.ndarray,
    positions_tree: cKDTree,
    global_cfg: Any,
    reference_points: np.ndarray,
    *,
    rank: int,
    candidate_limit: int,
    seed: int,
    target_count: int,
    min_geometry_distance: float,
) -> Dict[str, Any]:
    if rank < 0:
        raise ValueError(f"rank must be >= 0, got {rank}.")
    if min_geometry_distance < 0.0:
        raise ValueError(
            f"min_geometry_distance must be non-negative, got {min_geometry_distance}."
        )
    if not phase_atoms:
        raise ValueError("Cannot pick an alternate amorphous sample from an empty phase list.")

    reference_descriptor = _compute_local_environment_descriptor(reference_points)
    candidate_offsets = _choose_candidate_offsets(
        len(phase_atoms),
        candidate_limit=int(candidate_limit),
        seed=int(seed),
    )

    descriptor_candidates: List[Dict[str, Any]] = []
    for offset in candidate_offsets.tolist():
        atom = phase_atoms[int(offset)]
        local_points = _extract_local_neighborhood_points(
            atom,
            atom_positions,
            positions_tree,
            global_cfg,
            target_count=int(target_count),
        )
        descriptor = _compute_local_environment_descriptor(local_points)
        descriptor_candidates.append(
            {
                "atom": atom,
                "local_points": local_points,
                "descriptor_distance": float(
                    np.linalg.norm(descriptor - reference_descriptor)
                ),
            }
        )

    descriptor_candidates.sort(
        key=lambda record: (
            float(record["descriptor_distance"]),
            int(record["atom"]["final_index"]),
        )
    )

    qualifying_records: List[Dict[str, Any]] = []
    for candidate in descriptor_candidates:
        geometry = _prepare_local_structure_geometry(candidate["local_points"])
        geometry_distance = _symmetric_nearest_neighbor_distance(
            geometry["points"],
            reference_points,
        )
        if geometry_distance < float(min_geometry_distance):
            continue
        qualifying_records.append(
            {
                "atom": candidate["atom"],
                "geometry": geometry,
                "descriptor_distance": float(candidate["descriptor_distance"]),
                "geometry_distance": float(geometry_distance),
            }
        )
        if len(qualifying_records) > rank:
            chosen = dict(qualifying_records[rank])
            chosen["candidates_evaluated"] = int(len(candidate_offsets))
            chosen["qualifying_candidates"] = int(len(qualifying_records))
            return chosen

    raise RuntimeError(
        "Failed to find a distinct alternate amorphous sample. "
        f"rank={rank}, candidate_limit={candidate_limit}, seed={seed}, "
        f"min_geometry_distance={min_geometry_distance}, "
        f"evaluated_candidates={len(candidate_offsets)}."
    )


def _build_base_records(xyz_dir: pathlib.Path) -> List[Dict[str, Any]]:
    specs = [
        {
            "family": "BCC",
            "display_color": _PAPER_FAMILY_DISPLAY_COLORS["bcc"],
            "pure_xyz": xyz_dir / _paper_xyz_filename("BCC", "pure"),
            "perturbed_xyz": xyz_dir / _paper_xyz_filename("BCC", "perturbed"),
        },
        {
            "family": "FCC",
            "display_color": _PAPER_FAMILY_DISPLAY_COLORS["fcc"],
            "pure_xyz": xyz_dir / _paper_xyz_filename("FCC", "pure"),
            "perturbed_xyz": xyz_dir / _paper_xyz_filename("FCC", "perturbed"),
        },
        {
            "family": "HCP",
            "display_color": _PAPER_FAMILY_DISPLAY_COLORS["hcp"],
            "pure_xyz": xyz_dir / _paper_xyz_filename("HCP", "pure"),
            "perturbed_xyz": xyz_dir / _paper_xyz_filename("HCP", "perturbed"),
        },
        {
            "family": "Amorphous",
            "display_color": _PAPER_FAMILY_DISPLAY_COLORS["amorphous"],
            "pure_xyz": xyz_dir / _paper_xyz_filename("Amorphous", "pure"),
            "perturbed_xyz": None,
        },
    ]

    records: List[Dict[str, Any]] = []
    for spec in specs:
        pure_points = _load_xyz_points(spec["pure_xyz"])
        pure_geometry = _geometry_from_saved_points(pure_points)
        perturbed_xyz = spec["perturbed_xyz"]
        perturbed_geometry = None
        if perturbed_xyz is not None:
            perturbed_geometry = _geometry_from_saved_points(_load_xyz_points(perturbed_xyz))
        records.append(
            {
                "family": spec["family"],
                "display_color": spec["display_color"],
                "pure_geometry": pure_geometry,
                "perturbed_geometry": perturbed_geometry,
            }
        )
    return records


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir.resolve()
    xyz_dir = data_dir / "figure_local_base_paper_xyz"
    output_path = data_dir / str(args.output_name)
    metadata_path = output_path.with_suffix(".json")

    metadata = _load_metadata_json(data_dir / "metadata.json")
    phase_name_map = _load_phase_name_map(data_dir)
    global_cfg = _build_global_cfg_from_metadata(metadata, data_dir)

    atoms_full_path = data_dir / "atoms_full.npy"
    if not atoms_full_path.exists():
        raise FileNotFoundError(f"Missing atoms_full.npy required for rerendering: {atoms_full_path}.")
    atoms_full = np.load(atoms_full_path, allow_pickle=False)
    atoms = _build_visualization_atoms_list(atoms_full, phase_name_map)
    atom_positions = np.asarray([atom["position"] for atom in atoms], dtype=np.float32)
    positions_tree = cKDTree(atom_positions)

    phase_to_atoms: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for atom in atoms:
        phase_to_atoms[str(atom["phase_id"])].append(atom)

    reference_amorphous_points = _load_xyz_points(
        xyz_dir / _paper_xyz_filename("Amorphous", "pure")
    )
    alternate = _select_alternate_amorphous_record(
        phase_to_atoms["amorphous_pure"],
        atom_positions,
        positions_tree,
        global_cfg,
        reference_amorphous_points,
        rank=int(args.rank),
        candidate_limit=int(args.candidate_limit),
        seed=int(args.seed),
        target_count=int(args.target_count),
        min_geometry_distance=float(args.min_geometry_distance),
    )

    records = _build_base_records(xyz_dir)
    records[-1]["pure_geometry"] = alternate["geometry"]
    display_half_span = _compute_paper_display_half_span(records)
    _render_local_base_paper_variant(
        records,
        output_path,
        display_half_span=display_half_span,
        color_mode="family",
        label_mode="family",
        view_elev=22.0,
        view_azim=38.0,
    )

    xyz_output_name = output_path.stem + "_amorphous.xyz"
    _save_structure_xyz(
        xyz_dir,
        xyz_output_name,
        np.asarray(alternate["geometry"]["points"], dtype=np.float32),
        comment=(
            "Alternate paper amorphous panel | "
            f"rank={int(args.rank)} | "
            f"final_index={int(alternate['atom']['final_index'])}"
        ),
    )

    metadata_record = {
        "data_dir": str(data_dir),
        "reference_amorphous_xyz": str(xyz_dir / _paper_xyz_filename("Amorphous", "pure")),
        "output_figure": str(output_path),
        "output_amorphous_xyz": str(xyz_dir / xyz_output_name),
        "rank": int(args.rank),
        "candidate_limit": int(args.candidate_limit),
        "seed": int(args.seed),
        "min_geometry_distance": float(args.min_geometry_distance),
        "target_count": int(args.target_count),
        "selected_final_index": int(alternate["atom"]["final_index"]),
        "selected_grain_id": int(alternate["atom"]["grain_id"]),
        "descriptor_distance_to_reference": float(alternate["descriptor_distance"]),
        "geometry_distance_to_reference": float(alternate["geometry_distance"]),
        "candidates_evaluated": int(alternate["candidates_evaluated"]),
        "qualifying_candidates_seen": int(alternate["qualifying_candidates"]),
    }
    metadata_path.write_text(json.dumps(metadata_record, indent=2) + "\n")

    print(f"Wrote figure: {output_path}")
    print(f"Wrote amorphous XYZ: {xyz_dir / xyz_output_name}")
    print(f"Wrote metadata: {metadata_path}")
    print(
        "Selected alternate amorphous atom "
        f"final_index={metadata_record['selected_final_index']} "
        f"descriptor_distance={metadata_record['descriptor_distance_to_reference']:.6f} "
        f"geometry_distance={metadata_record['geometry_distance_to_reference']:.6f}"
    )


if __name__ == "__main__":
    main()
