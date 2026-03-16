"""PTM/CNA analysis helpers for cluster representative local point clouds."""

from __future__ import annotations

import csv
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.baselines.descriptor_baselines import CNADescriptorBaseline, infer_center_shell


def _format_cna_signature(signature: str) -> str:
    parts = str(signature).split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid CNA signature format {signature!r}.")
    return f"[{parts[0]} {parts[1]} {parts[2]}]"


def _find_center_atom_index(
    points: np.ndarray,
    *,
    center_atom_tolerance: float,
    context: str,
) -> tuple[int, float]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{context}: expected points with shape (N, 3), got {pts.shape}.")
    if pts.shape[0] == 0:
        raise ValueError(f"{context}: cannot analyze an empty point cloud.")
    norms = np.linalg.norm(pts, axis=1)
    center_idx = int(np.argmin(norms))
    center_dist = float(norms[center_idx])
    if center_dist > float(center_atom_tolerance):
        raise ValueError(
            f"{context}: representative point cloud must be centered on an atom at the origin, "
            f"but the nearest atom is {center_dist:.6e} away at index {center_idx}. "
            f"center_atom_tolerance={float(center_atom_tolerance):.6e}."
        )
    return center_idx, center_dist


def _compute_cna_signature_counts(
    points: np.ndarray,
    *,
    center_atom_tolerance: float,
    shell_min_neighbors: int,
    shell_max_neighbors: int,
) -> tuple[Counter[str], dict[str, Any]]:
    pts = np.asarray(points, dtype=np.float64)
    shell = infer_center_shell(
        pts,
        center_atom_tolerance=float(center_atom_tolerance),
        shell_min_neighbors=int(shell_min_neighbors),
        shell_max_neighbors=int(shell_max_neighbors),
    )
    local_indices = np.concatenate(
        [
            np.asarray([int(shell.center_idx)], dtype=np.int64),
            np.asarray(shell.shell_indices, dtype=np.int64),
        ]
    )
    local_points = np.asarray(pts[local_indices], dtype=np.float64)
    pairwise = np.linalg.norm(local_points[:, None, :] - local_points[None, :, :], axis=-1)
    within_cutoff = pairwise <= float(shell.cutoff)
    adjacency: dict[int, set[int]] = {}
    for point_idx in range(local_points.shape[0]):
        neighbor_idx = set(np.flatnonzero(within_cutoff[point_idx]).tolist())
        neighbor_idx.discard(point_idx)
        adjacency[int(point_idx)] = neighbor_idx

    counts: Counter[str] = Counter()
    shell_neighbor_set = set(range(1, int(local_points.shape[0])))
    for neighbor_idx in sorted(shell_neighbor_set):
        common = sorted(adjacency[0].intersection(adjacency[neighbor_idx]))
        common_set = set(common)
        subgraph = {
            int(node): adjacency[int(node)].intersection(common_set)
            for node in common
        }
        n_common = len(common)
        n_bonds = int(sum(len(neigh) for neigh in subgraph.values()) // 2)
        longest_chain = CNADescriptorBaseline._longest_chain_length(common, subgraph)
        counts[f"{n_common}-{n_bonds}-{longest_chain}"] += 1

    if sum(counts.values()) != len(shell_neighbor_set):
        raise RuntimeError(
            "CNA signature counting mismatch for representative analysis: "
            f"counted_bonds={sum(counts.values())}, shell_size={len(shell_neighbor_set)}."
        )

    return counts, {
        "center_atom_index": int(shell.center_idx),
        "shell_size": int(len(shell.shell_indices)),
        "cutoff": float(shell.cutoff),
        "shell_indices": [int(v) for v in np.asarray(shell.shell_indices, dtype=int).tolist()],
        "shell_distances": [float(v) for v in np.asarray(shell.shell_distances, dtype=float).tolist()],
    }


def _build_ovito_data_collection(points: np.ndarray) -> Any:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")
            from ovito.data import DataCollection
    except Exception as exc:
        raise ModuleNotFoundError(
            "OVITO is required for representative PTM/adaptive-CNA analysis. "
            "Install OVITO in the active environment and rerun. "
            f"Original import error: {type(exc).__name__}: {exc}"
        ) from exc

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"OVITO analysis expects points with shape (N, 3), got {pts.shape}.")
    span = float(np.max(np.ptp(pts, axis=0))) if pts.shape[0] > 0 else 0.0
    box_length = max(10.0, span + 10.0)

    data = DataCollection()
    data.create_particles()
    data.particles_.create_property("Position", data=pts)
    data.particles_.create_property(
        "Particle Type",
        data=np.ones((pts.shape[0],), dtype=np.int32),
    )
    data.create_cell(
        matrix=np.diag([box_length, box_length, box_length]),
        pbc=(False, False, False),
    )
    return data


def _ovito_structure_name_lookup(modifier: Any) -> dict[int, str]:
    return {
        int(structure.id): str(structure.name)
        for structure in modifier.structures
    }


def _count_structure_types(
    structure_ids: np.ndarray,
    structure_name_by_id: dict[int, str],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for raw_structure_id in np.asarray(structure_ids, dtype=int):
        counts[str(structure_name_by_id.get(int(raw_structure_id), "Unknown"))] += 1
    return {
        str(name): int(count)
        for name, count in sorted(counts.items(), key=lambda item: item[0])
    }


def _run_ovito_ptm_analysis(
    points: np.ndarray,
    *,
    center_atom_tolerance: float,
) -> dict[str, Any]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")
            from ovito.modifiers import PolyhedralTemplateMatchingModifier
    except Exception as exc:
        raise ModuleNotFoundError(
            "Representative PTM analysis requires OVITO. "
            "Install OVITO in the active environment and rerun. "
            f"Original import error: {type(exc).__name__}: {exc}"
        ) from exc

    center_idx, center_dist = _find_center_atom_index(
        points,
        center_atom_tolerance=float(center_atom_tolerance),
        context="PTM analysis",
    )
    data = _build_ovito_data_collection(points)
    modifier = PolyhedralTemplateMatchingModifier(
        output_rmsd=True,
        output_interatomic_distance=True,
        output_ordering=True,
        output_orientation=True,
    )
    data.apply(modifier)
    structure_ids = np.asarray(data.particles["Structure Type"], dtype=int)
    structure_name_by_id = _ovito_structure_name_lookup(modifier)
    rmsd = np.asarray(data.particles["RMSD"], dtype=np.float64)
    interatomic_distance = np.asarray(data.particles["Interatomic Distance"], dtype=np.float64)
    ordering_type = np.asarray(data.particles["Ordering Type"], dtype=int)
    orientation = np.asarray(data.particles["Orientation"], dtype=np.float64)
    ordering_value = int(ordering_type[center_idx])
    ordering_name = str(PolyhedralTemplateMatchingModifier.OrderingType(ordering_value).name)

    return {
        "center_atom_index": int(center_idx),
        "center_atom_distance": float(center_dist),
        "center_structure_type_id": int(structure_ids[center_idx]),
        "center_structure_type": str(structure_name_by_id.get(int(structure_ids[center_idx]), "Unknown")),
        "center_rmsd": float(rmsd[center_idx]),
        "center_interatomic_distance": float(interatomic_distance[center_idx]),
        "center_ordering_type_id": int(ordering_value),
        "center_ordering_type": ordering_name,
        "center_orientation_quaternion": [
            float(v) for v in np.asarray(orientation[center_idx], dtype=np.float64).tolist()
        ],
        "environment_structure_counts": _count_structure_types(structure_ids, structure_name_by_id),
    }


def _run_ovito_adaptive_cna_analysis(
    points: np.ndarray,
    *,
    center_atom_tolerance: float,
) -> dict[str, Any]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")
            from ovito.modifiers import CommonNeighborAnalysisModifier
    except Exception as exc:
        raise ModuleNotFoundError(
            "Representative adaptive-CNA analysis requires OVITO. "
            "Install OVITO in the active environment and rerun. "
            f"Original import error: {type(exc).__name__}: {exc}"
        ) from exc

    center_idx, center_dist = _find_center_atom_index(
        points,
        center_atom_tolerance=float(center_atom_tolerance),
        context="adaptive CNA analysis",
    )
    data = _build_ovito_data_collection(points)
    modifier = CommonNeighborAnalysisModifier(
        mode=CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff
    )
    data.apply(modifier)
    structure_ids = np.asarray(data.particles["Structure Type"], dtype=int)
    structure_name_by_id = _ovito_structure_name_lookup(modifier)
    return {
        "center_atom_index": int(center_idx),
        "center_atom_distance": float(center_dist),
        "center_structure_type_id": int(structure_ids[center_idx]),
        "center_structure_type": str(structure_name_by_id.get(int(structure_ids[center_idx]), "Unknown")),
        "environment_structure_counts": _count_structure_types(structure_ids, structure_name_by_id),
    }


def _write_structure_analysis_csv(
    csv_path: Path,
    *,
    representative_records: list[dict[str, Any]],
    cna_signature_vocab: list[str],
) -> None:
    fieldnames = [
        "cluster_id",
        "sample_index",
        "num_points_analyzed",
        "center_atom_index",
        "center_atom_distance",
        "cna_shell_size",
        "cna_shell_cutoff",
        "cna_top_signature",
        "cna_top_signature_label",
        "cna_top_signature_fraction",
        "cna_other_fraction",
        "adaptive_cna_center_structure_type",
        "ptm_center_structure_type",
        "ptm_center_rmsd",
        "ptm_center_interatomic_distance",
        "ptm_center_ordering_type",
    ]
    signature_columns = [f"cna_sig_{signature.replace('-', '_')}" for signature in cna_signature_vocab]
    fieldnames.extend(signature_columns)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in representative_records:
            cna = record.get("cna", {})
            ptm = record.get("ptm", {})
            adaptive_cna = record.get("adaptive_cna", {})
            fractions = cna.get("signature_fractions", {})
            row = {
                "cluster_id": int(record["cluster_id"]),
                "sample_index": int(record["sample_index"]),
                "num_points_analyzed": int(record["num_points_analyzed"]),
                "center_atom_index": int(record["center_atom_index"]),
                "center_atom_distance": float(record["center_atom_distance"]),
                "cna_shell_size": (
                    ""
                    if "shell_size" not in cna
                    else int(cna["shell_size"])
                ),
                "cna_shell_cutoff": (
                    ""
                    if "cutoff" not in cna
                    else float(cna["cutoff"])
                ),
                "cna_top_signature": str(cna.get("top_signature", "")),
                "cna_top_signature_label": str(cna.get("top_signature_label", "")),
                "cna_top_signature_fraction": (
                    ""
                    if "top_signature_fraction" not in cna
                    else float(cna["top_signature_fraction"])
                ),
                "cna_other_fraction": (
                    ""
                    if "other_fraction" not in cna
                    else float(cna["other_fraction"])
                ),
                "adaptive_cna_center_structure_type": str(
                    adaptive_cna.get("center_structure_type", "")
                ),
                "ptm_center_structure_type": str(ptm.get("center_structure_type", "")),
                "ptm_center_rmsd": (
                    ""
                    if "center_rmsd" not in ptm
                    else float(ptm["center_rmsd"])
                ),
                "ptm_center_interatomic_distance": (
                    ""
                    if "center_interatomic_distance" not in ptm
                    else float(ptm["center_interatomic_distance"])
                ),
                "ptm_center_ordering_type": str(ptm.get("center_ordering_type", "")),
            }
            for signature, column in zip(cna_signature_vocab, signature_columns):
                row[column] = float(fractions.get(signature, 0.0))
            writer.writerow(row)


def _build_cluster_representative_analysis_summary(
    prepared_records: list[dict[str, Any]],
    *,
    ptm_enabled: bool,
    cna_enabled: bool,
    cna_max_signatures: int,
    center_atom_tolerance: float,
    shell_min_neighbors: int,
    shell_max_neighbors: int,
) -> dict[str, Any] | None:
    if not bool(ptm_enabled) and not bool(cna_enabled):
        return None
    if not prepared_records:
        raise ValueError("Cannot analyze representative structures because prepared_records is empty.")
    if int(cna_max_signatures) <= 0:
        raise ValueError(
            f"cna_max_signatures must be a positive integer, got {cna_max_signatures}."
        )
    if int(shell_min_neighbors) < 2:
        raise ValueError(
            f"shell_min_neighbors must be >= 2 for representative CNA analysis, got {shell_min_neighbors}."
        )
    if int(shell_max_neighbors) <= int(shell_min_neighbors):
        raise ValueError(
            "shell_max_neighbors must exceed shell_min_neighbors for representative CNA analysis, "
            f"got {shell_max_neighbors} <= {shell_min_neighbors}."
        )

    aggregate_signature_counts: Counter[str] = Counter()
    per_record_cna_counts: list[Counter[str] | None] = []
    per_record_cna_shell_info: list[dict[str, Any] | None] = []
    for prepared in prepared_records:
        local_points = np.asarray(prepared["local_points"], dtype=np.float64)
        if bool(cna_enabled):
            cna_counts, shell_info = _compute_cna_signature_counts(
                local_points,
                center_atom_tolerance=float(center_atom_tolerance),
                shell_min_neighbors=int(shell_min_neighbors),
                shell_max_neighbors=int(shell_max_neighbors),
            )
            aggregate_signature_counts.update(cna_counts)
            per_record_cna_counts.append(cna_counts)
            per_record_cna_shell_info.append(shell_info)
        else:
            per_record_cna_counts.append(None)
            per_record_cna_shell_info.append(None)

    cna_signature_vocab = [
        str(signature)
        for signature, _ in aggregate_signature_counts.most_common(int(cna_max_signatures))
    ]

    ovito_available = False
    ovito_import_error: str | None = None
    if bool(ptm_enabled) or bool(cna_enabled):
        try:
            _build_ovito_data_collection(np.asarray(prepared_records[0]["local_points"], dtype=np.float64))
            ovito_available = True
        except ModuleNotFoundError as exc:
            ovito_import_error = str(exc)
            if bool(ptm_enabled):
                raise

    representative_records: list[dict[str, Any]] = []
    for prepared, cna_counts, shell_info in zip(
        prepared_records,
        per_record_cna_counts,
        per_record_cna_shell_info,
    ):
        local_points = np.asarray(prepared["local_points"], dtype=np.float64)
        center_atom_index, center_atom_distance = _find_center_atom_index(
            local_points,
            center_atom_tolerance=float(center_atom_tolerance),
            context=(
                f"representative analysis for cluster {int(prepared['cluster_id'])}, "
                f"sample {int(prepared['sample_index'])}"
            ),
        )
        record: dict[str, Any] = {
            "cluster_id": int(prepared["cluster_id"]),
            "sample_index": int(prepared["sample_index"]),
            "num_points_analyzed": int(local_points.shape[0]),
            "center_atom_index": int(center_atom_index),
            "center_atom_distance": float(center_atom_distance),
        }

        if bool(cna_enabled):
            if cna_counts is None or shell_info is None:
                raise RuntimeError(
                    "Representative CNA analysis bookkeeping mismatch: expected per-record CNA results."
                )
            total_signatures = int(sum(cna_counts.values()))
            if total_signatures <= 0:
                raise RuntimeError(
                    "Representative CNA analysis observed no signatures for "
                    f"cluster {int(prepared['cluster_id'])}, sample {int(prepared['sample_index'])}."
                )
            signature_fractions = {
                str(signature): float(int(cna_counts.get(signature, 0)) / total_signatures)
                for signature in cna_signature_vocab
            }
            other_count = total_signatures - sum(
                int(cna_counts.get(signature, 0)) for signature in cna_signature_vocab
            )
            top_signature = max(
                cna_counts.items(),
                key=lambda item: (int(item[1]), str(item[0])),
            )[0]
            cna_record = {
                **shell_info,
                "signature_vocab": list(cna_signature_vocab),
                "signature_counts": {
                    str(signature): int(count)
                    for signature, count in sorted(cna_counts.items(), key=lambda item: item[0])
                },
                "signature_fractions": signature_fractions,
                "other_count": int(other_count),
                "other_fraction": float(other_count / total_signatures),
                "top_signature": str(top_signature),
                "top_signature_label": str(_format_cna_signature(top_signature)),
                "top_signature_fraction": float(int(cna_counts[top_signature]) / total_signatures),
            }
            if bool(ovito_available):
                cna_record["adaptive_structure"] = _run_ovito_adaptive_cna_analysis(
                    local_points,
                    center_atom_tolerance=float(center_atom_tolerance),
                )
                record["adaptive_cna"] = dict(cna_record["adaptive_structure"])
            record["cna"] = cna_record

        if bool(ptm_enabled):
            record["ptm"] = _run_ovito_ptm_analysis(
                local_points,
                center_atom_tolerance=float(center_atom_tolerance),
            )

        representative_records.append(record)

    return {
        "ptm_enabled": bool(ptm_enabled),
        "cna_enabled": bool(cna_enabled),
        "analysis_points_source": "local_points",
        "center_atom_tolerance": float(center_atom_tolerance),
        "shell_min_neighbors": int(shell_min_neighbors),
        "shell_max_neighbors": int(shell_max_neighbors),
        "cna_max_signatures": int(cna_max_signatures),
        "ovito_available": bool(ovito_available),
        "ovito_import_error": None if ovito_import_error is None else str(ovito_import_error),
        "cna_signature_vocab": list(cna_signature_vocab),
        "representatives": representative_records,
    }


def materialize_cluster_representative_analysis_summary(
    structure_analysis_summary: dict[str, Any],
    out_dir: Path,
    *,
    k_token: str,
) -> dict[str, Any]:
    if not isinstance(structure_analysis_summary, dict):
        raise TypeError(
            "structure_analysis_summary must be a dict, "
            f"got {type(structure_analysis_summary)!r}."
        )
    representatives = structure_analysis_summary.get("representatives")
    if not isinstance(representatives, list) or not representatives:
        raise ValueError(
            "structure_analysis_summary must contain a non-empty 'representatives' list."
        )
    cna_signature_vocab = structure_analysis_summary.get("cna_signature_vocab", [])
    if not isinstance(cna_signature_vocab, list):
        raise ValueError(
            "structure_analysis_summary['cna_signature_vocab'] must be a list, "
            f"got {type(cna_signature_vocab)!r}."
        )

    json_path = Path(out_dir) / f"10_cluster_representatives_structure_analysis_k{k_token}.json"
    csv_path = Path(out_dir) / f"10_cluster_representatives_structure_analysis_k{k_token}.csv"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "ptm_enabled": bool(structure_analysis_summary.get("ptm_enabled", False)),
        "cna_enabled": bool(structure_analysis_summary.get("cna_enabled", False)),
        "analysis_points_source": str(
            structure_analysis_summary.get("analysis_points_source", "local_points")
        ),
        "center_atom_tolerance": float(
            structure_analysis_summary.get("center_atom_tolerance", 1.0e-6)
        ),
        "shell_min_neighbors": int(
            structure_analysis_summary.get("shell_min_neighbors", 8)
        ),
        "shell_max_neighbors": int(
            structure_analysis_summary.get("shell_max_neighbors", 24)
        ),
        "cna_max_signatures": int(
            structure_analysis_summary.get("cna_max_signatures", 5)
        ),
        "ovito_available": bool(structure_analysis_summary.get("ovito_available", False)),
        "ovito_import_error": (
            None
            if structure_analysis_summary.get("ovito_import_error", None) is None
            else str(structure_analysis_summary.get("ovito_import_error"))
        ),
        "cna_signature_vocab": [str(v) for v in cna_signature_vocab],
        "representatives": [dict(record) for record in representatives],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_structure_analysis_csv(
        csv_path,
        representative_records=payload["representatives"],
        cna_signature_vocab=[str(v) for v in cna_signature_vocab],
    )

    cloned_summary = dict(structure_analysis_summary)
    cloned_summary["json_file"] = str(json_path)
    cloned_summary["csv_file"] = str(csv_path)
    cloned_summary["representatives"] = payload["representatives"]
    cloned_summary["cna_signature_vocab"] = [str(v) for v in cna_signature_vocab]
    return cloned_summary


def analyze_cluster_representatives(
    prepared_records: list[dict[str, Any]],
    out_dir: Path,
    *,
    k_token: str,
    ptm_enabled: bool,
    cna_enabled: bool,
    cna_max_signatures: int,
    center_atom_tolerance: float,
    shell_min_neighbors: int,
    shell_max_neighbors: int,
) -> dict[str, Any] | None:
    summary = _build_cluster_representative_analysis_summary(
        prepared_records,
        ptm_enabled=bool(ptm_enabled),
        cna_enabled=bool(cna_enabled),
        cna_max_signatures=int(cna_max_signatures),
        center_atom_tolerance=float(center_atom_tolerance),
        shell_min_neighbors=int(shell_min_neighbors),
        shell_max_neighbors=int(shell_max_neighbors),
    )
    if summary is None:
        return None
    return materialize_cluster_representative_analysis_summary(
        summary,
        out_dir,
        k_token=str(k_token),
    )
