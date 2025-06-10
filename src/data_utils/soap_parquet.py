
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union, Dict, Any

import numpy as np
import pandas as pd
from ase import Atoms
from dscribe.descriptors import SOAP
from pathlib import Path
from typing import List, Dict, Any
from scipy.spatial import KDTree
import itertools
import json
import math
from tqdm import tqdm
import os

__all__ = [
    "build_soap",
    "compute_soap_vectors",
    "save_parquet",
    "load_parquet",
    "select_points_from_parquet",
    "select_and_save_points_to_parquet",
    "generate_soap_features_for_npy_files",
]


def build_soap(
    *,
    species: Sequence[Union[str, int]],
    r_cut: float = 5.0,
    n_max: int = 8,
    l_max: int = 6,
    sigma: float = 0.3,
    periodic: bool = False,
    compression_mode: str = "off",
    dtype: str = "float64",
) -> SOAP:
    """Return a configured DScribe :class:`SOAP` object."""
    return SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        periodic=periodic,
        compression={"mode": compression_mode, "species_weighting": None},
        sparse=False,
        dtype=dtype,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Compute vectors
# ──────────────────────────────────────────────────────────────────────────────

def compute_soap_vectors(
    samples: Iterable[Tuple[Atoms, Union[str, Sequence[int], np.ndarray, None]]],
    soap: SOAP,
    *,
    n_jobs: int = 1,
    show_progress: bool = False,
) -> np.ndarray:
    """Stack SOAP rows into a **(N, M)** NumPy array.

    Parameters
    ----------
    samples
        Iterable yielding ``(atoms, centres)`` where
        * *atoms* – ASE :class:`~ase.Atoms`.
        * *centres* – one of
            * ``None`` or ``"all"`` → use **all** atoms in *atoms*,
            * 1‑D array of integer **indices** *or* Cartesian **positions**.
    soap
        The descriptor built via :func:`build_soap`.
    n_jobs
        Parallel workers for DScribe (`-1` = all cores).
    show_progress
        Print a status line every 1 000 processed environments.
    """
    rows: List[np.ndarray] = []
    total = 0

    for i, (atoms, centres) in enumerate(samples):
        if centres is None or (isinstance(centres, str) and centres == "all"):
            centers_arg: Union[str, Sequence[int], np.ndarray, None] = atoms.positions
        else:
            centers_arg = np.asarray(centres)

        # DScribe ≥2.1 uses **centers** keyword (not *positions*)
        vec = soap.create(atoms, centers=centers_arg, n_jobs=n_jobs)
        rows.append(vec)
        total += vec.shape[0]

        if show_progress and total % 1000 == 0:
            print(f"  … processed {total:,} local environments", flush=True)

    return np.vstack(rows, dtype=soap.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Parquet I/O
# ──────────────────────────────────────────────────────────────────────────────

def save_parquet(
    features: np.ndarray,
    path: Union[str, Path],
    *,
    coordinates: np.ndarray | None = None,
    compression: str = "snappy",
) -> None:
    """Write *features* (and optionally *coordinates*) to *path* in **Apache Parquet** columnar format."""
    pd_path = Path(path)

    if coordinates is not None:
        if features.shape[0] != coordinates.shape[0]:
            raise ValueError(
                f"Features ({features.shape[0]} rows) and coordinates ({coordinates.shape[0]} rows) "
                "must have the same number of rows (atoms/centers)."
            )
        
        coord_cols = [f'coord_{j}' for j in range(coordinates.shape[1])]
        feature_cols = [f'f{i}' for i in range(features.shape[1])]
        all_columns = coord_cols + feature_cols
        
        data_to_save = np.hstack((coordinates, features))
        df = pd.DataFrame(data_to_save, columns=all_columns)
    else:
        df = pd.DataFrame(features, columns=[f"f{i}" for i in range(features.shape[1])])
        
    df.to_parquet(pd_path, engine="pyarrow", compression=compression)


def load_parquet(path: Union[str, Path]) -> np.ndarray:
    """Reload matrix saved with :func:`save_parquet`.
    
    If coordinates were saved alongside features, they will be the first N columns
    of the returned NumPy array.
    """
    return pd.read_parquet(path, engine="pyarrow").to_numpy()


def select_points_from_parquet(
    parquet_path: Union[str, Path],
    num_coord_dims: int,
    stride: float,
    *,
    show_progress: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Sample points on a stride-aligned grid, snap each virtual cell-centre to
    the nearest real point (KD-tree), drop true boundary layers, and return
    the unique point indices.

    Parameters
    ----------
    parquet_path
        File with at least `num_coord_dims` columns of coordinates.
    num_coord_dims
        Dimensionality of coordinates (e.g. 2 or 3).
    stride
        Side length of each grid cell (> 0).
    show_progress
        If True, display a tqdm progress bar.

    Returns
    -------
    sel_indices
        1D np.ndarray of unique integer indices into the original table.
    stats
        dict of diagnostic metadata.
    """
    if stride <= 0:
        raise ValueError("`stride` must be strictly positive.")

    # 1) Load data & extract coords
    data = pd.read_parquet(parquet_path, engine="pyarrow").to_numpy()
    if data.shape[1] < num_coord_dims:
        raise ValueError(
            f"Expected at least {num_coord_dims} columns, found {data.shape[1]}"
        )
    points = data[:, :num_coord_dims]
    if points.size == 0:
        return np.empty(0, dtype=int), {"notes": "no points in file"}

    # build KD-tree once
    kdt = KDTree(points)

    # 2) compute stride-aligned bounds
    p_min = points.min(axis=0)
    p_max = points.max(axis=0)
    # snap outwards to multiples of stride
    min_snap = np.floor(p_min / stride) * stride
    max_snap = np.ceil(p_max / stride) * stride
    span     = max_snap - min_snap
    # number of cells along each dim
    n_cells = (span / stride).astype(int)

    total_cells = int(math.prod(n_cells))
    if total_cells == 0:
        raise ValueError("grid collapsed (check stride)")

    # 3) determine interior‐only ranges
    # drop 0 and n-1 only if n > 2; else keep all
    cell_ranges = [
        range(1, n - 1) if n > 2 else range(n)
        for n in n_cells
    ]
    used_cells = int(math.prod(len(r) for r in cell_ranges))

    # 4) iterate & snap
    indices: set[int] = set()
    iterator = itertools.product(*cell_ranges)
    if show_progress:
        iterator = tqdm(iterator, total=used_cells, desc="Snapping cells")

    for cell in iterator:
        centre = min_snap + (np.array(cell) + 0.5) * stride
        _, idx = kdt.query(centre, k=1)
        indices.add(int(idx))

    sel = np.fromiter(indices, dtype=int)

    # 5) collect stats
    stats: Dict[str, Any] = {
        "total_points_loaded": points.shape[0],
        "num_coord_dims":       num_coord_dims,
        "stride":               float(stride),
        "min_bounds_original":  p_min.tolist(),
        "max_bounds_original":  p_max.tolist(),
        "min_bounds_aligned":   min_snap.tolist(),
        "max_bounds_aligned":   max_snap.tolist(),
        "grid_shape_nominal":   n_cells.tolist(),
        "grid_cells_total":     total_cells,
        "grid_cells_used":      used_cells,
        "unique_points":        len(indices),
    }

    return sel, stats


def select_and_save_points_to_parquet(
    input_parquet_path: Union[str, Path],
    output_parquet_path: Union[str, Path],
    num_coord_dims: int,
    stride: float,
    *,
    show_progress: bool = False,
) -> None:
    """
    1.  Run :pyfunc:`select_points_from_parquet` on *input_parquet_path* to obtain
        a sparse set of representative points (KD-tree snapping on a stride-aligned
        grid, boundary layer discarded).
    2.  Keep **only**:

        • the first *num_coord_dims* columns (Cartesian coordinates)  
        • the remaining columns, assumed to be the SOAP feature vectors

    3.  Persist the subset to *output_parquet_path* via :pyfunc:`save_parquet`.

    Parameters
    ----------
    input_parquet_path
        Source Parquet (coordinates + SOAP).
    output_parquet_path
        Destination Parquet (same column layout, but fewer rows).
    num_coord_dims
        Number of leading coordinate columns.
    stride
        Grid spacing used by :pyfunc:`select_points_from_parquet`.
    show_progress
        Forwarded to the selector – enables a live ``tqdm`` bar.
    selection_stats_json_path
        Optional path to dump selector diagnostics as JSON.
    """
    input_path  = Path(input_parquet_path)
    output_path = Path(output_parquet_path)

    print(f"⟹ selecting points from {input_path.resolve()}")

    # 1. sparse indices on a stride-aligned grid
    selected_idx, stats = select_points_from_parquet(
        parquet_path=input_path,
        num_coord_dims=num_coord_dims,
        stride=stride,
        show_progress=show_progress,
    )
    print(f"kept {len(selected_idx):,} / {stats['total_points_loaded']:,} atoms")

    if selected_idx.size == 0:
        raise RuntimeError("selector returned an empty index set – nothing to save")

    # 2. slice full matrix → coordinates | SOAP
    full = load_parquet(input_path)   # (N, num_coord_dims + n_features)
    if full.shape[1] < num_coord_dims:
        raise ValueError(
            f"Parquet has only {full.shape[1]} columns, "
            f"but num_coord_dims={num_coord_dims}"
        )

    subset_coords = full[selected_idx, :num_coord_dims]
    subset_soap   = full[selected_idx, num_coord_dims:]   # everything after coords

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Soap rows: {subset_soap.shape}")
    print(f"Coords rows: {subset_coords.shape}")
    save_parquet(
        features=subset_soap,
        path=output_path,
        coordinates=subset_coords,
    )
    print(f" wrote {subset_coords.shape[0]:,} rows → {output_path.resolve()}")
    


def generate_soap_features_for_npy_files(
    npy_file_paths: List[Union[str, Path]],
    output_directory: Union[str, Path],
    species_element_symbol: str,
    soap_parameters: Dict[str, Any],
) -> None:
    """
    Processes a list of .npy point cloud files, computes SOAP descriptors for each,
    and saves them (along with coordinates) as .parquet files in the specified output directory.

    Args:
        npy_file_paths: List of paths to .npy files containing point clouds.
        output_directory: Path to the directory where .parquet files will be saved.
        species_element_symbol: Symbol of the atomic species (e.g., "Al").
        soap_parameters: Dictionary of parameters for the SOAP descriptor,
                         e.g., {"r_cut": 5.0, "n_max": 8, "l_max": 6, "sigma": 0.2}.
    """
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")
    
    # Build SOAP descriptor (once for all files)
    # Assumes build_soap, compute_soap_vectors, save_parquet, load_parquet 
    # are defined in the same module or imported.
    soap_desc = build_soap(
        species=[species_element_symbol],
        **soap_parameters
    )
    print(f"Built SOAP descriptor with parameters: {soap_parameters}")
    
    for i, npy_file_str_path in enumerate(npy_file_paths):
        current_npy_file = Path(npy_file_str_path)
        print(f"\n[{i+1}/{len(npy_file_paths)}] Processing {current_npy_file}")
        
        points = np.load(current_npy_file)  
        print(f"  Loaded {len(points)} points with shape {points.shape}")

        
        # Convert point cloud to ASE Atoms object
        n_points = len(points)
        symbols = [species_element_symbol] * n_points
        atoms_obj = Atoms(symbols=symbols, positions=points)
        
        print(f"  Created Atoms object with {len(atoms_obj)} atoms")
        print(f"  Position range: min={points.min(axis=0)}, max={points.max(axis=0)}")
        
        # Compute SOAP vectors for all points
        print("  Computing SOAP descriptors...")
        feature_matrix = compute_soap_vectors(
            [(atoms_obj, "all")],  # Single atoms object, all centers
            soap_desc, 
            n_jobs=-1, 
            show_progress=True
        )
        print(f"  Feature matrix shape: {feature_matrix.shape}")
        
        # Save individual file (features + coordinates)
        file_stem = current_npy_file.stem
        output_file_path = output_dir / f"{file_stem}_soap_with_coords.parquet" # Changed filename slightly
        save_parquet(feature_matrix, output_file_path, coordinates=points)
        print(f"  Saved data (coordinates + features) → {output_file_path}")
        
        # Verify round-trip for this file
        reloaded_data = load_parquet(output_file_path)
        num_coord_dims = points.shape[1]
        
        reloaded_coords = reloaded_data[:, :num_coord_dims]
        reloaded_features = reloaded_data[:, num_coord_dims:]
        
        assert np.allclose(points, reloaded_coords), "Coordinates round-trip failed!"
        assert np.allclose(feature_matrix, reloaded_features), "Features round-trip failed!"
        print(f"  Round-trip verification successful!")


if __name__ == "__main__":
    default_npy_files = [
        "datasets/Al/inherent_configurations_off/166ps.npy",
        "datasets/Al/inherent_configurations_off/170ps.npy",
        "datasets/Al/inherent_configurations_off/174ps.npy",
        "datasets/Al/inherent_configurations_off/175ps.npy",
        "datasets/Al/inherent_configurations_off/177ps.npy",
        "datasets/Al/inherent_configurations_off/240ps.npy",
    ]
    
    default_output_path = "datasets/Al/soap_features/"
    default_species_symbol = "Al"
    
    default_soap_params = {
        "r_cut": 5.0,      # Cutoff radius
        "n_max": 8,        # Radial basis functions
        "l_max": 6,        # Angular basis functions  
        "sigma": 0.2       # Gaussian width
    }

    # Call the main processing function
    if True:
        generate_soap_features_for_npy_files(
            npy_file_paths=default_npy_files,
            output_directory=default_output_path,
            species_element_symbol=default_species_symbol,
            soap_parameters=default_soap_params,
        )
        
    for file in os.listdir(default_output_path):
        select_and_save_points_to_parquet(
            input_parquet_path=f"datasets/Al/soap_features/{file}",
            output_parquet_path=f"datasets/Al/soap_features/{file.replace('_soap_with_coords.parquet', '_selected.parquet')}",
            num_coord_dims=3,
            stride=5,
            show_progress=True,
        )
            
   
