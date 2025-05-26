# -*- coding: utf-8 -*-
"""soap_parquet.py ──────────────────────────────────────────────────────────────
End‑to‑end helpers to

1. build a **DScribe SOAP descriptor** with the modern API (`r_cut`, `n_max`,
   `l_max` …),
2. iterate over atomic point clouds (ASE `Atoms` objects) and compute per–center
   SOAP vectors, and
3. **persist / reload** the resulting feature matrix in Apache Parquet so that
   downstream PyTorch code can `torch.from_numpy()` it without extra copies.

*Tested with **DScribe 2.1.0** (May 2025).*
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from ase import Atoms
from dscribe.descriptors import SOAP
from pathlib import Path
from typing import List, Dict, Any


__all__ = [
    "build_soap",
    "compute_soap_vectors",
    "save_parquet",
    "load_parquet",
]

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Build descriptor
# ──────────────────────────────────────────────────────────────────────────────

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
    generate_soap_features_for_npy_files(
        npy_file_paths=default_npy_files,
        output_directory=default_output_path,
        species_element_symbol=default_species_symbol,
        soap_parameters=default_soap_params,
    )
    
   
