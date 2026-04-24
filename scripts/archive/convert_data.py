import os, sys, glob, argparse
sys.path.append(os.getcwd())
import numpy as np


def read_pos_file(filename: str) -> np.ndarray:
    """Read atomic positions from LAMMPS .pos dump file.

    Returns float32 array of shape (N, 3).
    """
    atoms = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if "ITEM: ATOMS" in line:
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError(f"No 'ITEM: ATOMS' header found in {filename}")

    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) >= 5:
            x, y, z = parts[2], parts[3], parts[4]
            atoms.append([float(x), float(y), float(z)])

    if not atoms:
        raise ValueError(f"No atoms parsed from {filename}")

    return np.array(atoms, dtype=np.float32)


def convert_pos_to_npy(input_path: str, output_path: str) -> None:
    """Convert .pos file to .npy float32."""
    atoms = read_pos_file(input_path)
    np.save(output_path, atoms)
    print(f"  {os.path.basename(input_path)} -> {os.path.basename(output_path)}  ({atoms.shape[0]} atoms)")


def convert_npz_to_npy(input_path: str, output_path: str) -> None:
    """Convert .npz (with 'positions' key) to .npy float32."""
    data = np.load(input_path)
    atoms = data['positions'].astype(np.float32)
    np.save(output_path, atoms)
    print(f"  {os.path.basename(input_path)} -> {os.path.basename(output_path)}  ({atoms.shape[0]} atoms)")


def ensure_npy_float32(npy_path: str) -> bool:
    """Re-save .npy file as float32 if needed. Returns True if rewritten."""
    arr = np.load(npy_path)
    if arr.dtype == np.float32:
        return False
    arr32 = arr.astype(np.float32)
    np.save(npy_path, arr32)
    print(f"  {os.path.basename(npy_path)}: {arr.dtype} -> float32")
    return True


def convert_directory(directory: str) -> None:
    """Convert all .pos files in a directory to .npy float32."""
    pos_files = sorted(glob.glob(os.path.join(directory, "*.pos")))
    if not pos_files:
        raise FileNotFoundError(
            f"convert_directory: no .pos files found in {directory}. "
            "Point the dataset spec at the correct location or remove it from the dataset map."
        )
    for pos_file in pos_files:
        base = os.path.splitext(pos_file)[0]
        npy_file = base + ".npy"
        if os.path.exists(npy_file):
            arr = np.load(npy_file)
            if arr.dtype == np.float32:
                print(f"  {os.path.basename(npy_file)} already exists (float32), skipping")
                continue
        convert_pos_to_npy(pos_file, npy_file)


def upgrade_al_directory(directory: str) -> None:
    """Ensure Al .npy files are float32, convert from .npz if only npz exists."""
    for npz_file in sorted(glob.glob(os.path.join(directory, "*.npz"))):
        base = os.path.splitext(npz_file)[0]
        npy_file = base + ".npy"
        if not os.path.exists(npy_file):
            convert_npz_to_npy(npz_file, npy_file)

    for npy_file in sorted(glob.glob(os.path.join(directory, "*.npy"))):
        ensure_npy_float32(npy_file)


def delete_non_npy(directory: str, extensions: list[str]) -> None:
    """Delete files with given extensions from directory."""
    for ext in extensions:
        for path in sorted(glob.glob(os.path.join(directory, f"*{ext}"))):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            os.remove(path)
            print(f"  Deleted {os.path.basename(path)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert atomistic data to .npy float32")
    parser.add_argument("--no-delete", action="store_true", help="Skip deleting old formats")
    args = parser.parse_args()

    datasets = {
        "datasets/Al/inherent_configurations_off": {"action": "upgrade_al", "delete_exts": [".off", ".npz"]},
        "datasets/Mg/inherent_configurations": {"action": "convert_pos", "delete_exts": [".pos"]},
        "datasets/Ta/inherent_configurations": {"action": "convert_pos", "delete_exts": [".pos"]},
        "datasets/Zr/inherent_configurations": {"action": "convert_pos", "delete_exts": [".pos"]},
        "datasets/Al50Ni50/inherent_configurations": {"action": "convert_pos", "delete_exts": [".pos"]},
    }

    missing = [d for d in datasets if not os.path.isdir(d)]
    if missing:
        raise FileNotFoundError(
            "convert_data: the following dataset directories are missing: "
            f"{missing}. Fix the hardcoded `datasets` map or create the directories before running."
        )

    for directory, spec in datasets.items():
        print(f"\n=== {directory} ===")
        if spec["action"] == "convert_pos":
            convert_directory(directory)
        elif spec["action"] == "upgrade_al":
            upgrade_al_directory(directory)

        if not args.no_delete:
            delete_non_npy(directory, spec["delete_exts"])

