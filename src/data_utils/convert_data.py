import os, sys
sys.path.append(os.getcwd())
import numpy as np

from src.data_utils.prepare_data import read_off_file


def read_xyz_file(filename):
    """Read atomic positions from the input file."""
    atoms = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    # Find where the atomic coordinates start
    for i, line in enumerate(lines):
        if "ITEM: ATOMS" in line:
            start_idx = i + 1
            break
    
    # Read atomic positions
    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) >= 5:
            _, _, x, y, z = parts[:5]  # We only need x, y, z coordinates
            atoms.append([float(x), float(y), float(z)])
    
    return atoms

def write_off_file(atoms, output_filename):
    """Write atomic positions to OFF format."""
    with open(output_filename, 'w') as f:
        # Write header
        f.write("OFF\n")
        
        # Write number of vertices, faces, and edges
        # Just writing vertices (no faces/edges)
        f.write(f"{len(atoms)} 0 0\n")
        # Write vertex coordinates
        for x, y, z in atoms:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def write_npz_file(atoms, output_filename):
    """Write atomic positions to NPZ format."""
    np_atoms = np.array(atoms, dtype=np.float32) # Convert to NumPy array
    np.savez(output_filename, positions=np_atoms)

def convert_xyz_to_off(input_filename, output_filename):
    """Convert XYZ format to OFF format."""
    atoms = read_xyz_file(input_filename)
    
    write_off_file(atoms, output_filename)
    
    print(f"Conversion complete! Wrote {len(atoms)} vertices to {output_filename}")

def convert_xyz_to_npz(input_filename, output_filename):
    """Convert XYZ format to NPZ format."""
    atoms = read_xyz_file(input_filename)
    
    write_npz_file(atoms, output_filename)
    
    print(f"Conversion complete! Wrote {len(atoms)} vertices to {output_filename} (NPZ format)")

def convert_off_to_npz(input_filename, output_filename):
    """Convert OFF format to NPZ format."""
    atoms = read_off_file(input_filename)
    
    write_npz_file(atoms, output_filename)
    
    print(f"Conversion complete! Wrote {len(atoms)} vertices from {input_filename} to {output_filename} (NPZ format)")


if __name__ == "__main__":
    original_folder = "datasets/Al/inherent_configurations_off"
    output_folder = "datasets/Al/inherent_configurations_off"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    input_files = ["166ps.pos", "170ps.pos", "174ps.pos", "175ps.pos", "177ps.pos", "240ps.pos"]
    input_files_off = ["166ps.off", "170ps.off", "174ps.off", "175ps.off", "177ps.off", "240ps.off"]

    # for in_file in input_files: 
    #     in_file_path = os.path.join(original_folder, in_file)
    #     out_file_path = os.path.join(output_folder, in_file.replace(".pos", ".off"))
    #     convert_xyz_to_off(in_file_path, out_file_path)

    for in_file_off in input_files_off:
        in_file_path = os.path.join(original_folder, in_file_off)
        out_file_path_npz = os.path.join(output_folder, in_file_off.replace(".off", ".npz"))
        convert_off_to_npz(in_file_path, out_file_path_npz)

