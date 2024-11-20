import os

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
        if len(parts) >= 5:  # Ensure line has enough components
            _, _, x, y, z = parts[:5]  # We only need x, y, z coordinates
            atoms.append([float(x), float(y), float(z)])
    
    return atoms

def write_off_file(atoms, output_filename):
    """Write atomic positions to OFF format."""
    with open(output_filename, 'w') as f:
        # Write header
        f.write("OFF\n")
        
        # Write number of vertices, faces, and edges
        # For now, we're just writing vertices (no faces/edges)
        f.write(f"{len(atoms)} 0 0\n")
        
        # Write vertex coordinates
        for x, y, z in atoms:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

def convert_xyz_to_off(input_filename, output_filename):
    """Convert XYZ format to OFF format."""
    atoms = read_xyz_file(input_filename)
    
    write_off_file(atoms, output_filename)
    
    print(f"Conversion complete! Wrote {len(atoms)} vertices to {output_filename}")


if __name__ == "__main__":
    original_folder = "datasets/Al/inherent_configurations"
    output_folder = "datasets/Al/inherent_configurations_off"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    input_files = ["166ps.pos", "170ps.pos", "240ps.pos"]

    for in_file in input_files: 
        in_file_path = os.path.join(original_folder, in_file)
        out_file_path = os.path.join(output_folder, in_file.replace(".pos", ".off"))
        convert_xyz_to_off(in_file_path, out_file_path)
