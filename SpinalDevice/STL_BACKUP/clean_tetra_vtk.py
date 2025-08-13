def strip_non_tetra(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Read header
    header = lines[:4]

    points = []
    raw_cells = []
    raw_types = []

    i = 4
    while i < len(lines):
        line = lines[i]

        if line.startswith("POINTS"):
            points.append(line)
            num_pts = int(line.split()[1])
            points += lines[i+1:i+1+num_pts]
            i += 1 + num_pts

        elif line.startswith("CELLS"):
            num_cells = int(line.split()[1])
            raw_cells = lines[i+1:i+1+num_cells]
            i += 1 + num_cells

        elif line.startswith("CELL_TYPES"):
            num_types = int(line.split()[1])
            raw_types = [int(lines[j].strip()) for j in range(i+1, i+1+num_types)]
            i += 1 + num_types

        else:
            i += 1

    # Filter tetrahedra only
    tetra_cells = []
    tetra_types = []
    for cell_line, ctype in zip(raw_cells, raw_types):
        if ctype == 10:  # 10 is tetrahedron cell type
            tetra_cells.append(cell_line)
            tetra_types.append("10\n")

    # Write output VTK
    with open(output_file, 'w') as f:
        f.writelines(header)
        f.writelines(points)
        f.write(f"CELLS {len(tetra_cells)} {len(tetra_cells)*5}\n")  # 4 nodes + 1 count = 5 per cell
        f.writelines(tetra_cells)
        f.write(f"CELL_TYPES {len(tetra_types)}\n")
        f.writelines(tetra_types)

    print(f"Written {len(tetra_cells)} tetrahedral cells to {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python clean_tetra_vtk.py input.vtk output.vtk")
        sys.exit(1)
    strip_non_tetra(sys.argv[1], sys.argv[2])
