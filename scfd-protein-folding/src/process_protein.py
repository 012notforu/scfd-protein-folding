import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
import os

# --- 1. Define Your "Biochemical Alphabet" ---
AMINO_ACID_MAP = {
    'ALA': 0, 'VAL': 0, 'ILE': 0, 'LEU': 0, 'MET': 0, 'PHE': 0, 'TYR': 0, 'TRP': 0,
    'GLY': 1, 'SER': 1, 'THR': 1, 'CYS': 1, 'ASN': 1, 'GLN': 1, 'PRO': 1,
    'ASP': 2, 'GLU': 2, 'LYS': 2, 'ARG': 2, 'HIS': 2,
}

def get_protein_structure(file_path):
    """Loads a protein structure from a PDB or mmCIF file."""
    parser = MMCIFParser(QUIET=True) if file_path.endswith('.cif') else PDBParser(QUIET=True)
    file_id = os.path.basename(file_path).split('.')[0]
    return parser.get_structure(file_id, file_path)

def voxelize_protein_to_symbolic_grid(structure, grid_size=64, voxel_size=1.5):
    """Converts a Biopython structure into a 3D symbolic grid."""
    atoms = [atom for atom in structure.get_atoms()]
    if not atoms:
        print("Warning: No atoms found in structure.")
        return np.full((grid_size, grid_size, grid_size), -1, dtype=np.int8)

    coords = np.array([atom.get_coord() for atom in atoms])
    min_coords = coords.min(axis=0)

    symbolic_grid = np.full((grid_size, grid_size, grid_size), -1, dtype=np.int8)

    for atom in atoms:
        res_name = atom.get_parent().get_resname()
        symbol = AMINO_ACID_MAP.get(res_name.strip().upper(), -1)

        if symbol == -1:
            continue

        x, y, z = atom.get_coord()
        grid_x = int((x - min_coords[0]) / voxel_size)
        grid_y = int((y - min_coords[1]) / voxel_size)
        grid_z = int((z - min_coords[2]) / voxel_size)

        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size and 0 <= grid_z < grid_size:
            symbolic_grid[grid_x, grid_y, grid_z] = symbol

    return symbolic_grid

# --- Main execution ---
if __name__ == '__main__':
    # --- PATHS CORRECTED TO BE RUN FROM THE MAIN PROJECT FOLDER ---
    # The file is inside a folder with the same name as the file
    raw_data_folder = 'raw/UP000000625_83333_ECOLI_v4/AF-A0A385XJ53-F1-model_v4.cif/'
    processed_data_folder = 'processed/'

    # --- ACTION: Paste the EXACT filename you copied in Step 2 here ---
    example_filename = 'AF-A0A385XJ53-F1-model_v4.cif'

    input_filepath = os.path.join(raw_data_folder, example_filename)
    output_filename = example_filename.replace('.cif', '.npy')
    output_filepath = os.path.join(processed_data_folder, output_filename)

    os.makedirs(processed_data_folder, exist_ok=True)

    print(f"Loading structure from: {input_filepath}")
    protein_structure = get_protein_structure(input_filepath)

    print("Voxelizing protein and mapping to symbolic alphabet...")
    symbolic_protein_grid = voxelize_protein_to_symbolic_grid(protein_structure)

    print(f"Saving symbolic grid to: {output_filepath}")
    np.save(output_filepath, symbolic_protein_grid)

    print("\n--- Processing Complete ---")
    print(f"A new file '{output_filename}' has been saved in your 'processed' folder.")