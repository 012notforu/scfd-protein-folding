# Processed Protein Data

This directory contains protein structures converted to symbolic 3D grids for SCFD analysis.

## File Format

- **Extension**: `.npy` (NumPy arrays)
- **Structure**: 64×64×64 integer arrays  
- **Values**: 
  - `0`: Hydrophobic amino acids
  - `1`: Polar amino acids
  - `2`: Charged amino acids
  - `-1`: Empty space

## Sample Data

This repository includes sample processed proteins used in validation experiments.

## Usage

```python
import numpy as np

# Load a processed protein
protein_grid = np.load('AF-P00350-F1-model_v4.npy')

# Check dimensions
print(f"Grid shape: {protein_grid.shape}")  # Should be (64, 64, 64)

# Count amino acids by type
hydrophobic = np.sum(protein_grid == 0)
polar = np.sum(protein_grid == 1) 
charged = np.sum(protein_grid == 2)
```

## Generating More Data

Use the `workflow.py` script to process additional AlphaFold structures:

1. Download AlphaFold .pdb/.cif files to `../raw/`
2. Run `python workflow.py`
3. Select batch processing option
4. Processed files will appear here automatically