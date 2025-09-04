# SCFD Workflow Guide

## Overview

The `workflow.py` script provides an interactive command-line interface for processing AlphaFold protein structures and converting them into symbolic 3D grids for SCFD analysis. It's like having Claude Code for protein folding experiments!

## Features

- 🔍 **Auto-discovery**: Scans for AlphaFold files (.pdb, .cif, .gz formats)
- ⚡ **Batch Processing**: Process multiple proteins automatically
- 📊 **3D Visualization**: Interactive and static 3D plotting
- 📈 **Statistics**: Detailed analysis of processed proteins
- 🎯 **SCFD Export**: Direct export for symbolic coherence field dynamics

## Quick Start

### 1. Setup Directory Structure

```
your-project/
├── workflow.py
├── src/
├── raw/              # Place AlphaFold files here
└── processed/        # Generated symbolic grids appear here
```

### 2. Download AlphaFold Data

Visit the [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/) and download:

- **Individual proteins**: Download .pdb.gz files
- **Proteome datasets**: Download complete organism datasets
- **Custom sets**: Search and download specific protein families

**Supported formats:**
- `.pdb` and `.pdb.gz` (Protein Data Bank format)
- `.cif` and `.cif.gz` (Crystallographic Information File)

### 3. Run the Interactive Workflow

```bash
python workflow.py
```

You'll see a menu like this:

```
==================================================
         AlphaFold-SCFD Workflow Menu
==================================================
1. Scan for AlphaFold files in raw/
2. Process batch of proteins (with limit)
3. Process all files
4. Show processing statistics
5. Visualize protein (3D interactive)
6. Visualize protein (3D static)
7. Export to SCFD format
8. Exit
==================================================
```

### 4. Typical Workflow

1. **Option 1**: Scan files to see what you have
2. **Option 2**: Process a small batch first (e.g., 5 proteins)
3. **Option 4**: Check statistics and quality
4. **Option 5**: Visualize results in 3D
5. **Option 7**: Export for SCFD analysis

## Processing Details

### Symbolic Mapping

Each amino acid is converted to a symbol based on biochemical properties:

- **Symbol 0 (Hydrophobic)**: ALA, VAL, ILE, LEU, MET, PHE, TYR, TRP
- **Symbol 1 (Polar)**: GLY, SER, THR, CYS, ASN, GLN, PRO
- **Symbol 2 (Charged)**: ASP, GLU, LYS, ARG, HIS
- **Symbol -1**: Empty space

### Grid Parameters

- **Grid Size**: 64×64×64 voxels (adjustable)
- **Voxel Size**: 1.5 Å resolution
- **Output Format**: NumPy arrays (.npy files)

### Batch Processing Tips

- Start with small batches (5-10 proteins) to test
- Large proteomes can take hours - use limits
- Processing speed: ~1-5 seconds per protein
- Memory usage: ~256 KB per processed protein

## Visualization Options

### Interactive 3D (Plotly)
```bash
# Menu option 5
# - Hover for amino acid information
# - Rotate, zoom, pan
# - Color-coded by biochemical type
```

### Static 3D (Matplotlib)
```bash
# Menu option 6
# - High-quality static plots
# - Good for publications
# - Faster rendering
```

## SCFD Integration

The workflow automatically generates SCFD-ready data:

```python
# After processing, use with SCFD frameworks
import numpy as np

# Load processed protein
protein_grid = np.load('processed/AF-P00350-F1-model_v4.npy')

# Your SCFD framework can now use this symbolic grid
# as initial conditions for folding simulations
```

## Example Session

```bash
$ python workflow.py

==================================================
         AlphaFold-SCFD Workflow Menu
==================================================

Choose option: 1
Scanning for AlphaFold files...
Found 247 files in raw/UP000000625_83333_ECOLI_v4/

Choose option: 2  
How many files to process? 10

Processing batch of 10 files...
Processing AF-P00350-F1-model_v4.pdb.gz... ✓
[... progress continues ...]
Batch processing complete!

Choose option: 5
Available processed proteins:
1. AF-P00350-F1-model_v4.npy
Choose file number: 1
Generating interactive 3D visualization...
[Opens browser with 3D protein visualization]
```

## Troubleshooting

**"No files found"**: Check that AlphaFold files are in the `raw/` directory

**"Processing failed"**: Some AlphaFold files may be corrupted - skip and continue

**"Memory issues"**: Reduce batch size or grid resolution

**"Visualization not working"**: Install optional packages: `pip install plotly matplotlib`

## Advanced Usage

### Command Line Arguments

```bash
# Custom directory
python workflow.py --base-dir /path/to/data

# Batch mode (non-interactive)
python workflow.py --batch --max-files 50

# Custom grid size
python workflow.py --grid-size 32  # Smaller, faster processing
```

### Integration with Other Tools

The symbolic grids are compatible with:
- SCFD simulation frameworks
- Machine learning protein models
- Custom analysis scripts
- Visualization tools

This workflow transforms the complex process of protein structure analysis into an intuitive, interactive experience - making cutting-edge protein folding research accessible to everyone!