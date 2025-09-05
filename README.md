# SCFD Protein Folding Research

Exploring protein folding through symbolic coherence field dynamics - a computational approach that represents proteins as discrete symbolic fields evolving according to physics-based rules.

## What's Here

This repository contains an experimental framework that applies **Symbolic Coherence Field Dynamics (SCFD)** to protein folding simulation. The work explores whether discrete symbolic representations can capture essential folding physics while being computationally efficient.

### Main Project: `scfd-protein-folding/` folder

The complete research framework including:

- **Interactive Pipeline** - `workflow.py` command-line tool for processing AlphaFold structures
- **Core Framework** - `src/` folder with protein processing and visualization tools  
- **Validation Experiments** - `experiments/` folder with blind folding tests and controls
- **Documentation** - `docs/` folder with detailed methodology and results
- **Sample Data** - `data/` folder with processed protein examples

## Key Innovation: The Interactive Workflow

The `workflow.py` script transforms complex protein analysis into an intuitive process. Run it and get a menu-driven interface for:

- Auto-discovering AlphaFold files (.pdb, .cif, .gz formats)
- Batch processing proteins into symbolic grids
- Interactive 3D visualization  
- Statistical analysis and export to various formats

Think of it as making advanced protein structure analysis as accessible as using any command-line application.

## SCFD Approach

**Traditional molecular dynamics**: Simulate every atom with precise physics  
**SCFD approach**: Represent amino acids as biochemical symbols, apply interaction rules

### The Symbolic Mapping
- **Hydrophobic** amino acids (ALA, VAL, ILE, etc.) → Symbol 0
- **Polar** amino acids (GLY, SER, THR, etc.) → Symbol 1  
- **Charged** amino acids (ASP, GLU, LYS, etc.) → Symbol 2
- **Empty space** → Symbol -1

### The Physics
Interaction energies derived from experimental biochemistry (oil/water partition coefficients, hydrogen bonding energies, electrostatic interactions) drive symbolic dynamics on a 64×64×64 voxel grid at 1.5Å resolution.

## Current Results

**Preliminary validation shows:**
- Energy-driven folding from random starting states (energy reduction from +1120 to -174)
- Different proteins produce distinct symbolic signatures
- Control tests with random parameters fail to fold properly
- Processing speeds approximately 1000x faster than molecular dynamics

**Important caveat:** This is exploratory research. The biological relevance of symbolic protein folding requires extensive validation.


## Getting Started

1. Navigate to the `scfd-protein-folding/` folder
2. Install requirements: `pip install -r requirements.txt`
3. Download AlphaFold .pdb or .cif files to `data/raw/`
4. Run the interactive workflow: `python workflow.py`
5. Follow the menu to process and visualize proteins

The workflow guide in the docs folder provides detailed instructions for processing AlphaFold proteome datasets.

## Why This Approach?

If symbolic dynamics can capture essential folding physics, it could enable:
- Rapid screening of protein stability predictions
- Understanding folding mechanisms through simpler computational models
- Integration with machine learning approaches for protein design
- Accessible tools for researchers without access to supercomputing resources

## Current Limitations

- Limited validation on small protein set (E. coli proteins)
- Coarse-grained representation may miss important details
- No experimental folding validation yet
- Simplified biochemical alphabet needs refinement

## Collaboration Welcome

This work sits at the intersection of computational biology, biophysics, complex systems, and scientific software development. We're actively seeking:

- Testing on diverse protein datasets
- Experimental validation collaborations
- Methodological improvements and extensions
- Integration with existing computational biology workflows

The goal is to explore whether this symbolic approach captures meaningful aspects of protein folding while remaining computationally tractable for large-scale studies.
