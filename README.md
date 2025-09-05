# SCFD Protein Folding Research

Exploring protein folding through symbolic coherence field dynamics - a computational approach that represents proteins as discrete symbolic fields evolving according to physics-based rules.

![Protein 3D Visualization](scfd-protein-folding/images/protein_3d_view.png)

## What's Here

This repository contains an experimental framework that applies **Symbolic Coherence Field Dynamics (SCFD)** to protein folding simulation. The work explores whether discrete symbolic representations can capture essential folding physics while being computationally efficient.

### 🎯 **Main Project: [`scfd-protein-folding/`](scfd-protein-folding/)**

The complete research framework including:

- **Interactive Pipeline** (`workflow.py`) - Command-line tool for processing AlphaFold structures
- **Core Framework** (`src/`) - Protein processing and visualization tools  
- **Validation Experiments** (`experiments/`) - Blind folding tests and controls
- **Documentation** (`docs/`) - Detailed methodology and results

![Protein Analysis Overview](scfd-protein-folding/images/protein_analysis_comprehensive.png)

## Key Innovation: The Interactive Workflow

The `workflow.py` script transforms complex protein analysis into an intuitive process:

```bash
python workflow.py
```

Provides a menu-driven interface for:
- 🔍 Auto-discovering AlphaFold files
- ⚡ Batch processing proteins into symbolic grids
- 📊 Interactive 3D visualization  
- 📈 Statistical analysis and export

Think of it as making advanced protein structure analysis as easy as using a command-line app.

## SCFD Approach

**Traditional molecular dynamics**: Simulate every atom with precise physics
**SCFD approach**: Represent amino acids as biochemical symbols, apply interaction rules

### The Mapping
- **Hydrophobic** amino acids → Symbol 0
- **Polar** amino acids → Symbol 1  
- **Charged** amino acids → Symbol 2
- **Empty space** → Symbol -1

### The Physics
Interaction energies derived from experimental biochemistry (partition coefficients, hydrogen bonding, electrostatics) drive symbolic dynamics on a 3D lattice.

![Folding Simulation Results](scfd-protein-folding/images/blind_folding_results.png)

## Current Status

**Preliminary validation shows:**
- Energy-driven folding from random starting states
- Different proteins produce distinct symbolic signatures
- Control tests with random parameters fail to fold
- ~1000x computational speedup vs molecular dynamics

**Important caveat:** This is exploratory research. The biological relevance of symbolic protein folding requires extensive validation.

## Getting Started

1. **Explore the main framework**: Navigate to [`scfd-protein-folding/`](scfd-protein-folding/)
2. **Read the documentation**: Check [`docs/WORKFLOW_GUIDE.md`](scfd-protein-folding/docs/WORKFLOW_GUIDE.md)
3. **See the validation**: Review [`docs/VALIDATION_RESULTS_SUMMARY.md`](scfd-protein-folding/docs/VALIDATION_RESULTS_SUMMARY.md)
4. **Try the pipeline**: Download AlphaFold data and run `workflow.py`

## Why This Matters

If symbolic dynamics can capture folding physics, it could enable:
- Rapid screening of protein stability
- Understanding folding mechanisms through simpler models
- Integration with machine learning approaches
- Accessible tools for protein structure analysis

## Collaboration Welcome

This work sits at the intersection of:
- **Computational Biology** - Novel protein simulation approaches
- **Biophysics** - Understanding folding mechanisms  
- **Complex Systems** - Symbolic dynamics applications
- **Software Engineering** - User-friendly scientific tools

We're actively seeking collaborators, feedback, and validation from the community.

---

**🚀 Ready to dive in? Start with the main framework:** [`scfd-protein-folding/`](scfd-protein-folding/)

**📚 Want the full story? Read the documentation:** [`docs/`](scfd-protein-folding/docs/)

**⚗️ Curious about the science? Check the experiments:** [`experiments/`](scfd-protein-folding/experiments/)
