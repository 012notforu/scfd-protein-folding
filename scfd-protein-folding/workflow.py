#!/usr/bin/env python3
"""
AlphaFold-SCFD Workflow Manager
===============================

Interactive workflow for processing AlphaFold protein structures 
and visualizing them as symbolic 3D grids for SCFD analysis.

Usage:
    python workflow.py
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from batch_processor import AlphaFoldBatchProcessor
from visualizer import SymbolicProteinVisualizer
import numpy as np

class WorkflowManager:
    """Main workflow manager for AlphaFold-SCFD processing."""
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.base_dir = Path(base_dir)
        self.raw_folder = self.base_dir / "raw"
        self.processed_folder = self.base_dir / "processed"
        
        # Initialize components
        self.processor = AlphaFoldBatchProcessor(
            raw_folder=str(self.raw_folder),
            processed_folder=str(self.processed_folder)
        )
        self.visualizer = SymbolicProteinVisualizer()
        
        print(f"Workflow initialized:")
        print(f"  Base directory: {self.base_dir}")
        print(f"  Raw data: {self.raw_folder}")
        print(f"  Processed data: {self.processed_folder}")
    
    def show_menu(self):
        """Display the main menu."""
        print("\n" + "="*50)
        print("AlphaFold-SCFD Workflow Manager")
        print("="*50)
        print("1. Scan for protein files")
        print("2. Process proteins (batch)")
        print("3. Show processing statistics")
        print("4. List processed files")
        print("5. Visualize protein (3D)")
        print("6. Analyze protein structure")
        print("7. Compare multiple proteins")
        print("8. Export for SCFD analysis")
        print("9. Help")
        print("0. Exit")
        print("-"*50)
    
    def scan_files(self):
        """Scan for available protein files."""
        print("\nScanning for protein files...")
        files = self.processor.find_protein_files()
        
        if not files:
            print("No protein files found!")
            print(f"Please ensure protein files are in: {self.raw_folder}")
            return
        
        print(f"\nFound {len(files)} protein files:")
        
        # Group by directory
        dirs = {}
        for f in files:
            dir_name = os.path.dirname(f)
            if dir_name not in dirs:
                dirs[dir_name] = []
            dirs[dir_name].append(os.path.basename(f))
        
        for dir_path, filenames in dirs.items():
            rel_dir = os.path.relpath(dir_path, self.raw_folder)
            print(f"\n  {rel_dir}/ ({len(filenames)} files)")
            
            # Show first few files as examples
            for fname in filenames[:3]:
                print(f"    - {fname}")
            if len(filenames) > 3:
                print(f"    ... and {len(filenames) - 3} more")
    
    def process_batch(self):
        """Run batch processing with user options."""
        print("\nBatch Processing Options:")
        print("1. Process all files")
        print("2. Process limited number (for testing)")
        print("3. Process specific directory")
        
        choice = input("Select option (1-3): ").strip()
        
        max_files = None
        if choice == '2':
            try:
                max_files = int(input("Number of files to process: "))
            except ValueError:
                print("Invalid number, processing all files...")
        
        # Get processing parameters
        try:
            grid_size = int(input(f"Grid size (default 64): ") or "64")
            voxel_size = float(input(f"Voxel size in Å (default 1.5): ") or "1.5")
        except ValueError:
            print("Using default parameters...")
            grid_size, voxel_size = 64, 1.5
        
        print(f"\nStarting batch processing...")
        print(f"  Grid size: {grid_size}³")
        print(f"  Voxel size: {voxel_size} Å")
        if max_files:
            print(f"  Max files: {max_files}")
        
        results = self.processor.process_batch(
            grid_size=grid_size,
            voxel_size=voxel_size,
            max_files=max_files
        )
        
        return results
    
    def show_statistics(self):
        """Display processing statistics."""
        stats = self.processor.get_file_stats()
        
        if stats["total_files"] == 0:
            print("\nNo processed files found.")
            print("Run batch processing first (option 2).")
            return
        
        print(f"\n=== Processing Statistics ===")
        print(f"Total processed files: {stats['total_files']}")
        print(f"Grid shape: {stats['grid_shape']}")
        print(f"File size: {stats['grid_size_mb']:.1f} MB per file")
        print(f"Total dataset size: {stats['total_size_mb']:.1f} MB")
        
        print(f"\nSymbol distribution (sample):")
        for symbol, count in stats['symbol_counts'].items():
            print(f"  {symbol}: {count:,} voxels")
    
    def list_processed(self):
        """List all processed files."""
        files = self.processor.get_processed_files()
        
        if not files:
            print("\nNo processed files found.")
            return
        
        print(f"\nProcessed files ({len(files)}):")
        for i, file_path in enumerate(files, 1):
            name = os.path.basename(file_path)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  {i:3d}. {name:<40} ({size_mb:.1f} MB)")
    
    def visualize_protein(self):
        """Visualize a specific protein."""
        files = self.processor.get_processed_files()
        
        if not files:
            print("\nNo processed files found.")
            print("Run batch processing first (option 2).")
            return
        
        # Let user select file
        print(f"\nSelect protein to visualize:")
        for i, file_path in enumerate(files, 1):
            name = os.path.basename(file_path).replace('.npy', '')
            print(f"  {i:3d}. {name}")
        
        try:
            choice = int(input(f"\nEnter number (1-{len(files)}): ")) - 1
            if not 0 <= choice < len(files):
                print("Invalid choice.")
                return
        except ValueError:
            print("Invalid input.")
            return
        
        selected_file = files[choice]
        print(f"\nLoading: {os.path.basename(selected_file)}")
        
        # Load and analyze
        grid = self.visualizer.load_grid(selected_file)
        analysis = self.visualizer.analyze_structure(grid)
        
        # Visualization options
        print("\nVisualization options:")
        print("1. 3D scatter plot (matplotlib)")
        print("2. Interactive 3D plot (plotly)")
        print("3. 2D slice views")
        print("4. Density projections")
        print("5. All visualizations")
        
        viz_choice = input("Select visualization (1-5): ").strip()
        
        try:
            import matplotlib.pyplot as plt
            
            if viz_choice in ['1', '5']:
                print("Creating 3D scatter plot...")
                fig_3d, ax_3d = self.visualizer.create_3d_scatter_matplotlib(grid)
                plt.show()
            
            if viz_choice in ['2', '5']:
                print("Creating interactive 3D plot...")
                fig_interactive = self.visualizer.create_3d_scatter_plotly(grid)
                fig_interactive.show()
            
            if viz_choice in ['3', '5']:
                print("Creating slice views...")
                fig_slice, ax_slice = self.visualizer.create_slice_view(grid)
                plt.show()
            
            if viz_choice in ['4', '5']:
                print("Creating density projections...")
                fig_proj, ax_proj = self.visualizer.create_density_projection(grid)
                plt.show()
                
        except ImportError as e:
            print(f"Visualization error: {e}")
            print("Please install required packages: pip install matplotlib plotly")
    
    def analyze_protein(self):
        """Detailed analysis of a protein structure."""
        files = self.processor.get_processed_files()
        
        if not files:
            print("\nNo processed files found.")
            return
        
        # Let user select file  
        print(f"\nSelect protein to analyze:")
        for i, file_path in enumerate(files[:10], 1):  # Show first 10
            name = os.path.basename(file_path).replace('.npy', '')
            print(f"  {i:3d}. {name}")
        
        try:
            choice = int(input(f"\nEnter number (1-{min(10, len(files))}): ")) - 1
            selected_file = files[choice]
        except (ValueError, IndexError):
            print("Invalid choice.")
            return
        
        print(f"\nAnalyzing: {os.path.basename(selected_file)}")
        grid = self.visualizer.load_grid(selected_file)
        analysis = self.visualizer.analyze_structure(grid)
        
        # Additional SCFD-specific analysis
        print(f"\n=== SCFD Compatibility Analysis ===")
        
        # Check alphabet usage
        symbols_present = set(np.unique(grid))
        scfd_symbols = {0, 1, 2}  # Expected SCFD alphabet
        
        if scfd_symbols.issubset(symbols_present):
            print("✓ Compatible with SCFD ternary alphabet")
        else:
            missing = scfd_symbols - symbols_present
            print(f"⚠ Missing symbols for SCFD: {missing}")
        
        # Density analysis for SCFD
        occupancy = analysis['occupancy_ratio']
        if 0.1 <= occupancy <= 0.5:
            print(f"✓ Good density for SCFD dynamics ({occupancy:.1%})")
        elif occupancy < 0.1:
            print(f"⚠ Low density ({occupancy:.1%}) - may need smaller voxels")
        else:
            print(f"⚠ High density ({occupancy:.1%}) - may be too crowded")
    
    def export_for_scfd(self):
        """Export processed data for SCFD analysis."""
        files = self.processor.get_processed_files()
        
        if not files:
            print("\nNo processed files found.")
            return
        
        # Create SCFD export directory
        scfd_dir = self.base_dir / "scfd_ready"
        scfd_dir.mkdir(exist_ok=True)
        
        print(f"\nExporting {len(files)} files for SCFD analysis...")
        
        export_info = {
            'grid_size': None,
            'voxel_size': 1.5,  # Default, could be made configurable
            'alphabet': {0: 'hydrophobic', 1: 'polar', 2: 'charged'},
            'files': []
        }
        
        for file_path in files:
            grid = np.load(file_path)
            
            if export_info['grid_size'] is None:
                export_info['grid_size'] = list(grid.shape)
            
            # Copy to SCFD directory with metadata
            base_name = os.path.basename(file_path)
            new_path = scfd_dir / base_name
            np.save(new_path, grid)
            
            # Add file info
            unique, counts = np.unique(grid, return_counts=True)
            file_info = {
                'filename': base_name,
                'protein_id': base_name.replace('.npy', ''),
                'occupancy': np.sum(grid != -1) / grid.size,
                'symbol_counts': dict(zip([int(u) for u in unique], [int(c) for c in counts]))
            }
            export_info['files'].append(file_info)
        
        # Save export metadata
        import json
        with open(scfd_dir / "export_metadata.json", 'w') as f:
            json.dump(export_info, f, indent=2)
        
        print(f"✓ Exported to: {scfd_dir}")
        print(f"✓ Created metadata file: export_metadata.json")
        print(f"✓ Ready for SCFD analysis with {len(files)} proteins")
    
    def show_help(self):
        """Display help information."""
        print("""
=== AlphaFold-SCFD Workflow Help ===

This workflow processes AlphaFold protein structures into symbolic 3D grids
suitable for SCFD (Symbolic Coherence Field Dynamics) analysis.

Processing Pipeline:
1. AlphaFold structures (.pdb/.cif files) 
   ↓
2. 3D voxelization (64³ grid, 1.5Å resolution)
   ↓  
3. Symbolic mapping (hydrophobic=0, polar=1, charged=2)
   ↓
4. SCFD-ready discrete lattice

Key Features:
- Batch processing of multiple proteins
- 3D visualization (matplotlib + plotly)  
- Structure analysis and statistics
- Direct export for SCFD experiments

Required Libraries:
- biopython (structure parsing)
- numpy (arrays and processing)
- matplotlib (basic visualization)
- plotly (interactive 3D plots)
- tqdm (progress bars)

File Organization:
- raw/ : Original AlphaFold files (.pdb, .cif, .gz)
- processed/ : Symbolic grids (.npy files)
- scfd_ready/ : Exported data with metadata

For SCFD Integration:
The exported grids use a ternary alphabet {0,1,2} representing
biochemical properties, making them directly compatible with
your SCFD framework's symbolic lattice operations.
""")
    
    def run(self):
        """Run the interactive workflow."""
        while True:
            try:
                self.show_menu()
                choice = input("\nSelect option: ").strip()
                
                if choice == '0':
                    print("Goodbye!")
                    break
                elif choice == '1':
                    self.scan_files()
                elif choice == '2':
                    self.process_batch()
                elif choice == '3':
                    self.show_statistics()
                elif choice == '4':
                    self.list_processed()
                elif choice == '5':
                    self.visualize_protein()
                elif choice == '6':
                    self.analyze_protein()
                elif choice == '7':
                    print("Multi-protein comparison coming soon...")
                elif choice == '8':
                    self.export_for_scfd()
                elif choice == '9':
                    self.show_help()
                else:
                    print("Invalid choice. Please try again.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                input("Press Enter to continue...")

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="AlphaFold-SCFD Workflow Manager")
    parser.add_argument('--base-dir', help='Base directory for workflow')
    parser.add_argument('--batch', action='store_true', help='Run batch processing non-interactively')
    parser.add_argument('--max-files', type=int, help='Maximum files to process in batch mode')
    
    args = parser.parse_args()
    
    workflow = WorkflowManager(args.base_dir)
    
    if args.batch:
        print("Running in batch mode...")
        workflow.processor.process_batch(max_files=args.max_files)
    else:
        workflow.run()

if __name__ == "__main__":
    main()