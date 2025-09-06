#!/usr/bin/env python3
"""
Quick test of the SCFD folding system using existing processed proteins.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from pathlib import Path
from src.scfd_folding_pipeline import SCFDFoldingSimulator
from src.scfd_pathway_exporter import SCFDPathwayExporter

def test_single_protein_folding():
    """Test folding simulation on a single protein."""
    
    # Load an existing processed protein
    processed_files = list(Path("processed").glob("*.npy"))
    if not processed_files:
        print("No processed protein files found!")
        return
        
    test_protein = processed_files[0]
    print(f"Testing with protein: {test_protein.stem}")
    
    # Load the protein grid
    original_grid = np.load(test_protein)
    print(f"Loaded protein grid: {original_grid.shape}")
    print(f"Occupied voxels: {np.sum(original_grid != -1)}")
    print(f"Unique symbols: {np.unique(original_grid[original_grid != -1])}")
    
    # Initialize simulator and exporter
    simulator = SCFDFoldingSimulator(alphabet_type='biochemical_12', temperature=1.0)
    exporter = SCFDPathwayExporter(
        protein_id=test_protein.stem + "_test",
        alphabet_type='biochemical_12',
        output_dir='training_data'
    )
    
    # Create unfolded starting configuration (randomize symbols)
    unfolded_grid = original_grid.copy()
    occupied_positions = np.where(original_grid != -1)
    symbols = original_grid[occupied_positions]
    np.random.shuffle(symbols)
    unfolded_grid[occupied_positions] = symbols
    
    print("Starting folding simulation...")
    
    # Set up exporter
    sequence = ''.join(['A'] * len(symbols))  # Simplified sequence
    exporter.set_sequence_info(sequence, original_grid)
    exporter.set_simulation_params({
        'grid_size': original_grid.shape[0],
        'temperature': simulator.temperature,
        'max_timesteps': 20
    })
    
    # Run simulation
    timestep_count = 0
    for timestep_data in simulator.run_folding_simulation(unfolded_grid, max_timesteps=20):
        timestep_count += 1
        
        # Log this timestep
        exporter.log_timestep(
            timestep_data['timestep'],
            timestep_data['grid'],
            timestep_data['fields'],
            timestep_data['mutations'],
            timestep_data['energies']
        )
        
        print(f"Timestep {timestep_data['timestep']}: "
              f"Energy = {timestep_data['energies']['total']:.2f}, "
              f"Mutations = {len(timestep_data['mutations'])}")
              
        # Log individual mutations
        for mut in timestep_data['mutations']:
            exporter.log_mutation(
                mut['position'],
                mut['from_symbol'], 
                mut['to_symbol'],
                mut['energy_delta'],
                mut['reason'],
                {'neighborhood': [], 'coherence_before': 0, 'coherence_after': 0}
            )
    
    # Finalize and export
    final_grid = timestep_data['grid'] if timestep_count > 0 else unfolded_grid
    exporter.finalize_simulation(
        final_grid,
        {'converged': True, 'timesteps': timestep_count},
        folding_success=True
    )
    
    # Export data
    trajectory_file = exporter.export_to_json()
    summary_file = exporter.export_compressed_summary()
    
    print(f"\nSimulation completed!")
    print(f"Trajectory data: {trajectory_file}")
    print(f"Summary data: {summary_file}")
    
    # Show some stats
    print(f"\nFolding Statistics:")
    print(f"Total timesteps: {timestep_count}")
    print(f"Final energy: {timestep_data['energies']['total']:.2f}" if timestep_count > 0 else "N/A")
    
    return trajectory_file, summary_file

def test_alphabet_conversion():
    """Test the enhanced alphabet system."""
    from src.enhanced_alphabets import convert_residue_to_symbol, get_symbol_description
    
    print("\nTesting Enhanced Alphabet System:")
    test_residues = ['ALA', 'LEU', 'SER', 'ASP', 'LYS', 'GLY', 'PRO']
    
    for alphabet_type in ['ternary', 'biochemical_12']:
        print(f"\n{alphabet_type.upper()} alphabet:")
        for residue in test_residues:
            symbol = convert_residue_to_symbol(residue, alphabet_type)
            description = get_symbol_description(symbol, alphabet_type)
            print(f"  {residue} -> {symbol} ({description})")

if __name__ == '__main__':
    print("="*60)
    print("SCFD Folding System Test")
    print("="*60)
    
    # Test alphabet system
    test_alphabet_conversion()
    
    # Test folding simulation
    print("\n" + "="*60)
    print("Running Folding Simulation Test")
    print("="*60)
    
    try:
        # Generate multiple trajectories for AI training
        for i in range(3):
            print(f"\nGenerating trajectory {i+1}/3...")
            trajectory_file, summary_file = test_single_protein_folding()
            
        print(f"\nTest completed successfully!")
        print(f"Generated 3 trajectories in training_data/ directory.")
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()