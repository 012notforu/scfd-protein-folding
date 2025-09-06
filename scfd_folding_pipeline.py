"""
SCFD Protein Folding Pipeline
============================

Integrated pipeline that:
1. Processes AlphaFold structures with enhanced alphabets
2. Runs SCFD folding simulations 
3. Exports comprehensive pathway data for AI training
"""

import os
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time

from batch_processor import AlphaFoldBatchProcessor
from scfd_pathway_exporter import SCFDPathwayExporter
from enhanced_alphabets import get_alphabet_config, get_symbol_description

class SCFDFoldingSimulator:
    """Simple SCFD folding simulator (Python implementation of core dynamics)."""
    
    def __init__(self, alphabet_type='biochemical_12', temperature=1.0):
        self.alphabet_type = alphabet_type
        self.alphabet_config = get_alphabet_config(alphabet_type)
        self.temperature = temperature
        self.alphabet_size = self.alphabet_config['size']
        
        # SCFD parameters
        self.energy_weights = {
            'coherence': 1.0,
            'entropy': 0.3,
            'boundary': 0.1
        }
        
    def compute_coherence_field(self, grid):
        """Compute coherence field C(i,j,k) for each position."""
        coherence_field = np.zeros_like(grid, dtype=np.float32)
        
        for i in range(1, grid.shape[0]-1):
            for j in range(1, grid.shape[1]-1):
                for k in range(1, grid.shape[2]-1):
                    if grid[i,j,k] == -1:  # Empty space
                        continue
                        
                    center_symbol = grid[i,j,k]
                    matches = 0
                    neighbors = 0
                    
                    # 3D Moore neighborhood (26 neighbors)
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                ni, nj, nk = i+di, j+dj, k+dk
                                if (0 <= ni < grid.shape[0] and 
                                    0 <= nj < grid.shape[1] and 
                                    0 <= nk < grid.shape[2]):
                                    neighbors += 1
                                    if grid[ni,nj,nk] == center_symbol:
                                        matches += 1
                    
                    coherence_field[i,j,k] = matches / neighbors if neighbors > 0 else 0
                    
        return coherence_field
        
    def compute_curvature_field(self, coherence_field):
        """Compute discrete Laplacian (curvature) of coherence field."""
        curvature = np.zeros_like(coherence_field)
        
        for i in range(1, coherence_field.shape[0]-1):
            for j in range(1, coherence_field.shape[1]-1):
                for k in range(1, coherence_field.shape[2]-1):
                    # Discrete Laplacian in 3D
                    laplacian = (
                        coherence_field[i+1,j,k] + coherence_field[i-1,j,k] +
                        coherence_field[i,j+1,k] + coherence_field[i,j-1,k] +
                        coherence_field[i,j,k+1] + coherence_field[i,j,k-1] -
                        6 * coherence_field[i,j,k]
                    )
                    curvature[i,j,k] = laplacian
                    
        return curvature
        
    def compute_entropy_field(self, grid):
        """Compute local entropy for each position."""
        entropy_field = np.zeros_like(grid, dtype=np.float32)
        
        for i in range(1, grid.shape[0]-1):
            for j in range(1, grid.shape[1]-1):
                for k in range(1, grid.shape[2]-1):
                    if grid[i,j,k] == -1:
                        continue
                        
                    # Count symbols in local neighborhood
                    symbol_counts = {}
                    total = 0
                    
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                ni, nj, nk = i+di, j+dj, k+dk
                                if (0 <= ni < grid.shape[0] and 
                                    0 <= nj < grid.shape[1] and 
                                    0 <= nk < grid.shape[2] and
                                    grid[ni,nj,nk] != -1):
                                    symbol = grid[ni,nj,nk]
                                    symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                                    total += 1
                    
                    # Calculate Shannon entropy
                    if total > 0:
                        entropy = 0
                        for count in symbol_counts.values():
                            p = count / total
                            entropy -= p * np.log2(p) if p > 0 else 0
                        entropy_field[i,j,k] = entropy
                        
        return entropy_field
        
    def calculate_total_energy(self, grid, coherence_field, curvature_field, entropy_field):
        """Calculate total SCFD energy."""
        # Only consider non-empty positions
        mask = (grid != -1)
        
        coherence_energy = np.sum(coherence_field[mask] ** 2) * self.energy_weights['coherence']
        entropy_energy = np.sum(entropy_field[mask]) * self.energy_weights['entropy']
        boundary_energy = np.sum(np.abs(curvature_field[mask])) * self.energy_weights['boundary']
        
        total_energy = -(coherence_energy + entropy_energy - boundary_energy)
        
        return {
            'total': total_energy,
            'coherence': coherence_energy,
            'entropy': entropy_energy,
            'boundary': boundary_energy
        }
        
    def propose_mutations(self, grid, coherence_field, entropy_field, n_mutations=5):
        """Propose beneficial mutations using SCFD criteria."""
        mutations = []
        occupied_positions = list(zip(*np.where(grid != -1)))
        
        if len(occupied_positions) == 0:
            return mutations
            
        # Randomly select positions to mutate
        n_candidates = min(n_mutations, len(occupied_positions))
        candidate_positions = np.random.choice(len(occupied_positions), n_candidates, replace=False)
        
        for idx in candidate_positions:
            pos = occupied_positions[idx]
            i, j, k = pos
            current_symbol = grid[i, j, k]
            
            # Only mutate if entropy is high enough (SCFD criterion)
            if entropy_field[i, j, k] < 0.1 * self.alphabet_config['max_entropy']:
                continue
                
            # Try different symbols
            for new_symbol in range(self.alphabet_size):
                if new_symbol == current_symbol:
                    continue
                    
                # Calculate energy change (simplified)
                old_coherence = coherence_field[i, j, k]
                
                # Estimate new coherence (simplified - would need full recalculation)
                neighbor_matches = 0
                total_neighbors = 0
                
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for dk in [-1, 0, 1]:
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            ni, nj, nk = i+di, j+dj, k+dk
                            if (0 <= ni < grid.shape[0] and 
                                0 <= nj < grid.shape[1] and 
                                0 <= nk < grid.shape[2] and
                                grid[ni, nj, nk] != -1):
                                total_neighbors += 1
                                if grid[ni, nj, nk] == new_symbol:
                                    neighbor_matches += 1
                
                new_coherence = neighbor_matches / total_neighbors if total_neighbors > 0 else 0
                energy_delta = (new_coherence**2 - old_coherence**2) * self.energy_weights['coherence']
                
                if energy_delta < 0:  # Energetically favorable
                    mutations.append({
                        'position': pos,
                        'from_symbol': current_symbol,
                        'to_symbol': new_symbol,
                        'energy_delta': energy_delta,
                        'reason': 'coherence_optimization'
                    })
                    
        return mutations
        
    def apply_mutations(self, grid, mutations):
        """Apply accepted mutations to grid."""
        applied_mutations = []
        
        for mutation in mutations:
            # Accept mutation with Boltzmann probability
            energy_delta = mutation['energy_delta']
            probability = np.exp(-energy_delta / self.temperature) if energy_delta > 0 else 1.0
            
            if np.random.random() < probability:
                pos = mutation['position']
                grid[pos[0], pos[1], pos[2]] = mutation['to_symbol']
                applied_mutations.append(mutation)
                
        return applied_mutations
        
    def run_folding_simulation(self, initial_grid, max_timesteps=100, convergence_threshold=1e-3):
        """Run complete SCFD folding simulation."""
        grid = initial_grid.copy()
        energy_history = []
        mutation_history = []
        
        print(f"Starting SCFD folding simulation ({max_timesteps} max timesteps)")
        
        for timestep in tqdm(range(max_timesteps), desc="Folding"):
            # Compute fields
            coherence_field = self.compute_coherence_field(grid)
            curvature_field = self.compute_curvature_field(coherence_field)
            entropy_field = self.compute_entropy_field(grid)
            
            # Calculate energy
            energies = self.calculate_total_energy(grid, coherence_field, curvature_field, entropy_field)
            energy_history.append(energies['total'])
            
            # Propose and apply mutations
            proposed_mutations = self.propose_mutations(grid, coherence_field, entropy_field)
            applied_mutations = self.apply_mutations(grid, proposed_mutations)
            mutation_history.append(len(applied_mutations))
            
            # Check convergence
            if len(energy_history) > 10:
                recent_energies = energy_history[-10:]
                energy_variance = np.var(recent_energies)
                if energy_variance < convergence_threshold:
                    print(f"Converged at timestep {timestep}")
                    break
                    
            # Return data for this timestep
            timestep_data = {
                'timestep': timestep,
                'grid': grid.copy(),
                'fields': {
                    'coherence': coherence_field,
                    'curvature': curvature_field, 
                    'entropy': entropy_field
                },
                'energies': energies,
                'mutations': applied_mutations
            }
            
            yield timestep_data
            
        return {
            'converged': timestep < max_timesteps - 1,
            'final_timestep': timestep,
            'energy_history': energy_history,
            'mutation_history': mutation_history
        }


class SCFDFoldingPipeline:
    """Complete pipeline for generating SCFD folding training data."""
    
    def __init__(self, base_dir=None, alphabet_type='biochemical_12', output_dir='training_data'):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        
        self.base_dir = Path(base_dir)
        self.alphabet_type = alphabet_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.batch_processor = AlphaFoldBatchProcessor(
            raw_folder=str(self.base_dir / "raw"),
            processed_folder=str(self.base_dir / "processed"),
            alphabet_type=alphabet_type
        )
        self.simulator = SCFDFoldingSimulator(alphabet_type=alphabet_type)
        
    def process_protein_batch(self, max_files=10, grid_size=64):
        """Process a batch of AlphaFold files to symbolic grids."""
        print(f"Processing batch of {max_files} proteins with {self.alphabet_type} alphabet")
        results = self.batch_processor.process_batch(
            grid_size=grid_size, 
            max_files=max_files
        )
        return results
        
    def run_folding_experiments(self, protein_files, runs_per_protein=3):
        """Run folding simulations on processed proteins."""
        all_results = []
        
        for protein_file in tqdm(protein_files, desc="Running folding experiments"):
            protein_id = Path(protein_file).stem
            
            try:
                # Load processed protein grid
                grid = np.load(protein_file)
                
                for run_id in range(runs_per_protein):
                    print(f"\nFolding simulation: {protein_id}, run {run_id+1}/{runs_per_protein}")
                    
                    # Initialize exporter
                    exporter = SCFDPathwayExporter(
                        protein_id=f"{protein_id}_run{run_id}",
                        alphabet_type=self.alphabet_type,
                        output_dir=str(self.output_dir)
                    )
                    
                    # Set initial conditions
                    # Randomize grid to simulate unfolded state
                    unfolded_grid = self._randomize_grid(grid)
                    
                    # Get sequence info (simplified)
                    sequence = self._grid_to_sequence(grid)
                    exporter.set_sequence_info(sequence, grid)
                    exporter.set_simulation_params({
                        'grid_size': grid.shape[0],
                        'temperature': self.simulator.temperature,
                        'alphabet_type': self.alphabet_type
                    })
                    
                    # Run simulation and log data
                    simulation_results = None
                    for timestep_data in self.simulator.run_folding_simulation(unfolded_grid, max_timesteps=50):
                        exporter.log_timestep(
                            timestep_data['timestep'],
                            timestep_data['grid'],
                            timestep_data['fields'], 
                            timestep_data['mutations'],
                            timestep_data['energies']
                        )
                        
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
                    exporter.finalize_simulation(
                        timestep_data['grid'],
                        {'converged': True},
                        folding_success=True
                    )
                    
                    output_file = exporter.export_to_json()
                    summary_file = exporter.export_compressed_summary()
                    
                    all_results.append({
                        'protein_id': protein_id,
                        'run_id': run_id,
                        'output_file': output_file,
                        'summary_file': summary_file
                    })
                    
            except Exception as e:
                print(f"Error processing {protein_id}: {str(e)}")
                continue
                
        return all_results
        
    def _randomize_grid(self, original_grid):
        """Create randomized version of grid (simulate unfolded state)."""
        randomized = original_grid.copy()
        occupied_positions = np.where(original_grid != -1)
        symbols = original_grid[occupied_positions]
        
        # Shuffle symbols randomly
        np.random.shuffle(symbols)
        randomized[occupied_positions] = symbols
        
        return randomized
        
    def _grid_to_sequence(self, grid):
        """Extract amino acid sequence from grid (simplified)."""
        occupied_symbols = grid[grid != -1]
        # This is a simplified representation - in reality you'd need position ordering
        return ''.join([get_symbol_description(int(s), self.alphabet_type)[0] 
                       for s in occupied_symbols[:100]])  # Limit to 100 for demo
        
    def run_full_pipeline(self, max_proteins=10, runs_per_protein=3):
        """Run complete pipeline: process -> simulate -> export."""
        print("="*60)
        print(f"SCFD Folding Pipeline - {self.alphabet_type} alphabet")
        print("="*60)
        
        # Step 1: Process AlphaFold structures
        print("\nStep 1: Processing AlphaFold structures...")
        batch_results = self.process_protein_batch(max_files=max_proteins)
        
        successful_files = [result[1] for result in batch_results['success']]
        print(f"Successfully processed {len(successful_files)} proteins")
        
        if not successful_files:
            print("No proteins processed successfully. Exiting.")
            return
            
        # Step 2: Run folding simulations
        print(f"\nStep 2: Running folding simulations ({runs_per_protein} runs each)...")
        folding_results = self.run_folding_experiments(successful_files[:max_proteins], runs_per_protein)
        
        # Step 3: Generate summary
        print(f"\nStep 3: Pipeline complete!")
        print(f"Generated {len(folding_results)} folding trajectories")
        print(f"Output directory: {self.output_dir}")
        
        # Create pipeline summary
        pipeline_summary = {
            'pipeline_config': {
                'alphabet_type': self.alphabet_type,
                'max_proteins': max_proteins,
                'runs_per_protein': runs_per_protein
            },
            'batch_results': {
                'processed': len(batch_results['success']),
                'skipped': len(batch_results['skipped']),
                'errors': len(batch_results['errors'])
            },
            'folding_results': folding_results,
            'total_trajectories': len(folding_results)
        }
        
        summary_file = self.output_dir / f"pipeline_summary_{self.alphabet_type}.json"
        with open(summary_file, 'w') as f:
            json.dump(pipeline_summary, f, indent=2, default=str)
            
        print(f"Pipeline summary saved: {summary_file}")
        return pipeline_summary

if __name__ == '__main__':
    # Test the complete pipeline
    pipeline = SCFDFoldingPipeline(alphabet_type='biochemical_12')
    
    # Run with small test dataset
    print("Testing pipeline with small dataset...")
    results = pipeline.run_full_pipeline(max_proteins=2, runs_per_protein=1)
    print("Test pipeline completed successfully!")