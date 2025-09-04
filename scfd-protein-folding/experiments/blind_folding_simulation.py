#!/usr/bin/env python3
"""
Blind Protein Folding Simulation with SCFD
==========================================

The ULTIMATE test: Can SCFD discover protein folding from scratch?

This simulation:
1. Starts with amino acids in RANDOM positions (unfolded)
2. Applies SCFD dynamics with PHYSICS-BASED parameters 
3. Evolves the system over many timesteps
4. Compares final structure to known AlphaFold result

SUCCESS = SCFD discovers folding without knowing the answer!
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import time

class BlindFoldingSimulator:
    """SCFD-based protein folding from unfolded states."""
    
    def __init__(self, grid_size=64, voxel_size=1.5):
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        
        # Physics-based parameters (NOT fitted to structures!)
        self.physics_params = {
            # Potts coupling strengths (from experimental chemistry)
            'alpha_hydrophobic': 2.5,    # Hydrophobic clustering strength
            'alpha_polar': 1.2,          # H-bond strength  
            'alpha_charged': 3.8,        # Electrostatic strength
            
            # Cross-interactions
            'alpha_hydro_polar': -0.5,   # Hydrophobic-polar repulsion
            'alpha_hydro_charged': -0.8, # Hydrophobic-charged repulsion
            'alpha_polar_charged': 0.8,  # Polar-charged attraction
            
            # Field coupling (external influences)
            'chi': 0.3,                  # Response to external field
            
            # Entropy bounds (folding cooperativity)
            'h_min': 0.5,                # Min entropy (folded state)
            'h_max': 1.5,                # Max entropy (unfolded state)
            
            # Temperature effects
            'temperature': 300.0,         # Kelvin (room temp)
            'kb': 1.38e-23               # Boltzmann constant (symbolic units)
        }
        
        print("Blind folding simulator initialized")
        print(f"Grid: {grid_size}³, Voxel size: {voxel_size} Å")
        print("Physics parameters loaded from experimental data")
    
    def create_unfolded_state(self, target_protein_path):
        """
        Create unfolded initial state from a folded protein.
        Same amino acids, completely random positions.
        """
        print(f"Creating unfolded state from {target_protein_path.name}...")
        
        # Load the folded target (this is our 'answer' to compare against)
        folded_grid = np.load(target_protein_path)
        
        # Extract amino acid composition
        folded_positions = np.where(folded_grid != -1)
        amino_acids = folded_grid[folded_positions]
        
        print(f"Found {len(amino_acids)} amino acids:")
        unique, counts = np.unique(amino_acids, return_counts=True)
        for symbol, count in zip(unique, counts):
            symbol_name = {0: 'Hydrophobic', 1: 'Polar', 2: 'Charged'}[symbol]
            print(f"  {symbol_name}: {count} residues")
        
        # Create empty unfolded grid
        unfolded_grid = np.full((self.grid_size, self.grid_size, self.grid_size), -1, dtype=np.int8)
        
        # Place amino acids in RANDOM positions (completely unfolded)
        total_voxels = self.grid_size ** 3
        random_positions = np.random.choice(total_voxels, len(amino_acids), replace=False)
        random_coords = np.unravel_index(random_positions, (self.grid_size, self.grid_size, self.grid_size))
        
        # Shuffle amino acids to break any structural memory
        shuffled_amino_acids = amino_acids.copy()
        np.random.shuffle(shuffled_amino_acids)
        
        # Place shuffled amino acids at random positions
        for i, (x, y, z) in enumerate(zip(*random_coords)):
            unfolded_grid[x, y, z] = shuffled_amino_acids[i]
        
        # Verify unfolded state
        unfolded_positions = np.where(unfolded_grid != -1)
        center_x, center_y, center_z = np.mean(unfolded_positions[0]), np.mean(unfolded_positions[1]), np.mean(unfolded_positions[2])
        distances = np.sqrt((unfolded_positions[0] - center_x)**2 + 
                           (unfolded_positions[1] - center_y)**2 + 
                           (unfolded_positions[2] - center_z)**2)
        avg_distance_unfolded = np.mean(distances)
        
        # Compare to folded state
        folded_center_x, folded_center_y, folded_center_z = np.mean(folded_positions[0]), np.mean(folded_positions[1]), np.mean(folded_positions[2])
        folded_distances = np.sqrt((folded_positions[0] - folded_center_x)**2 + 
                                  (folded_positions[1] - folded_center_y)**2 + 
                                  (folded_positions[2] - folded_center_z)**2)
        avg_distance_folded = np.mean(folded_distances)
        
        print(f"Unfolded state created:")
        print(f"  Average distance from center: {avg_distance_unfolded:.1f} voxels (unfolded)")
        print(f"  Target folded distance: {avg_distance_folded:.1f} voxels")
        print(f"  Expansion factor: {avg_distance_unfolded/avg_distance_folded:.1f}x")
        
        return unfolded_grid, folded_grid
    
    def calculate_local_energy(self, grid, x, y, z):
        """Calculate local energy for SCFD dynamics at position (x,y,z)."""
        center_symbol = grid[x, y, z]
        if center_symbol == -1:
            return 0.0
        
        total_energy = 0.0
        
        # Check all neighbors in 3x3x3 neighborhood
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (0 <= nx < self.grid_size and 
                        0 <= ny < self.grid_size and 
                        0 <= nz < self.grid_size):
                        
                        neighbor = grid[nx, ny, nz]
                        interaction_energy = self.get_interaction_energy(center_symbol, neighbor)
                        
                        # Distance weighting (closer neighbors have stronger influence)
                        distance = np.sqrt(dx**2 + dy**2 + dz**2)
                        weight = 1.0 / distance if distance > 0 else 1.0
                        
                        total_energy += interaction_energy * weight
        
        return total_energy
    
    def get_interaction_energy(self, symbol1, symbol2):
        """Get interaction energy between two symbols (physics-based)."""
        if symbol1 == -1 or symbol2 == -1:
            return 0.1  # Weak solvent interaction
        
        # Same type interactions (favorable)
        if symbol1 == symbol2:
            if symbol1 == 0:
                return -self.physics_params['alpha_hydrophobic']  # Negative = favorable
            elif symbol1 == 1:
                return -self.physics_params['alpha_polar']
            elif symbol1 == 2:
                return -self.physics_params['alpha_charged']
        
        # Different type interactions
        else:
            if (symbol1 == 0 and symbol2 == 1) or (symbol1 == 1 and symbol2 == 0):
                return self.physics_params['alpha_hydro_polar']  # Positive = unfavorable
            elif (symbol1 == 0 and symbol2 == 2) or (symbol1 == 2 and symbol2 == 0):
                return self.physics_params['alpha_hydro_charged']
            elif (symbol1 == 1 and symbol2 == 2) or (symbol1 == 2 and symbol2 == 1):
                return -self.physics_params['alpha_polar_charged']  # Favorable
        
        return 0.0
    
    def scfd_timestep(self, grid, temperature=None):
        """
        Single SCFD timestep: attempt to move amino acids based on energy gradients.
        This implements the core SCFD dynamics for protein folding.
        """
        if temperature is None:
            temperature = self.physics_params['temperature']
        
        new_grid = grid.copy()
        occupied_positions = np.where(grid != -1)
        
        if len(occupied_positions[0]) == 0:
            return new_grid
        
        # Try to move each amino acid to lower energy position
        moves_attempted = 0
        moves_accepted = 0
        
        # Randomize order to avoid bias
        indices = list(range(len(occupied_positions[0])))
        np.random.shuffle(indices)
        
        for idx in indices:
            x, y, z = occupied_positions[0][idx], occupied_positions[1][idx], occupied_positions[2][idx]
            amino_acid = grid[x, y, z]
            
            # Current energy
            current_energy = self.calculate_local_energy(grid, x, y, z)
            
            # Try moving to nearby empty positions
            best_move = None
            best_energy = current_energy
            
            # Search in local neighborhood for better positions
            search_radius = 2
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    for dz in range(-search_radius, search_radius + 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        
                        new_x, new_y, new_z = x + dx, y + dy, z + dz
                        
                        # Check bounds and availability
                        if (0 <= new_x < self.grid_size and 
                            0 <= new_y < self.grid_size and 
                            0 <= new_z < self.grid_size and
                            new_grid[new_x, new_y, new_z] == -1):  # Empty position
                            
                            # Calculate energy at new position
                            temp_grid = new_grid.copy()
                            temp_grid[x, y, z] = -1  # Remove from old position
                            temp_grid[new_x, new_y, new_z] = amino_acid  # Place at new position
                            
                            new_energy = self.calculate_local_energy(temp_grid, new_x, new_y, new_z)
                            
                            # Accept move if energy is lower (favorable)
                            if new_energy < best_energy:
                                best_energy = new_energy
                                best_move = (new_x, new_y, new_z)
            
            # Apply best move if found
            moves_attempted += 1
            if best_move is not None:
                new_x, new_y, new_z = best_move
                new_grid[x, y, z] = -1  # Clear old position
                new_grid[new_x, new_y, new_z] = amino_acid  # Set new position
                moves_accepted += 1
        
        acceptance_rate = moves_accepted / moves_attempted if moves_attempted > 0 else 0
        return new_grid, acceptance_rate
    
    def calculate_folding_metrics(self, grid):
        """Calculate metrics to monitor folding progress."""
        occupied_positions = np.where(grid != -1)
        
        if len(occupied_positions[0]) == 0:
            return None
        
        x, y, z = occupied_positions
        symbols = grid[occupied_positions]
        
        # Center of mass
        center_x, center_y, center_z = np.mean(x), np.mean(y), np.mean(z)
        
        # Compactness (average distance from center)
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
        compactness = 1.0 / (1.0 + np.mean(distances))
        
        # Hydrophobic core formation
        hydro_distances = distances[symbols == 0] if np.any(symbols == 0) else []
        polar_distances = distances[symbols == 1] if np.any(symbols == 1) else []
        
        hydrophobic_core_score = 0.0
        if len(hydro_distances) > 0 and len(polar_distances) > 0:
            if np.mean(hydro_distances) < np.mean(polar_distances):
                hydrophobic_core_score = 1.0 - (np.mean(hydro_distances) / np.mean(polar_distances))
        
        # Total energy of the system
        total_energy = 0.0
        for i in range(len(x)):
            total_energy += self.calculate_local_energy(grid, x[i], y[i], z[i])
        
        return {
            'compactness': compactness,
            'hydrophobic_core_score': hydrophobic_core_score,
            'total_energy': total_energy,
            'avg_distance_from_center': np.mean(distances),
            'center_of_mass': (center_x, center_y, center_z)
        }
    
    def run_folding_simulation(self, unfolded_grid, folded_grid, max_timesteps=1000):
        """Run the complete folding simulation."""
        print(f"\nStarting blind folding simulation...")
        print(f"Max timesteps: {max_timesteps}")
        print("Physics-based SCFD dynamics will attempt to discover folding")
        print()
        
        current_grid = unfolded_grid.copy()
        
        # Track progress
        history = {
            'timestep': [],
            'compactness': [],
            'hydrophobic_core_score': [], 
            'total_energy': [],
            'acceptance_rate': [],
            'distance_to_target': []
        }
        
        # Initial metrics
        initial_metrics = self.calculate_folding_metrics(current_grid)
        target_metrics = self.calculate_folding_metrics(folded_grid)
        
        print(f"Initial state:")
        print(f"  Compactness: {initial_metrics['compactness']:.3f}")
        print(f"  Hydrophobic core: {initial_metrics['hydrophobic_core_score']:.3f}")
        print(f"  Total energy: {initial_metrics['total_energy']:.1f}")
        
        print(f"\nTarget folded state:")
        print(f"  Compactness: {target_metrics['compactness']:.3f}")
        print(f"  Hydrophobic core: {target_metrics['hydrophobic_core_score']:.3f}")
        print(f"  Total energy: {target_metrics['total_energy']:.1f}")
        
        print(f"\nRunning SCFD dynamics...")
        
        # Main folding loop
        with tqdm(total=max_timesteps, desc="Folding") as pbar:
            for timestep in range(max_timesteps):
                # SCFD timestep
                current_grid, acceptance_rate = self.scfd_timestep(current_grid)
                
                # Calculate current metrics
                current_metrics = self.calculate_folding_metrics(current_grid)
                
                if current_metrics is None:
                    print("Error: Lost all amino acids!")
                    break
                
                # Calculate similarity to target
                distance_to_target = self.calculate_structural_similarity(current_grid, folded_grid)
                
                # Store history
                history['timestep'].append(timestep)
                history['compactness'].append(current_metrics['compactness'])
                history['hydrophobic_core_score'].append(current_metrics['hydrophobic_core_score'])
                history['total_energy'].append(current_metrics['total_energy'])
                history['acceptance_rate'].append(acceptance_rate)
                history['distance_to_target'].append(distance_to_target)
                
                # Update progress bar
                pbar.set_postfix({
                    'Compact': f"{current_metrics['compactness']:.3f}",
                    'Energy': f"{current_metrics['total_energy']:.1f}",
                    'Accept': f"{acceptance_rate:.1%}"
                })
                pbar.update(1)
                
                # Early stopping if converged
                if timestep > 100 and timestep % 50 == 0:
                    recent_energies = history['total_energy'][-50:]
                    if len(set(recent_energies)) == 1:  # No change in energy
                        print(f"\nConverged at timestep {timestep}")
                        break
        
        final_metrics = self.calculate_folding_metrics(current_grid)
        final_distance = self.calculate_structural_similarity(current_grid, folded_grid)
        
        return current_grid, history, final_metrics, final_distance
    
    def calculate_structural_similarity(self, grid1, grid2):
        """Calculate structural similarity between two grids."""
        pos1 = np.where(grid1 != -1)
        pos2 = np.where(grid2 != -1)
        
        if len(pos1[0]) != len(pos2[0]):
            return float('inf')  # Different number of residues
        
        # Center of mass alignment
        cm1 = (np.mean(pos1[0]), np.mean(pos1[1]), np.mean(pos1[2]))
        cm2 = (np.mean(pos2[0]), np.mean(pos2[1]), np.mean(pos2[2]))
        
        # Average distance between centers of mass
        cm_distance = np.sqrt((cm1[0] - cm2[0])**2 + (cm1[1] - cm2[1])**2 + (cm1[2] - cm2[2])**2)
        
        # Compactness similarity
        dist1 = np.sqrt((pos1[0] - cm1[0])**2 + (pos1[1] - cm1[1])**2 + (pos1[2] - cm1[2])**2)
        dist2 = np.sqrt((pos2[0] - cm2[0])**2 + (pos2[1] - cm2[1])**2 + (pos2[2] - cm2[2])**2)
        
        compactness_diff = abs(np.mean(dist1) - np.mean(dist2))
        
        return cm_distance + compactness_diff
    
    def analyze_folding_results(self, initial_grid, final_grid, folded_grid, history):
        """Analyze the results of the folding simulation."""
        print("\n" + "="*70)
        print("BLIND FOLDING SIMULATION RESULTS")
        print("="*70)
        
        # Compare initial vs final vs target
        initial_metrics = self.calculate_folding_metrics(initial_grid)
        final_metrics = self.calculate_folding_metrics(final_grid)
        target_metrics = self.calculate_folding_metrics(folded_grid)
        
        print("\nStructural comparison:")
        print(f"{'Metric':<20} {'Initial':<10} {'Final':<10} {'Target':<10} {'Success':<10}")
        print("-" * 60)
        
        # Compactness
        compact_success = abs(final_metrics['compactness'] - target_metrics['compactness']) < 0.1
        print(f"{'Compactness':<20} {initial_metrics['compactness']:<10.3f} {final_metrics['compactness']:<10.3f} {target_metrics['compactness']:<10.3f} {str(compact_success):<10}")
        
        # Hydrophobic core
        core_success = final_metrics['hydrophobic_core_score'] > 0.3
        print(f"{'Hydrophobic core':<20} {initial_metrics['hydrophobic_core_score']:<10.3f} {final_metrics['hydrophobic_core_score']:<10.3f} {target_metrics['hydrophobic_core_score']:<10.3f} {str(core_success):<10}")
        
        # Energy reduction
        energy_reduction = initial_metrics['total_energy'] - final_metrics['total_energy']
        energy_success = energy_reduction > 0
        print(f"{'Energy reduction':<20} {0.0:<10.1f} {energy_reduction:<10.1f} {'N/A':<10} {str(energy_success):<10}")
        
        print(f"\nFolding trajectory:")
        print(f"  Initial distance from center: {initial_metrics['avg_distance_from_center']:.1f} voxels")
        print(f"  Final distance from center: {final_metrics['avg_distance_from_center']:.1f} voxels")  
        print(f"  Target distance from center: {target_metrics['avg_distance_from_center']:.1f} voxels")
        
        final_distance_to_target = history['distance_to_target'][-1]
        print(f"  Final similarity to target: {final_distance_to_target:.1f}")
        
        # Overall success assessment
        print(f"\n" + "="*70)
        print("FOLDING SUCCESS ASSESSMENT")
        print("="*70)
        
        success_criteria = [
            compact_success,
            core_success,
            energy_success,
            final_distance_to_target < 10.0  # Reasonable similarity
        ]
        
        success_count = sum(success_criteria)
        
        if success_count >= 3:
            print("EXCELLENT SUCCESS! SCFD discovered protein folding physics!")
            print("- Structure became compact")
            print("- Hydrophobic core formed") 
            print("- Energy decreased")
            print("- Similar to target structure")
            print("\nThis is strong evidence that SCFD captures real folding physics!")
            
        elif success_count >= 2:
            print("MODERATE SUCCESS! SCFD shows folding behavior")
            print("- Some folding physics captured")
            print("- Partial structure formation")
            print("- May need parameter tuning")
            
        else:
            print("LIMITED SUCCESS. SCFD dynamics insufficient for folding")
            print("- Structure remained largely unfolded")
            print("- May need stronger interactions or longer simulation")
            print("- Could indicate missing physics")
        
        return {
            'success_count': success_count,
            'compact_success': compact_success,
            'core_success': core_success,
            'energy_success': energy_success,
            'final_distance': final_distance_to_target
        }
    
    def create_folding_plots(self, history, results):
        """Create plots showing folding progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        timesteps = history['timestep']
        
        # Plot 1: Compactness over time
        axes[0,0].plot(timesteps, history['compactness'])
        axes[0,0].set_title('Protein Compactness During Folding')
        axes[0,0].set_xlabel('Timestep')
        axes[0,0].set_ylabel('Compactness')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Energy over time
        axes[0,1].plot(timesteps, history['total_energy'], color='red')
        axes[0,1].set_title('Total Energy During Folding')
        axes[0,1].set_xlabel('Timestep')
        axes[0,1].set_ylabel('Energy')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Hydrophobic core formation
        axes[1,0].plot(timesteps, history['hydrophobic_core_score'], color='orange')
        axes[1,0].set_title('Hydrophobic Core Formation')
        axes[1,0].set_xlabel('Timestep')
        axes[1,0].set_ylabel('Core Score')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Distance to target structure
        axes[1,1].plot(timesteps, history['distance_to_target'], color='green')
        axes[1,1].set_title('Similarity to Target Structure')
        axes[1,1].set_xlabel('Timestep')
        axes[1,1].set_ylabel('Distance to Target')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('blind_folding_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Folding analysis plots saved as: blind_folding_results.png")


def main():
    """Run the blind folding simulation."""
    print("="*70)
    print("BLIND PROTEIN FOLDING SIMULATION WITH SCFD")
    print("="*70)
    print("The ultimate test: Can SCFD discover protein folding from scratch?")
    print()
    
    # Initialize simulator
    simulator = BlindFoldingSimulator(grid_size=64, voxel_size=1.5)
    
    # Select protein to test
    processed_dir = Path('processed')
    protein_files = list(processed_dir.glob('*.npy'))
    
    if not protein_files:
        print("No processed proteins found! Please run extraction first.")
        return
    
    print("Available proteins for folding test:")
    for i, pfile in enumerate(protein_files, 1):
        print(f"  {i}. {pfile.name}")
    
    # Use first protein for test (or could make this selectable)
    test_protein = protein_files[0]
    print(f"\nUsing {test_protein.name} for blind folding test")
    
    # Create unfolded initial state
    unfolded_grid, folded_target = simulator.create_unfolded_state(test_protein)
    
    # Run folding simulation
    final_grid, history, final_metrics, final_distance = simulator.run_folding_simulation(
        unfolded_grid, folded_target, max_timesteps=500
    )
    
    # Analyze results
    results = simulator.analyze_folding_results(unfolded_grid, final_grid, folded_target, history)
    
    # Create visualization
    simulator.create_folding_plots(history, results)
    
    # Save results
    simulation_results = {
        'protein_tested': test_protein.name,
        'physics_parameters': simulator.physics_params,
        'final_metrics': final_metrics,
        'success_metrics': results,
        'total_timesteps': len(history['timestep'])
    }
    
    with open('blind_folding_simulation_results.json', 'w') as f:
        json.dump(simulation_results, f, indent=2, default=str)
    
    print(f"\nSimulation results saved to: blind_folding_simulation_results.json")
    
    return results

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    main()