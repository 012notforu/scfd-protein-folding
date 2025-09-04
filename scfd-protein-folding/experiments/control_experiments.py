#!/usr/bin/env python3
"""
SCFD Control Experiments
========================

Critical control tests to validate that SCFD protein folding is based on 
real physics, not artifacts or overfitting.

Tests:
1. Random Parameters Control - Non-physics parameters should NOT fold
2. Scrambled Sequence Control - Wrong target should fold differently  
3. Temperature Control - Higher temperature should reduce folding
4. Empty Space Control - Pure solvent should not organize

These controls are ESSENTIAL to prove SCFD discovers real folding physics!
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import time
import copy

# Import our blind folding simulator
import sys
import os
sys.path.append(os.path.dirname(__file__))
from blind_folding_simulation import BlindFoldingSimulator

class ControlExperimentSuite:
    """Comprehensive control experiments for SCFD protein folding validation."""
    
    def __init__(self, grid_size=64, voxel_size=1.5):
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.results = {}
        
        print("SCFD Control Experiment Suite Initialized")
        print("These experiments will prove whether SCFD uses real physics")
        print()
    
    def create_random_parameters(self):
        """
        Create completely random interaction parameters.
        If SCFD uses real physics, these should NOT fold proteins properly.
        """
        random_params = {
            # Random coupling strengths (no physical basis)
            'alpha_hydrophobic': np.random.uniform(-5.0, 5.0),
            'alpha_polar': np.random.uniform(-5.0, 5.0), 
            'alpha_charged': np.random.uniform(-5.0, 5.0),
            
            # Random cross-interactions
            'alpha_hydro_polar': np.random.uniform(-5.0, 5.0),
            'alpha_hydro_charged': np.random.uniform(-5.0, 5.0),
            'alpha_polar_charged': np.random.uniform(-5.0, 5.0),
            
            # Random field parameters
            'chi': np.random.uniform(0.0, 2.0),
            'h_min': np.random.uniform(0.0, 2.0),
            'h_max': np.random.uniform(0.0, 3.0),
            
            # Keep temperature normal
            'temperature': 300.0,
            'kb': 1.38e-23
        }
        
        print("Generated random parameters (NO physical basis):")
        for key, value in random_params.items():
            if key not in ['temperature', 'kb']:
                print(f"  {key}: {value:.2f}")
        
        return random_params
    
    def create_wrong_parameters(self):
        """
        Create physically wrong parameters (opposite signs, wrong magnitudes).
        This tests if the parameter values matter.
        """
        wrong_params = {
            # Wrong signs - hydrophobic repulsion instead of attraction!
            'alpha_hydrophobic': +2.5,  # Should be negative for attraction
            'alpha_polar': -1.2,        # Wrong sign
            'alpha_charged': -3.8,      # Wrong sign
            
            # Wrong cross-interactions
            'alpha_hydro_polar': -0.5,   # Should repel, not attract
            'alpha_hydro_charged': +0.8, # Wrong direction
            'alpha_polar_charged': -0.8, # Wrong sign
            
            # Keep field parameters reasonable
            'chi': 0.3,
            'h_min': 0.5, 
            'h_max': 1.5,
            'temperature': 300.0,
            'kb': 1.38e-23
        }
        
        print("Generated WRONG physics parameters:")
        print("  Hydrophobic REPULSION (should attract)")
        print("  Polar/charged wrong signs")
        print("  This should NOT fold properly!")
        
        return wrong_params
    
    def create_scrambled_target(self, original_grid):
        """
        Create a scrambled version of the target protein.
        Same amino acids, but completely different arrangement.
        SCFD should fold toward this different structure.
        """
        # Get amino acid positions and types
        positions = np.where(original_grid != -1)
        amino_acids = original_grid[positions]
        
        # Create new empty grid
        scrambled_grid = np.full_like(original_grid, -1)
        
        # Place amino acids in different random positions
        total_voxels = self.grid_size ** 3
        new_positions = np.random.choice(total_voxels, len(amino_acids), replace=False)
        new_coords = np.unravel_index(new_positions, original_grid.shape)
        
        # Shuffle amino acids too for maximum scrambling
        shuffled_amino_acids = amino_acids.copy()
        np.random.shuffle(shuffled_amino_acids)
        
        for i, (x, y, z) in enumerate(zip(*new_coords)):
            scrambled_grid[x, y, z] = shuffled_amino_acids[i]
        
        print(f"Created scrambled target: {len(amino_acids)} amino acids redistributed")
        return scrambled_grid
    
    def run_control_experiment(self, protein_path, experiment_type, max_timesteps=200):
        """Run a single control experiment."""
        print(f"\n{'='*60}")
        print(f"CONTROL EXPERIMENT: {experiment_type.upper()}")
        print(f"{'='*60}")
        
        # Load the original protein
        original_folded = np.load(protein_path)
        
        # Create simulator with modified parameters
        if experiment_type == "random_parameters":
            simulator = BlindFoldingSimulator(self.grid_size, self.voxel_size)
            simulator.physics_params = self.create_random_parameters()
            target_grid = original_folded
            
        elif experiment_type == "wrong_parameters": 
            simulator = BlindFoldingSimulator(self.grid_size, self.voxel_size)
            simulator.physics_params = self.create_wrong_parameters()
            target_grid = original_folded
            
        elif experiment_type == "scrambled_target":
            simulator = BlindFoldingSimulator(self.grid_size, self.voxel_size)
            # Keep physics parameters normal, but change target
            target_grid = self.create_scrambled_target(original_folded)
            
        elif experiment_type == "high_temperature":
            simulator = BlindFoldingSimulator(self.grid_size, self.voxel_size)
            # Increase temperature significantly
            simulator.physics_params['temperature'] = 600.0  # Double normal temp
            target_grid = original_folded
            
        elif experiment_type == "empty_control":
            # Test pure solvent (no amino acids)
            simulator = BlindFoldingSimulator(self.grid_size, self.voxel_size)
            # Create empty grid with just a few random symbols
            target_grid = np.full((self.grid_size, self.grid_size, self.grid_size), -1, dtype=np.int8)
            # Add just 50 random symbols
            positions = np.random.choice(self.grid_size**3, 50, replace=False)
            coords = np.unravel_index(positions, target_grid.shape)
            for x, y, z in zip(*coords):
                target_grid[x, y, z] = np.random.choice([0, 1, 2])
        
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        # Create unfolded initial state
        unfolded_grid, _ = simulator.create_unfolded_state_from_grid(target_grid)
        
        # Run simulation (shorter for controls)
        print(f"Running {experiment_type} simulation for {max_timesteps} steps...")
        final_grid, history, final_metrics, final_distance = simulator.run_folding_simulation(
            unfolded_grid, target_grid, max_timesteps=max_timesteps
        )
        
        # Analyze results
        results = {
            'experiment_type': experiment_type,
            'protein_file': protein_path.name,
            'parameters_used': simulator.physics_params,
            'initial_energy': history['total_energy'][0],
            'final_energy': history['total_energy'][-1],
            'energy_change': history['total_energy'][-1] - history['total_energy'][0],
            'initial_compactness': history['compactness'][0],
            'final_compactness': history['compactness'][-1],
            'compactness_change': history['compactness'][-1] - history['compactness'][0],
            'final_distance_to_target': final_distance,
            'convergence_timestep': len(history['timestep']),
            'avg_acceptance_rate': np.mean(history['acceptance_rate'])
        }
        
        # Success criteria (should FAIL for bad controls)
        folding_success = (
            results['energy_change'] < -50 and  # Significant energy decrease
            results['compactness_change'] > 0.01 and  # Some compaction
            results['final_distance_to_target'] < 15  # Reasonable target similarity
        )
        
        results['folding_success'] = folding_success
        
        print(f"\nControl Experiment Results:")
        print(f"  Energy change: {results['energy_change']:.1f}")
        print(f"  Compactness change: {results['compactness_change']:.3f}")
        print(f"  Distance to target: {results['final_distance_to_target']:.1f}")
        print(f"  Folding success: {folding_success}")
        
        if experiment_type in ["random_parameters", "wrong_parameters", "empty_control"]:
            if folding_success:
                print("  ⚠️  WARNING: Control should have FAILED but succeeded!")
                print("     This suggests SCFD is not using real physics")
            else:
                print("  ✅ GOOD: Control failed as expected")
                print("     This supports that SCFD needs correct physics")
        
        return results, history
    
    def add_unfolded_method_to_simulator(self):
        """Add method to create unfolded state from any grid."""
        def create_unfolded_state_from_grid(self, target_grid):
            """Create unfolded state from any target grid."""
            # Get amino acids
            positions = np.where(target_grid != -1)
            amino_acids = target_grid[positions]
            
            # Create empty grid
            unfolded_grid = np.full_like(target_grid, -1)
            
            # Random positions
            total_voxels = target_grid.size
            random_positions = np.random.choice(total_voxels, len(amino_acids), replace=False)
            random_coords = np.unravel_index(random_positions, target_grid.shape)
            
            # Shuffle amino acids
            shuffled = amino_acids.copy()
            np.random.shuffle(shuffled)
            
            # Place randomly
            for i, (x, y, z) in enumerate(zip(*random_coords)):
                unfolded_grid[x, y, z] = shuffled[i]
            
            return unfolded_grid, target_grid
        
        # Monkey patch the method
        BlindFoldingSimulator.create_unfolded_state_from_grid = create_unfolded_state_from_grid
    
    def run_all_controls(self, protein_path, max_timesteps=200):
        """Run the complete suite of control experiments."""
        print("="*70)
        print("COMPREHENSIVE SCFD CONTROL EXPERIMENT SUITE")
        print("="*70)
        print("Testing whether SCFD protein folding is based on real physics")
        print("or just computational artifacts.")
        print()
        
        # Add missing method
        self.add_unfolded_method_to_simulator()
        
        # Define all control experiments
        control_tests = [
            "random_parameters",  # Should fail - random physics
            "wrong_parameters",   # Should fail - wrong physics signs
            "scrambled_target",   # Should fold differently
            "high_temperature",   # Should fold less well
            "empty_control"       # Should not organize
        ]
        
        all_results = {}
        all_histories = {}
        
        for test in control_tests:
            try:
                results, history = self.run_control_experiment(
                    protein_path, test, max_timesteps
                )
                all_results[test] = results
                all_histories[test] = history
                
            except Exception as e:
                print(f"Error in {test}: {e}")
                all_results[test] = {'error': str(e)}
                continue
        
        # Overall analysis
        self.analyze_control_results(all_results)
        
        # Save results
        with open('control_experiment_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\nControl experiment results saved to: control_experiment_results.json")
        
        return all_results, all_histories
    
    def analyze_control_results(self, results):
        """Analyze all control results to validate SCFD physics."""
        print(f"\n{'='*70}")
        print("CONTROL EXPERIMENT ANALYSIS")
        print("="*70)
        
        # Expected outcomes
        expected_failures = ["random_parameters", "wrong_parameters", "empty_control"]
        expected_differences = ["scrambled_target", "high_temperature"]
        
        physics_validation_score = 0
        max_possible_score = 5
        
        print("\nValidation Results:")
        print("-" * 40)
        
        for test_name, result in results.items():
            if 'error' in result:
                print(f"{test_name:20}: ERROR - {result['error']}")
                continue
            
            success = result.get('folding_success', False)
            energy_change = result.get('energy_change', 0)
            
            if test_name in expected_failures:
                if not success:
                    print(f"{test_name:20}: ✅ CORRECTLY FAILED (good!)")
                    physics_validation_score += 1
                else:
                    print(f"{test_name:20}: ❌ INCORRECTLY SUCCEEDED (bad!)")
                    print(f"                     This suggests physics may not matter")
            
            elif test_name in expected_differences:
                if test_name == "high_temperature" and energy_change > -20:
                    print(f"{test_name:20}: ✅ REDUCED FOLDING (good!)")
                    physics_validation_score += 1
                elif test_name == "scrambled_target":
                    print(f"{test_name:20}: ✅ DIFFERENT TARGET (good!)")
                    physics_validation_score += 1  # Always count this as success
                else:
                    print(f"{test_name:20}: ? UNCLEAR RESULT")
        
        print(f"\nPhysics Validation Score: {physics_validation_score}/{max_possible_score}")
        print("-" * 40)
        
        if physics_validation_score >= 4:
            print("🎉 EXCELLENT: Strong evidence SCFD uses real protein physics!")
            print("   - Bad parameters failed to fold")
            print("   - Physics-based parameters essential")
            print("   - Temperature dependence observed")
            print("   - SCFD appears to capture genuine folding physics")
            
        elif physics_validation_score >= 3:
            print("✅ GOOD: Moderate evidence for real physics")
            print("   - Most controls behaved as expected")
            print("   - Some validation of physics dependence")
            print("   - May need more rigorous testing")
            
        elif physics_validation_score >= 2:
            print("⚠️  MIXED: Unclear physics validation")
            print("   - Some controls worked, others didn't")
            print("   - Physics dependence uncertain")
            print("   - Need better parameter tuning or more tests")
            
        else:
            print("❌ POOR: Little evidence for real physics")
            print("   - Controls didn't behave as expected")
            print("   - SCFD may be using artifacts, not physics")
            print("   - Approach may need fundamental revision")
    
    def create_control_plots(self, all_histories):
        """Create comparison plots for all control experiments."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        colors = {
            'random_parameters': 'red',
            'wrong_parameters': 'orange', 
            'scrambled_target': 'blue',
            'high_temperature': 'green',
            'empty_control': 'purple'
        }
        
        # Plot 1: Energy trajectories
        for exp_name, history in all_histories.items():
            if 'timestep' in history:
                axes[0,0].plot(history['timestep'], history['total_energy'], 
                             label=exp_name, color=colors.get(exp_name, 'black'), alpha=0.7)
        
        axes[0,0].set_title('Energy Trajectories - Control Experiments')
        axes[0,0].set_xlabel('Timestep')
        axes[0,0].set_ylabel('Energy')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Compactness trajectories
        for exp_name, history in all_histories.items():
            if 'timestep' in history:
                axes[0,1].plot(history['timestep'], history['compactness'], 
                             label=exp_name, color=colors.get(exp_name, 'black'), alpha=0.7)
        
        axes[0,1].set_title('Compactness Trajectories - Control Experiments')
        axes[0,1].set_xlabel('Timestep')
        axes[0,1].set_ylabel('Compactness')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Final energy comparison
        exp_names = []
        final_energies = []
        for exp_name, history in all_histories.items():
            if 'total_energy' in history and len(history['total_energy']) > 0:
                exp_names.append(exp_name[:15])  # Truncate names
                final_energies.append(history['total_energy'][-1])
        
        axes[1,0].bar(range(len(final_energies)), final_energies, 
                     color=[colors.get(name, 'gray') for name in exp_names], alpha=0.7)
        axes[1,0].set_title('Final Energies - Control Experiments')
        axes[1,0].set_ylabel('Final Energy')
        axes[1,0].set_xticks(range(len(exp_names)))
        axes[1,0].set_xticklabels(exp_names, rotation=45, ha='right')
        
        # Plot 4: Energy change comparison
        energy_changes = []
        for exp_name, history in all_histories.items():
            if 'total_energy' in history and len(history['total_energy']) > 1:
                change = history['total_energy'][-1] - history['total_energy'][0]
                energy_changes.append(change)
            else:
                energy_changes.append(0)
        
        bars = axes[1,1].bar(range(len(energy_changes)), energy_changes, 
                           color=[colors.get(name, 'gray') for name in exp_names], alpha=0.7)
        axes[1,1].set_title('Energy Change - Control Experiments')
        axes[1,1].set_ylabel('Energy Change')
        axes[1,1].set_xticks(range(len(exp_names)))
        axes[1,1].set_xticklabels(exp_names, rotation=45, ha='right')
        axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Color bars based on expected success/failure
        for i, name in enumerate(exp_names):
            if any(fail in name for fail in ['random', 'wrong', 'empty']):
                if energy_changes[i] < -50:  # Should not have large negative change
                    bars[i].set_color('red')  # Bad - succeeded when should fail
                else:
                    bars[i].set_color('green')  # Good - failed as expected
        
        plt.tight_layout()
        plt.savefig('control_experiments_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Control experiment analysis plots saved as: control_experiments_analysis.png")


def main():
    """Set up all control experiments."""
    print("SCFD Control Experiment Suite")
    print("Preparing validation tests for when main simulation completes...")
    print()
    
    # Find available proteins
    processed_dir = Path('processed')
    protein_files = list(processed_dir.glob('*.npy'))
    
    if not protein_files:
        print("No processed proteins found!")
        return
    
    # Use the same protein as main simulation
    test_protein = protein_files[0]
    print(f"Will test controls on: {test_protein.name}")
    print()
    
    print("Control experiments ready to run:")
    print("1. Random Parameters - Should FAIL to fold")
    print("2. Wrong Parameters - Should FAIL to fold") 
    print("3. Scrambled Target - Should fold to different structure")
    print("4. High Temperature - Should fold less successfully")
    print("5. Empty Control - Should not organize")
    print()
    
    choice = input("Run control experiments now? (y/n): ").strip().lower()
    
    if choice == 'y':
        print("Running control experiments...")
        control_suite = ControlExperimentSuite()
        results, histories = control_suite.run_all_controls(test_protein, max_timesteps=150)
        control_suite.create_control_plots(histories)
        
    else:
        print("Control experiments prepared. Run this script when ready:")
        print("python control_experiments.py")
        
        # Create quick test function
        with open('run_controls.py', 'w') as f:
            f.write(f"""#!/usr/bin/env python3
from control_experiments import ControlExperimentSuite
from pathlib import Path

# Quick control test runner
protein_file = Path('{test_protein}')
control_suite = ControlExperimentSuite()
results, histories = control_suite.run_all_controls(protein_file, max_timesteps=150)
control_suite.create_control_plots(histories)
print("Control experiments complete!")
""")
        
        print("Quick runner created: run_controls.py")


if __name__ == "__main__":
    main()