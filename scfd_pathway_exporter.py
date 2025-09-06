"""
SCFD Folding Pathway Data Exporter
==================================

Comprehensive data logging for SCFD protein folding simulations.
Captures ALL simulation data for AI training on folding processes.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
import hashlib
from enhanced_alphabets import get_alphabet_config, get_symbol_description

class SCFDPathwayExporter:
    """Exports comprehensive SCFD folding simulation data for AI training."""
    
    def __init__(self, protein_id, alphabet_type='biochemical_12', output_dir='training_data'):
        self.protein_id = protein_id
        self.alphabet_type = alphabet_type
        self.alphabet_config = get_alphabet_config(alphabet_type)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize comprehensive trajectory data
        self.trajectory_data = {
            "protein_id": protein_id,
            "export_timestamp": datetime.now().isoformat(),
            "alphabet_config": {
                "type": alphabet_type,
                "size": self.alphabet_config['size'],
                "max_entropy": self.alphabet_config['max_entropy'],
                "description": self.alphabet_config['description']
            },
            "sequence_info": {},
            "simulation_params": {},
            "folding_trajectory": [],
            "physics_logs": {
                "energy_evolution": [],
                "entropy_evolution": [], 
                "coherence_stats": [],
                "mutation_statistics": []
            },
            "structural_analysis": {
                "secondary_structure_formation": [],
                "hydrophobic_clustering": [],
                "charge_interactions": []
            },
            "final_metrics": {}
        }
        
        self.current_timestep = 0
        
    def set_sequence_info(self, sequence, structure_data):
        """Store original protein sequence and structure information."""
        self.trajectory_data["sequence_info"] = {
            "amino_acid_sequence": sequence,
            "sequence_length": len(sequence),
            "initial_structure_hash": self._hash_array(structure_data),
            "composition": self._analyze_composition(sequence)
        }
        
    def set_simulation_params(self, params):
        """Store SCFD simulation parameters."""
        self.trajectory_data["simulation_params"] = {
            "grid_size": params.get("grid_size", 64),
            "voxel_size": params.get("voxel_size", 1.5),
            "temperature": params.get("temperature", 1.0),
            "energy_weights": params.get("energy_weights", {}),
            "entropy_threshold": params.get("entropy_threshold", 0.1),
            "max_timesteps": params.get("max_timesteps", 1000),
            "convergence_criteria": params.get("convergence_criteria", {})
        }
        
    def log_timestep(self, timestep, grid_state, fields, mutations, energies, physics_data=None):
        """Log comprehensive data for a single simulation timestep."""
        
        # Core timestep data
        timestep_data = {
            "timestep": timestep,
            "grid_state": self._compress_grid(grid_state),
            "field_data": {
                "coherence_field": self._compress_field(fields.get("coherence")),
                "curvature_field": self._compress_field(fields.get("curvature")), 
                "entropy_field": self._compress_field(fields.get("entropy"))
            },
            "energy_data": {
                "total_energy": energies.get("total", 0.0),
                "potential_energy": energies.get("potential", 0.0),
                "kinetic_energy": energies.get("kinetic", 0.0),
                "entropy_contribution": energies.get("entropy", 0.0),
                "energy_breakdown": energies.get("breakdown", {})
            },
            "mutations_this_step": self._format_mutations(mutations),
            "field_statistics": self._calculate_field_stats(fields),
            "structural_metrics": self._analyze_structure_timestep(grid_state)
        }
        
        # Optional physics data
        if physics_data:
            timestep_data["physics_details"] = physics_data
            
        self.trajectory_data["folding_trajectory"].append(timestep_data)
        self.current_timestep = timestep
        
        # Update running statistics
        self._update_running_stats(timestep_data)
        
    def log_mutation(self, position, from_symbol, to_symbol, energy_delta, reason, local_context):
        """Log detailed mutation information."""
        mutation_data = {
            "timestep": self.current_timestep,
            "position": list(position),
            "from_symbol": int(from_symbol),
            "to_symbol": int(to_symbol),
            "from_description": get_symbol_description(from_symbol, self.alphabet_type),
            "to_description": get_symbol_description(to_symbol, self.alphabet_type),
            "energy_delta": float(energy_delta),
            "reason": reason,
            "local_context": {
                "neighborhood_symbols": local_context.get("neighborhood", []),
                "local_coherence_before": local_context.get("coherence_before", 0.0),
                "local_coherence_after": local_context.get("coherence_after", 0.0),
                "local_entropy": local_context.get("entropy", 0.0)
            }
        }
        self.trajectory_data["physics_logs"]["mutation_statistics"].append(mutation_data)
        
    def finalize_simulation(self, final_grid, convergence_info, folding_success=True):
        """Finalize simulation and compute final metrics."""
        self.trajectory_data["final_metrics"] = {
            "folding_success": folding_success,
            "total_timesteps": self.current_timestep,
            "convergence_info": convergence_info,
            "final_structure_hash": self._hash_array(final_grid),
            "final_energy": self.trajectory_data["folding_trajectory"][-1]["energy_data"]["total_energy"] if self.trajectory_data["folding_trajectory"] else 0.0,
            "folding_efficiency": self._calculate_folding_efficiency(),
            "structural_quality_metrics": self._assess_final_structure(final_grid)
        }
        
    def export_to_json(self, filename=None):
        """Export complete trajectory data to JSON file."""
        if filename is None:
            filename = f"{self.protein_id}_folding_trajectory_{self.alphabet_type}.json"
            
        output_path = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        json_data = self._prepare_for_json(self.trajectory_data)
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=self._json_serialize)
            
        print(f"Exported folding trajectory to: {output_path}")
        return output_path
        
    def export_compressed_summary(self, filename=None):
        """Export compressed summary for large-scale training."""
        if filename is None:
            filename = f"{self.protein_id}_summary_{self.alphabet_type}.json"
            
        summary_data = {
            "protein_id": self.trajectory_data["protein_id"],
            "alphabet_type": self.alphabet_type,
            "sequence_length": self.trajectory_data["sequence_info"]["sequence_length"],
            "total_timesteps": self.trajectory_data["final_metrics"]["total_timesteps"],
            "folding_success": self.trajectory_data["final_metrics"]["folding_success"],
            "key_transitions": self._extract_key_transitions(),
            "energy_trajectory": [step["energy_data"]["total_energy"] 
                                for step in self.trajectory_data["folding_trajectory"]],
            "mutation_count_per_timestep": self._count_mutations_per_timestep()
        }
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=self._json_serialize)
            
        return output_path
        
    # Helper methods
    def _hash_array(self, arr):
        """Create hash of numpy array for tracking changes."""
        return hashlib.md5(arr.tobytes()).hexdigest()[:16]
        
    def _analyze_composition(self, sequence):
        """Analyze amino acid composition of sequence."""
        composition = {}
        for aa in sequence:
            symbol = self.alphabet_config['map'].get(aa, -1)
            description = get_symbol_description(symbol, self.alphabet_type)
            composition[description] = composition.get(description, 0) + 1
        return composition
        
    def _compress_grid(self, grid):
        """Compress grid data for storage (only non-empty positions)."""
        non_empty = np.where(grid != -1)
        return {
            "positions": [list(pos) for pos in zip(*non_empty)],
            "symbols": grid[non_empty].tolist(),
            "shape": list(grid.shape)
        }
        
    def _compress_field(self, field):
        """Compress field data (statistical summary + key regions)."""
        if field is None:
            return None
        return {
            "mean": float(np.mean(field)),
            "std": float(np.std(field)),
            "min": float(np.min(field)),
            "max": float(np.max(field)),
            "high_value_positions": np.where(field > np.percentile(field, 90))
        }
        
    def _format_mutations(self, mutations):
        """Format mutation data for storage."""
        if not mutations:
            return []
        formatted = []
        for mut in mutations:
            # Handle different mutation data formats
            if isinstance(mut, dict):
                formatted.append({
                    "position": mut.get("position", mut.get("pos", [0,0,0])),
                    "from": mut.get("from_symbol", mut.get("from", -1)),
                    "to": mut.get("to_symbol", mut.get("to", -1)),
                    "energy_delta": mut.get("energy_delta", mut.get("delta", 0.0))
                })
            else:
                # Handle other formats if needed
                formatted.append({"position": [0,0,0], "from": -1, "to": -1, "energy_delta": 0.0})
        return formatted
        
    def _calculate_field_stats(self, fields):
        """Calculate comprehensive field statistics."""
        stats = {}
        for field_name, field_data in fields.items():
            if field_data is not None:
                stats[field_name] = {
                    "mean": float(np.mean(field_data)),
                    "variance": float(np.var(field_data)), 
                    "entropy": self._calculate_field_entropy(field_data)
                }
        return stats
        
    def _calculate_field_entropy(self, field):
        """Calculate entropy of field values."""
        hist, _ = np.histogram(field.flatten(), bins=10)
        hist = hist[hist > 0]  # Remove zero bins
        prob = hist / np.sum(hist)
        return -np.sum(prob * np.log2(prob)) if len(prob) > 1 else 0.0
        
    def _analyze_structure_timestep(self, grid_state):
        """Analyze structural features at current timestep."""
        non_empty = grid_state[grid_state != -1]
        if len(non_empty) == 0:
            return {"empty_structure": True}
            
        return {
            "occupied_voxels": int(np.sum(grid_state != -1)),
            "symbol_counts": {int(k): int(v) for k, v in zip(*np.unique(non_empty, return_counts=True))},
            "center_of_mass": list(np.mean(np.where(grid_state != -1), axis=1)),
            "radius_of_gyration": self._calculate_radius_of_gyration(grid_state)
        }
        
    def _calculate_radius_of_gyration(self, grid_state):
        """Calculate radius of gyration as structural compactness measure."""
        positions = np.where(grid_state != -1)
        if len(positions[0]) == 0:
            return 0.0
        positions = np.array(positions).T
        center = np.mean(positions, axis=0)
        distances = np.sum((positions - center)**2, axis=1)
        return float(np.sqrt(np.mean(distances)))
        
    def _update_running_stats(self, timestep_data):
        """Update running statistics during simulation."""
        self.trajectory_data["physics_logs"]["energy_evolution"].append({
            "timestep": timestep_data["timestep"],
            "total_energy": timestep_data["energy_data"]["total_energy"]
        })
        
    def _calculate_folding_efficiency(self):
        """Calculate how efficiently the protein folded."""
        if not self.trajectory_data["folding_trajectory"]:
            return 0.0
        total_mutations = sum(len(step["mutations_this_step"]) 
                            for step in self.trajectory_data["folding_trajectory"])
        return total_mutations / max(self.current_timestep, 1)
        
    def _assess_final_structure(self, final_grid):
        """Assess quality of final folded structure."""
        return {
            "compactness": self._calculate_radius_of_gyration(final_grid),
            "occupancy_fraction": float(np.sum(final_grid != -1) / final_grid.size),
            "symbol_diversity": len(np.unique(final_grid[final_grid != -1]))
        }
        
    def _extract_key_transitions(self):
        """Extract key folding transition events."""
        transitions = []
        energy_data = [step["energy_data"]["total_energy"] 
                      for step in self.trajectory_data["folding_trajectory"]]
        
        # Find major energy drops (folding events)
        for i in range(1, len(energy_data)):
            energy_drop = energy_data[i-1] - energy_data[i]
            if energy_drop > np.std(energy_data):  # Significant drop
                transitions.append({
                    "timestep": i,
                    "energy_drop": energy_drop,
                    "type": "major_folding_event"
                })
        return transitions
        
    def _count_mutations_per_timestep(self):
        """Count mutations per timestep for analysis."""
        return [len(step["mutations_this_step"]) 
                for step in self.trajectory_data["folding_trajectory"]]
        
    def _prepare_for_json(self, data):
        """Prepare data structure for JSON serialization."""
        if isinstance(data, dict):
            return {key: self._prepare_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        else:
            return data
            
    def _json_serialize(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return str(obj)

if __name__ == '__main__':
    # Test the exporter
    exporter = SCFDPathwayExporter("test_protein", "biochemical_12")
    
    # Simulate some data
    exporter.set_sequence_info("ACDEFGHIKLMNPQRSTVWY", np.random.randint(0, 14, (64, 64, 64)))
    exporter.set_simulation_params({"grid_size": 64, "temperature": 1.0})
    
    # Test timestep logging
    test_grid = np.random.randint(-1, 14, (64, 64, 64))
    test_fields = {
        "coherence": np.random.rand(64, 64, 64),
        "curvature": np.random.rand(64, 64, 64),
        "entropy": np.random.rand(64, 64, 64)
    }
    test_energies = {"total": -100.5, "potential": -120.0, "kinetic": 19.5}
    test_mutations = [{"pos": [10, 20, 30], "from": 0, "to": 5, "delta": -2.1}]
    
    exporter.log_timestep(1, test_grid, test_fields, test_mutations, test_energies)
    
    # Export test data
    exporter.finalize_simulation(test_grid, {"converged": True}, True)
    output_file = exporter.export_to_json()
    print(f"Test export successful: {output_file}")