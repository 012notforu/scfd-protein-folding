"""
SCFD Folding AI Predictor
========================

Uses trained models to predict protein folding pathways and analyze results.
"""

import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from folding_ai_trainer import FoldingTransformer, FoldingTrajectoryDataset

class FoldingPredictor:
    """Predicts protein folding pathways using trained SCFD AI model."""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        
        # Load model
        self.load_model()
        print(f"Loaded folding predictor on {self.device}")
    
    def load_model(self):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model configuration
        self.input_dim = checkpoint['input_dim']
        model_config = checkpoint.get('model_config', {})
        
        # Create model with saved configuration
        self.model = FoldingTransformer(
            input_dim=self.input_dim,
            d_model=model_config.get('d_model', 128),
            nhead=model_config.get('nhead', 8),
            num_layers=model_config.get('num_layers', 4),
            max_seq_len=model_config.get('max_seq_len', 50)
        ).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded: {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def predict_folding_pathway(self, initial_state, max_steps=50):
        """Predict complete folding pathway from initial state."""
        self.model.eval()
        
        with torch.no_grad():
            # Convert initial state to tensor
            if isinstance(initial_state, dict):
                # Convert from trajectory format
                input_tensor = self._trajectory_to_tensor(initial_state)
            else:
                input_tensor = torch.FloatTensor(initial_state).unsqueeze(0)
            
            input_tensor = input_tensor.to(self.device)
            
            # Autoregressive prediction
            predicted_pathway = []
            current_input = input_tensor
            
            for step in range(max_steps):
                # Get predictions for current state
                outputs = self.model(current_input)
                
                # Extract predictions
                grid_pred = outputs['grid_prediction'][:, -1, :].cpu().numpy()[0]
                energy_pred = outputs['energy_prediction'][:, -1].cpu().numpy()[0]
                success_pred = outputs['success_prediction'].cpu().numpy()[0]
                
                predicted_pathway.append({
                    'timestep': step,
                    'predicted_grid': grid_pred,
                    'predicted_energy': energy_pred,
                    'folding_probability': success_pred,
                    'confidence': self._calculate_confidence(outputs)
                })
                
                # Use prediction as next input (teacher forcing disabled)
                next_features = self._grid_to_features(grid_pred, energy_pred)
                next_input = torch.FloatTensor(next_features).unsqueeze(0).unsqueeze(0).to(self.device)
                current_input = torch.cat([current_input, next_input], dim=1)
                
                # Stop if folding is predicted to be successful
                if success_pred > 0.8:
                    print(f"Folding predicted successful at step {step}")
                    break
            
            return predicted_pathway
    
    def _trajectory_to_tensor(self, trajectory_data):
        """Convert trajectory data to model input tensor."""
        # This would use the same processing as in FoldingTrajectoryDataset
        # Simplified version for demo
        grid_features = np.array(trajectory_data.get('grid_states', [[0]*20]))
        field_features = np.array(trajectory_data.get('field_states', [[0]*12])) 
        mutation_features = np.array(trajectory_data.get('mutations', [[0]*5]))
        
        # Combine features
        combined = np.concatenate([grid_features, field_features, mutation_features], axis=1)
        return torch.FloatTensor(combined)
    
    def _grid_to_features(self, grid_pred, energy_pred):
        """Convert predicted grid to feature vector for next timestep."""
        # Simplified conversion - in practice would need full grid reconstruction
        features = np.concatenate([
            grid_pred,  # Grid features (20 dims)
            [energy_pred, 0, 0, 0] * 3,  # Field features (12 dims) 
            [0] * 5  # Mutation features (5 dims)
        ])
        return features
    
    def _calculate_confidence(self, outputs):
        """Calculate prediction confidence from model outputs."""
        # Use variance of predictions as confidence measure
        grid_var = torch.var(outputs['grid_prediction']).item()
        energy_var = torch.var(outputs['energy_prediction']).item()
        
        # Lower variance = higher confidence
        confidence = 1.0 / (1.0 + grid_var + energy_var)
        return confidence
    
    def compare_with_ground_truth(self, prediction, ground_truth_file):
        """Compare predictions with actual SCFD simulation."""
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
        
        gt_energies = [step['energy_data']['total_energy'] 
                      for step in ground_truth['folding_trajectory']]
        pred_energies = [step['predicted_energy'] for step in prediction]
        
        # Calculate metrics
        min_length = min(len(gt_energies), len(pred_energies))
        gt_energies = gt_energies[:min_length]
        pred_energies = pred_energies[:min_length]
        
        mse = np.mean((np.array(gt_energies) - np.array(pred_energies))**2)
        correlation = np.corrcoef(gt_energies, pred_energies)[0,1]
        
        comparison = {
            'mse': mse,
            'correlation': correlation,
            'ground_truth_energies': gt_energies,
            'predicted_energies': pred_energies,
            'timesteps_compared': min_length
        }
        
        return comparison
    
    def analyze_folding_patterns(self, predictions_list):
        """Analyze common folding patterns across multiple predictions."""
        analysis = {
            'average_folding_time': [],
            'energy_trajectories': [],
            'success_rates': [],
            'confidence_scores': []
        }
        
        for pred in predictions_list:
            # Folding time (when success probability > 0.8)
            folding_time = len(pred)
            for i, step in enumerate(pred):
                if step['folding_probability'] > 0.8:
                    folding_time = i
                    break
            analysis['average_folding_time'].append(folding_time)
            
            # Energy trajectory
            energies = [step['predicted_energy'] for step in pred]
            analysis['energy_trajectories'].append(energies)
            
            # Final success rate
            final_success = pred[-1]['folding_probability'] if pred else 0
            analysis['success_rates'].append(final_success)
            
            # Average confidence
            avg_confidence = np.mean([step['confidence'] for step in pred]) if pred else 0
            analysis['confidence_scores'].append(avg_confidence)
        
        # Summary statistics
        summary = {
            'mean_folding_time': np.mean(analysis['average_folding_time']),
            'std_folding_time': np.std(analysis['average_folding_time']),
            'overall_success_rate': np.mean(analysis['success_rates']),
            'mean_confidence': np.mean(analysis['confidence_scores']),
            'energy_convergence': self._analyze_energy_convergence(analysis['energy_trajectories'])
        }
        
        return analysis, summary
    
    def _analyze_energy_convergence(self, energy_trajectories):
        """Analyze how consistently energies converge."""
        if not energy_trajectories:
            return {}
        
        # Align trajectories by length
        min_length = min(len(traj) for traj in energy_trajectories)
        aligned_energies = [traj[:min_length] for traj in energy_trajectories]
        
        # Calculate statistics
        mean_trajectory = np.mean(aligned_energies, axis=0)
        std_trajectory = np.std(aligned_energies, axis=0)
        
        return {
            'mean_energy_trajectory': mean_trajectory.tolist(),
            'std_energy_trajectory': std_trajectory.tolist(),
            'convergence_consistency': np.mean(std_trajectory[-5:]) if min_length > 5 else 0
        }
    
    def visualize_predictions(self, prediction, save_path=None):
        """Visualize predicted folding pathway."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        timesteps = [step['timestep'] for step in prediction]
        energies = [step['predicted_energy'] for step in prediction]
        success_probs = [step['folding_probability'] for step in prediction]
        confidences = [step['confidence'] for step in prediction]
        
        # Energy trajectory
        ax1.plot(timesteps, energies, 'b-o', markersize=4)
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Predicted Energy')
        ax1.set_title('Energy Evolution During Folding')
        ax1.grid(True, alpha=0.3)
        
        # Success probability
        ax2.plot(timesteps, success_probs, 'g-o', markersize=4)
        ax2.axhline(y=0.8, color='r', linestyle='--', label='Success Threshold')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Folding Success Probability')
        ax2.set_title('Predicted Folding Success Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Prediction confidence
        ax3.plot(timesteps, confidences, 'purple', marker='o', markersize=4)
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Prediction Confidence')
        ax3.set_title('Model Confidence During Prediction')
        ax3.grid(True, alpha=0.3)
        
        # Grid feature evolution (first few features)
        grid_features = [step['predicted_grid'][:5] for step in prediction]  # First 5 features
        grid_array = np.array(grid_features).T
        
        for i in range(5):
            ax4.plot(timesteps, grid_array[i], label=f'Feature {i}', marker='o', markersize=3)
        ax4.set_xlabel('Timestep')
        ax4.set_ylabel('Grid Feature Value')
        ax4.set_title('Structural Feature Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
        
        plt.show()


class FoldingAnalyzer:
    """Analyzes trained models and folding predictions."""
    
    def __init__(self, model_dir='models', data_dir='training_data'):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
    
    def evaluate_model_performance(self, model_path, test_data_path=None):
        """Comprehensive evaluation of trained model."""
        predictor = FoldingPredictor(model_path)
        
        results = {
            'model_info': {
                'model_path': str(model_path),
                'parameters': sum(p.numel() for p in predictor.model.parameters()),
                'device': str(predictor.device)
            },
            'predictions': [],
            'ground_truth_comparisons': []
        }
        
        # Test on available trajectory files
        trajectory_files = list(self.data_dir.glob("*_trajectory_*.json"))[:5]  # Test on 5 proteins
        
        for traj_file in trajectory_files:
            print(f"Testing on {traj_file.name}...")
            
            # Load ground truth
            with open(traj_file, 'r') as f:
                ground_truth = json.load(f)
            
            # Make prediction from initial state
            initial_state = ground_truth['folding_trajectory'][0] if ground_truth['folding_trajectory'] else {}
            prediction = predictor.predict_folding_pathway(initial_state, max_steps=20)
            
            # Compare with ground truth
            comparison = predictor.compare_with_ground_truth(prediction, traj_file)
            
            results['predictions'].append({
                'protein_id': ground_truth['protein_id'],
                'prediction': prediction,
                'folding_success_predicted': prediction[-1]['folding_probability'] > 0.8 if prediction else False,
                'folding_success_actual': ground_truth['final_metrics']['folding_success'],
                'prediction_length': len(prediction)
            })
            
            results['ground_truth_comparisons'].append(comparison)
        
        # Calculate overall metrics
        correlations = [comp['correlation'] for comp in results['ground_truth_comparisons'] if not np.isnan(comp['correlation'])]
        mse_values = [comp['mse'] for comp in results['ground_truth_comparisons']]
        
        results['overall_metrics'] = {
            'mean_correlation': np.mean(correlations) if correlations else 0,
            'mean_mse': np.mean(mse_values) if mse_values else float('inf'),
            'success_prediction_accuracy': self._calculate_success_accuracy(results['predictions'])
        }
        
        print(f"Model Evaluation Results:")
        print(f"Mean Energy Correlation: {results['overall_metrics']['mean_correlation']:.3f}")
        print(f"Mean Energy MSE: {results['overall_metrics']['mean_mse']:.3f}")
        print(f"Success Prediction Accuracy: {results['overall_metrics']['success_prediction_accuracy']:.3f}")
        
        return results
    
    def _calculate_success_accuracy(self, predictions):
        """Calculate accuracy of folding success predictions."""
        correct = 0
        total = len(predictions)
        
        for pred in predictions:
            if pred['folding_success_predicted'] == pred['folding_success_actual']:
                correct += 1
        
        return correct / total if total > 0 else 0
    
    def create_model_report(self, evaluation_results, save_path='model_evaluation_report.json'):
        """Create comprehensive model evaluation report."""
        report = {
            'evaluation_timestamp': str(np.datetime64('now')),
            'model_performance': evaluation_results['overall_metrics'],
            'detailed_results': evaluation_results,
            'recommendations': self._generate_recommendations(evaluation_results)
        }
        
        # Save report
        report_path = self.model_dir / save_path
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Model evaluation report saved: {report_path}")
        return report
    
    def _generate_recommendations(self, results):
        """Generate recommendations based on model performance."""
        recommendations = []
        
        correlation = results['overall_metrics']['mean_correlation']
        mse = results['overall_metrics']['mean_mse']
        success_accuracy = results['overall_metrics']['success_prediction_accuracy']
        
        if correlation < 0.7:
            recommendations.append("Energy prediction correlation is low. Consider increasing model capacity or training data.")
        
        if success_accuracy < 0.8:
            recommendations.append("Success prediction accuracy could be improved. Try class balancing or different loss weighting.")
        
        if mse > 100:
            recommendations.append("Energy MSE is high. Consider normalizing energy values or adjusting loss function.")
        
        if not recommendations:
            recommendations.append("Model performance looks good! Consider testing on larger datasets.")
        
        return recommendations


def run_prediction_demo():
    """Demonstration of the prediction system."""
    print("SCFD Folding AI Prediction Demo")
    print("=" * 50)
    
    # Check if trained model exists
    model_dir = Path('models')
    model_files = list(model_dir.glob('*.pt')) if model_dir.exists() else []
    
    if not model_files:
        print("No trained models found. Please run training first:")
        print("python folding_ai_trainer.py")
        return
    
    # Use the most recent model
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"Using model: {latest_model}")
    
    # Evaluate model
    analyzer = FoldingAnalyzer()
    results = analyzer.evaluate_model_performance(latest_model)
    
    # Create report
    report = analyzer.create_model_report(results)
    
    print("\nPrediction demo completed!")
    return results

if __name__ == '__main__':
    results = run_prediction_demo()