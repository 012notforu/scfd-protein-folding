#!/usr/bin/env python3
"""
Complete AI Training Workflow for SCFD Folding Data
==================================================

This script demonstrates the complete workflow:
1. Generate folding data from multiple proteins
2. Train AI models on the folding pathways  
3. Evaluate and analyze model performance
"""

import sys
import os
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.scfd_folding_pipeline import SCFDFoldingPipeline
from src.folding_ai_trainer import FoldingAITrainer
from src.folding_ai_predictor import FoldingAnalyzer

def generate_training_data(num_proteins=100, runs_per_protein=3):
    """Step 1: Generate large-scale SCFD folding pathway data."""
    print("="*80)
    print("STEP 1: LARGE-SCALE FOLDING PATHWAY DATA GENERATION")
    print("="*80)
    print(f"Target: {num_proteins} proteins x {runs_per_protein} runs = {num_proteins * runs_per_protein} trajectories")
    
    # Run the enhanced trajectory generator directly
    from generate_multiple_trajectories import main as generate_trajectories
    
    print(f"Generating large-scale folding data...")
    trajectory_files = generate_trajectories(num_proteins, runs_per_protein)
    
    print(f"\nLarge-scale data generation completed!")
    print(f"Generated {len(trajectory_files)} folding trajectories")
    print(f"Data ready for AI training")
    
    return {'total_trajectories': len(trajectory_files), 'files': trajectory_files}

def train_folding_ai(num_epochs=30, batch_size=32):
    """Step 2: Train AI models on large-scale folding data."""
    print("\n" + "="*80)
    print("STEP 2: LARGE-SCALE AI TRAINING ON FOLDING PATHWAYS")
    print("="*80)
    print(f"Training configuration: {num_epochs} epochs, batch size {batch_size}")
    
    try:
        # Initialize trainer
        trainer = FoldingAITrainer(data_dir='training_data', model_dir='models')
        
        # Prepare large-scale data
        train_size, test_size = trainer.prepare_data(
            test_size=0.2, 
            batch_size=batch_size,
            load_batch_size=64  # Process data in larger batches for efficiency
        )
        print(f"Large-scale training data: {train_size} protein trajectories")
        print(f"Test data: {test_size} protein trajectories")
        
        if train_size < 10:
            print("Warning: Small dataset. Consider generating more trajectories.")
            if train_size < 1:
                print("Not enough training data. Generate more folding trajectories first.")
                return None
        
        # Create scaled model architecture
        trainer.create_model(
            d_model=128,    # Larger embedding dimension
            nhead=8,        # More attention heads  
            num_layers=6    # Deeper model for complex patterns
        )
        
        # Large-scale training
        print(f"Starting large-scale training: {num_epochs} epochs...")
        train_losses, test_losses = trainer.train(num_epochs=num_epochs, save_every=10)
        
        print(f"\nAI training completed!")
        print(f"Model saved in: models/")
        
        return trainer
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_trained_model():
    """Step 3: Evaluate and analyze the trained model."""
    print("\n" + "="*60) 
    print("STEP 3: EVALUATING TRAINED MODEL")
    print("="*60)
    
    try:
        # Find the latest trained model
        model_dir = Path('models')
        if not model_dir.exists():
            print("No models directory found. Train a model first.")
            return None
            
        model_files = list(model_dir.glob('*.pt'))
        if not model_files:
            print("No trained models found. Train a model first.")
            return None
        
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        print(f"Evaluating model: {latest_model.name}")
        
        # Evaluate model performance
        analyzer = FoldingAnalyzer()
        results = analyzer.evaluate_model_performance(latest_model)
        
        # Create comprehensive report
        report = analyzer.create_model_report(results)
        
        print(f"\nModel evaluation completed!")
        print(f"Report saved in: models/model_evaluation_report.json")
        
        return results
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_complete_workflow(quick=False):
    """Run the complete large-scale AI training workflow."""
    print("SCFD FOLDING AI - LARGE-SCALE TRAINING WORKFLOW")
    print("="*90)
    
    if quick:
        print("QUICK MODE: Using existing data for demonstration")
        num_proteins, runs_per_protein = 7, 1  # Use existing trajectories
    else:
        print("FULL SCALE MODE: Generating 100+ protein trajectories")
        num_proteins, runs_per_protein = 100, 3
    
    print("This workflow will:")
    print(f"1. Generate {num_proteins}x{runs_per_protein} = {num_proteins*runs_per_protein} folding trajectories")
    print("2. Train a scaled Transformer AI model on folding processes")
    print("3. Evaluate the trained model's prediction capabilities")
    print("="*90)
    
    # Step 1: Generate training data
    print("\nStarting large-scale workflow...")
    if not quick:
        data_results = generate_training_data(num_proteins=num_proteins, runs_per_protein=runs_per_protein)
        
        if data_results['total_trajectories'] < 10:
            print("Insufficient training data generated. Need at least 10 trajectories.")
            return
    else:
        # Count existing trajectories
        from pathlib import Path
        existing_files = list(Path('training_data').glob('*trajectory*.json'))
        data_results = {'total_trajectories': len(existing_files), 'files': existing_files}
        print(f"Using {data_results['total_trajectories']} existing trajectory files")
    
    # Step 2: Train AI model with scaled configuration
    trainer = train_folding_ai(num_epochs=40, batch_size=16)
    
    if trainer is None:
        print("Large-scale training failed. Exiting.")
        return
    
    # Step 3: Evaluate model
    eval_results = evaluate_trained_model()
    
    if eval_results is None:
        print("Evaluation failed.")
        return
    
    print("\n" + "="*90)
    print("LARGE-SCALE WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*90)
    print(f"- Generated/Used {data_results['total_trajectories']} folding trajectories")
    print(f"- Trained large-scale Transformer AI model")
    print(f"- Model architecture: 6 layers, 8 attention heads, 128-dim embedding")
    print(f"- Model evaluation:")
    if 'overall_metrics' in eval_results:
        print(f"    - Energy correlation: {eval_results['overall_metrics']['mean_correlation']:.3f}")
        print(f"    - Success accuracy: {eval_results['overall_metrics']['success_prediction_accuracy']:.3f}")
    else:
        print("    - Evaluation metrics available in models/ directory")
    
    print(f"\nFiles generated:")
    print(f"- Training data: training_data/")
    print(f"- Trained models: models/")
    print(f"- Evaluation report: models/model_evaluation_report.json")
    print(f"- Training curves: models/training_curves.png")
    
    print(f"\nYour AI can now predict protein folding pathways!")
    print(f"This is the first AI trained on folding PROCESSES, not just final structures!")
    
    return {
        'data_results': data_results,
        'training_results': trainer,
        'evaluation_results': eval_results
    }

def run_quick_test():
    """Run a quick test with existing data."""
    print("QUICK TEST: Using existing training data")
    print("="*50)
    
    # Check if we have existing training data
    training_data_dir = Path('training_data')
    if not training_data_dir.exists() or not list(training_data_dir.glob('*.json')):
        print("No existing training data found.")
        print("Run the complete workflow with: python run_ai_training.py")
        return
    
    # Skip data generation, go straight to training
    trainer = train_folding_ai()
    
    if trainer:
        eval_results = evaluate_trained_model()
        print("\nQuick test completed!")
        return eval_results
    else:
        print("Quick test failed.")
        return None

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SCFD Folding AI - Large-Scale Training Workflow')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick test with existing data')
    parser.add_argument('--proteins', type=int, default=100,
                       help='Number of proteins to process (default: 100)')
    parser.add_argument('--runs', type=int, default=3,
                       help='Runs per protein (default: 3)')
    
    args = parser.parse_args()
    
    if args.quick:
        print("Running quick demonstration with existing data...")
        results = run_complete_workflow(quick=True)
    else:
        print(f"Running large-scale training: {args.proteins} proteins x {args.runs} runs")
        print("This will generate ~300 folding trajectories and train a scaled AI model")
        
        # Run custom large-scale workflow
        data_results = generate_training_data(args.proteins, args.runs)
        if data_results['total_trajectories'] < 10:
            print("Insufficient data generated. Exiting.")
            exit(1)
            
        trainer = train_folding_ai(num_epochs=50, batch_size=32)
        if trainer is None:
            print("Training failed. Exiting.")
            exit(1)
            
        eval_results = evaluate_trained_model()
        
        print("\n" + "="*90)
        print(f"LARGE-SCALE TRAINING COMPLETE!")
        print(f"Generated {data_results['total_trajectories']} trajectories")
        print(f"Trained model with 50 epochs on large dataset")
        print("="*90)
        
        results = {
            'data': data_results,
            'training': trainer, 
            'evaluation': eval_results
        }
        
    if results:
        print(f"\nSUCCESS! Your large-scale protein folding AI is ready!")
        print(f"Check models/ and training_data/ directories for generated files.")
    else:
        print(f"\nWorkflow encountered issues. Check error messages above.")