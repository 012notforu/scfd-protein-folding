"""
AI Training Pipeline for SCFD Folding Pathways
==============================================

Trains neural networks on protein folding process data from SCFD simulations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

class FoldingTrajectoryDataset(Dataset):
    """Dataset for loading SCFD folding trajectory data."""
    
    def __init__(self, trajectory_files, sequence_length=64, max_timesteps=50, batch_size=32):
        self.trajectory_files = trajectory_files
        self.sequence_length = sequence_length
        self.max_timesteps = max_timesteps
        self.batch_size = batch_size
        self.data = []
        
        print(f"Loading {len(trajectory_files)} trajectory files for large-scale training...")
        
        # Process in batches for memory efficiency
        batch_count = 0
        for i in range(0, len(trajectory_files), batch_size):
            batch_files = trajectory_files[i:i+batch_size]
            batch_count += 1
            print(f"Processing batch {batch_count}/{(len(trajectory_files) + batch_size - 1) // batch_size}")
            
            for file_path in tqdm(batch_files, desc=f"Batch {batch_count}"):
                try:
                    with open(file_path, 'r') as f:
                        trajectory = json.load(f)
                    self.data.append(self._process_trajectory(trajectory))
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
                    
        print(f"Large-scale dataset loaded: {len(self.data)} valid trajectories")
    
    def _process_trajectory(self, trajectory):
        """Convert JSON trajectory to training tensors."""
        folding_steps = trajectory['folding_trajectory']
        
        # Pad/truncate to fixed length
        if len(folding_steps) > self.max_timesteps:
            folding_steps = folding_steps[:self.max_timesteps]
        
        processed_data = {
            'protein_id': trajectory['protein_id'],
            'sequence_length': trajectory['sequence_info']['sequence_length'],
            'timesteps': [],
            'grid_states': [],
            'field_states': [],
            'energies': [],
            'mutations': [],
            'success': trajectory['final_metrics']['folding_success']
        }
        
        for step in folding_steps:
            # Grid state (simplified - just symbol counts per region)
            grid_summary = self._summarize_grid(step.get('grid_state', {}))
            processed_data['grid_states'].append(grid_summary)
            
            # Field states
            field_summary = self._summarize_fields(step.get('field_data', {}))
            processed_data['field_states'].append(field_summary)
            
            # Energy
            processed_data['energies'].append(step['energy_data']['total_energy'])
            
            # Mutations
            mutation_summary = self._summarize_mutations(step.get('mutations_this_step', []))
            processed_data['mutations'].append(mutation_summary)
            
            processed_data['timesteps'].append(step['timestep'])
        
        return processed_data
    
    def _summarize_grid(self, grid_state):
        """Summarize grid state into fixed-size feature vector."""
        if not grid_state or 'symbols' not in grid_state:
            return np.zeros(20)  # 14 symbol counts + 6 spatial features
        
        symbols = np.array(grid_state['symbols'])
        positions = np.array(grid_state['positions'])
        
        # Symbol histogram (14 symbols)
        symbol_hist = np.histogram(symbols, bins=range(15), density=True)[0]
        
        # Spatial features
        if len(positions) > 0:
            center_of_mass = np.mean(positions, axis=0)
            radius_of_gyration = np.sqrt(np.mean(np.sum((positions - center_of_mass)**2, axis=1)))
            spatial_features = [
                len(symbols),  # total occupied voxels
                radius_of_gyration,  # compactness
                np.std(positions[:, 0]) if len(positions) > 1 else 0,  # x spread
                np.std(positions[:, 1]) if len(positions) > 1 else 0,  # y spread  
                np.std(positions[:, 2]) if len(positions) > 1 else 0,  # z spread
                len(np.unique(symbols))  # symbol diversity
            ]
        else:
            spatial_features = [0] * 6
            
        return np.concatenate([symbol_hist, spatial_features])
    
    def _summarize_fields(self, field_data):
        """Summarize field states into feature vector."""
        features = []
        for field_name in ['coherence_field', 'curvature_field', 'entropy_field']:
            field = field_data.get(field_name, {})
            if field:
                features.extend([
                    field.get('mean', 0),
                    field.get('std', 0),
                    field.get('min', 0),
                    field.get('max', 0)
                ])
            else:
                features.extend([0, 0, 0, 0])
        return np.array(features)
    
    def _summarize_mutations(self, mutations):
        """Summarize mutations into feature vector."""
        if not mutations:
            return np.zeros(5)
        
        mutation_count = len(mutations)
        avg_energy_delta = np.mean([m.get('energy_delta', 0) for m in mutations])
        symbol_changes = {}
        
        for mut in mutations:
            from_sym = mut.get('from', -1)
            to_sym = mut.get('to', -1)
            change_type = f"{from_sym}->{to_sym}"
            symbol_changes[change_type] = symbol_changes.get(change_type, 0) + 1
        
        return np.array([
            mutation_count,
            avg_energy_delta,
            len(symbol_changes),  # diversity of mutation types
            max(symbol_changes.values()) if symbol_changes else 0,  # most common change
            np.sum([abs(m.get('energy_delta', 0)) for m in mutations])  # total energy change
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        trajectory = self.data[idx]
        
        # Convert to tensors and pad sequences
        max_steps = self.max_timesteps
        actual_steps = len(trajectory['grid_states'])
        
        # Debug print
        # print(f"Trajectory {idx}: {actual_steps} steps, max={max_steps}")
        
        # Input features: grid + fields + mutations
        grid_features = np.array(trajectory['grid_states'])
        field_features = np.array(trajectory['field_states']) 
        mutation_features = np.array(trajectory['mutations'])
        
        # Debug shapes
        # print(f"  Grid: {grid_features.shape}, Field: {field_features.shape}, Mut: {mutation_features.shape}")
        
        # Truncate or pad sequences to fixed length
        if actual_steps > max_steps:
            grid_features = grid_features[:max_steps]
            field_features = field_features[:max_steps]
            mutation_features = mutation_features[:max_steps]
            actual_steps = max_steps
        elif actual_steps < max_steps:
            pad_length = max_steps - actual_steps
            grid_features = np.pad(grid_features, ((0, pad_length), (0, 0)), 'constant')
            field_features = np.pad(field_features, ((0, pad_length), (0, 0)), 'constant')
            mutation_features = np.pad(mutation_features, ((0, pad_length), (0, 0)), 'constant')
        
        # Combine all features
        input_features = np.concatenate([grid_features, field_features, mutation_features], axis=1)
        
        # Create input/target pairs with matching sequence lengths
        input_sequence_length = max_steps - 1  # Reserve one step for targets
        
        # Input features: first (max_steps-1) timesteps
        input_features_final = input_features[:input_sequence_length]
        
        # Target grids: timesteps 1 to max_steps (shifted by one)
        target_grids = grid_features[1:input_sequence_length+1]
        
        # Target energies: match the target grids exactly
        energies_array = np.array(trajectory['energies'])
        
        # Ensure we have enough energies for the target sequence
        needed_energy_length = input_sequence_length + 1  # Need one extra for shifting
        
        if len(energies_array) < needed_energy_length:
            # Pad with last energy value
            last_energy = energies_array[-1] if len(energies_array) > 0 else 0
            pad_length = needed_energy_length - len(energies_array)
            energies_array = np.pad(energies_array, (0, pad_length), 'constant', constant_values=last_energy)
        
        # Target energies: timesteps 1 to input_sequence_length+1
        target_energies = energies_array[1:input_sequence_length+1]
        
        # Ensure all tensors have exactly the same sequence length
        assert len(input_features_final) == len(target_grids) == len(target_energies), \
            f"Sequence length mismatch: input={len(input_features_final)}, target_grids={len(target_grids)}, target_energies={len(target_energies)}"
        
        return {
            'input_features': torch.FloatTensor(input_features_final),
            'target_grids': torch.FloatTensor(target_grids),
            'target_energies': torch.FloatTensor(target_energies), 
            'sequence_length': len(input_features_final),
            'folding_success': trajectory['success']
        }


class FoldingTransformer(nn.Module):
    """Transformer model for predicting folding pathways."""
    
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=6, max_seq_len=50):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output heads
        self.grid_predictor = nn.Linear(d_model, 20)  # Grid state features
        self.energy_predictor = nn.Linear(d_model, 1)  # Energy prediction
        self.success_predictor = nn.Linear(d_model, 1)  # Folding success
        
    def _create_positional_encoding(self, max_len, d_model):
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x, mask=None):
        # Input projection and positional encoding
        x = self.input_projection(x) * np.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer encoding
        if mask is not None:
            # Create causal mask for autoregressive prediction
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            transformer_output = self.transformer(x, mask=causal_mask.to(x.device))
        else:
            transformer_output = self.transformer(x)
        
        # Predictions
        grid_pred = self.grid_predictor(transformer_output)
        energy_pred = self.energy_predictor(transformer_output).squeeze(-1)
        success_pred = torch.sigmoid(self.success_predictor(transformer_output[:, -1, :]).squeeze(-1))
        
        return {
            'grid_prediction': grid_pred,
            'energy_prediction': energy_pred,
            'success_prediction': success_pred
        }


class FoldingAITrainer:
    """Complete training pipeline for folding AI."""
    
    def __init__(self, data_dir='training_data', model_dir='models'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def prepare_data(self, test_size=0.2, batch_size=16, load_batch_size=32):
        """Load and prepare large-scale training data."""
        # Find all trajectory files
        trajectory_files = list(self.data_dir.glob("*_folding_trajectory_*.json"))
        print(f"Found {len(trajectory_files)} trajectory files for large-scale training")
        
        if len(trajectory_files) < 2:
            raise ValueError("Need at least 2 trajectory files for train/test split")
        
        # Split files into train/test
        train_files, test_files = train_test_split(
            trajectory_files, test_size=test_size, random_state=42
        )
        
        print(f"Large-scale data split: {len(train_files)} train, {len(test_files)} test files")
        
        # Create datasets with batch processing
        train_dataset = FoldingTrajectoryDataset(train_files, batch_size=load_batch_size)
        test_dataset = FoldingTrajectoryDataset(test_files, batch_size=load_batch_size)
        
        # Create dataloaders with optimized settings for large datasets
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Get input dimension from first batch
        sample_batch = next(iter(self.train_loader))
        self.input_dim = sample_batch['input_features'].shape[-1]
        print(f"Input dimension: {self.input_dim}")
        
        return len(train_files), len(test_files)
    
    def create_model(self, d_model=128, nhead=8, num_layers=4):
        """Create and initialize the model."""
        self.model = FoldingTransformer(
            input_dim=self.input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        ).to(self.device)
        
        # Store model config for saving
        self.model_config = {
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'max_seq_len': 50
        }
        
        # Loss functions
        self.grid_criterion = nn.MSELoss()
        self.energy_criterion = nn.MSELoss()
        self.success_criterion = nn.BCELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        print(f"Created model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        grid_loss_sum = 0
        energy_loss_sum = 0
        success_loss_sum = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            # Move to device
            inputs = batch['input_features'].to(self.device)
            target_grids = batch['target_grids'].to(self.device)
            target_energies = batch['target_energies'].to(self.device)
            target_success = batch['folding_success'].float().to(self.device)
            
            # Forward pass
            outputs = self.model(inputs, mask=True)
            
            # Compute losses
            grid_loss = self.grid_criterion(outputs['grid_prediction'], target_grids)
            energy_loss = self.energy_criterion(outputs['energy_prediction'], target_energies)
            success_loss = self.success_criterion(outputs['success_prediction'], target_success)
            
            # Combined loss
            total_batch_loss = grid_loss + energy_loss + success_loss
            
            # Backward pass
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_batch_loss.item()
            grid_loss_sum += grid_loss.item()
            energy_loss_sum += energy_loss.item()
            success_loss_sum += success_loss.item()
        
        return {
            'total_loss': total_loss / len(self.train_loader),
            'grid_loss': grid_loss_sum / len(self.train_loader),
            'energy_loss': energy_loss_sum / len(self.train_loader),
            'success_loss': success_loss_sum / len(self.train_loader)
        }
    
    def evaluate(self):
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                inputs = batch['input_features'].to(self.device)
                target_grids = batch['target_grids'].to(self.device)
                target_energies = batch['target_energies'].to(self.device)
                target_success = batch['folding_success'].float().to(self.device)
                
                outputs = self.model(inputs)
                
                # Compute losses
                grid_loss = self.grid_criterion(outputs['grid_prediction'], target_grids)
                energy_loss = self.energy_criterion(outputs['energy_prediction'], target_energies)
                success_loss = self.success_criterion(outputs['success_prediction'], target_success)
                
                total_loss += (grid_loss + energy_loss + success_loss).item()
                
                # Success prediction accuracy
                success_pred = (outputs['success_prediction'] > 0.5).float()
                correct_predictions += (success_pred == target_success).sum().item()
                total_predictions += target_success.size(0)
        
        return {
            'test_loss': total_loss / len(self.test_loader),
            'success_accuracy': correct_predictions / total_predictions
        }
    
    def train(self, num_epochs=50, save_every=10):
        """Complete training loop."""
        train_losses = []
        test_losses = []
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch()
            train_losses.append(train_metrics['total_loss'])
            
            # Evaluation
            test_metrics = self.evaluate()
            test_losses.append(test_metrics['test_loss'])
            
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"Test Loss: {test_metrics['test_loss']:.4f}")
            print(f"Success Accuracy: {test_metrics['success_accuracy']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(test_metrics['test_loss'])
            
            # Save model checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1, train_metrics, test_metrics)
            
        # Final model save
        self.save_model('final_folding_model.pt')
        
        # Plot training curves
        self.plot_training_curves(train_losses, test_losses)
        
        return train_losses, test_losses
    
    def save_checkpoint(self, epoch, train_metrics, test_metrics):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'input_dim': self.input_dim
        }
        
        checkpoint_path = self.model_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def save_model(self, filename):
        """Save final trained model."""
        model_path = self.model_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'model_config': self.model_config
        }, model_path)
        print(f"Model saved: {model_path}")
    
    def plot_training_curves(self, train_losses, test_losses):
        """Plot training and test loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(test_losses, label='Test Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Over Time')
        plt.legend()
        plt.grid(True)
        
        plot_path = self.model_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training curves saved: {plot_path}")


def run_training_pipeline():
    """Run the complete training pipeline."""
    trainer = FoldingAITrainer()
    
    # Prepare data
    train_size, test_size = trainer.prepare_data(batch_size=8)
    print(f"Training on {train_size} proteins, testing on {test_size} proteins")
    
    # Create model
    trainer.create_model(d_model=128, nhead=8, num_layers=4)
    
    # Train
    train_losses, test_losses = trainer.train(num_epochs=30, save_every=10)
    
    print("Training completed!")
    return trainer

if __name__ == '__main__':
    trainer = run_training_pipeline()