import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

class SymbolicProteinVisualizer:
    """3D visualization tools for symbolic protein grids."""
    
    def __init__(self):
        self.symbol_colors = {
            -1: 'lightgray',    # Empty space
            0: 'red',           # Hydrophobic
            1: 'blue',          # Polar
            2: 'yellow'         # Charged
        }
        
        self.symbol_names = {
            -1: 'Empty',
            0: 'Hydrophobic', 
            1: 'Polar',
            2: 'Charged'
        }
    
    def load_grid(self, file_path):
        """Load a symbolic grid from .npy file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        grid = np.load(file_path)
        print(f"Loaded grid shape: {grid.shape}")
        
        # Get statistics
        unique, counts = np.unique(grid, return_counts=True)
        print("Symbol distribution:")
        for symbol, count in zip(unique, counts):
            name = self.symbol_names.get(symbol, f"Unknown({symbol})")
            percentage = count / grid.size * 100
            print(f"  {name}: {count:,} voxels ({percentage:.1f}%)")
        
        return grid
    
    def get_occupied_voxels(self, grid, symbol=None):
        """Get coordinates of occupied voxels, optionally filtered by symbol."""
        if symbol is None:
            # All non-empty voxels
            coords = np.where(grid != -1)
        else:
            # Specific symbol
            coords = np.where(grid == symbol)
        
        x, y, z = coords
        symbols = grid[coords]
        
        return x, y, z, symbols
    
    def create_3d_scatter_matplotlib(self, grid, max_points=10000, figsize=(12, 10)):
        """Create 3D scatter plot using matplotlib."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get occupied voxels
        x, y, z, symbols = self.get_occupied_voxels(grid)
        
        # Subsample if too many points
        if len(x) > max_points:
            indices = np.random.choice(len(x), max_points, replace=False)
            x, y, z, symbols = x[indices], y[indices], z[indices], symbols[indices]
            print(f"Subsampled to {max_points} points for visualization")
        
        # Plot each symbol type
        for symbol in np.unique(symbols):
            if symbol == -1:
                continue  # Skip empty voxels
            
            mask = symbols == symbol
            if np.any(mask):
                ax.scatter(x[mask], y[mask], z[mask], 
                          c=self.symbol_colors[symbol], 
                          label=self.symbol_names[symbol],
                          alpha=0.6, s=20)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('3D Symbolic Protein Structure')
        
        return fig, ax
    
    def create_3d_scatter_plotly(self, grid, max_points=50000):
        """Create interactive 3D scatter plot using plotly."""
        # Get occupied voxels
        x, y, z, symbols = self.get_occupied_voxels(grid)
        
        # Subsample if too many points
        if len(x) > max_points:
            indices = np.random.choice(len(x), max_points, replace=False)
            x, y, z, symbols = x[indices], y[indices], z[indices], symbols[indices]
            print(f"Subsampled to {max_points} points for visualization")
        
        # Create color mapping
        colors = [self.symbol_colors[s] for s in symbols]
        text = [self.symbol_names[s] for s in symbols]
        
        fig = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=3,
                color=colors,
                opacity=0.7
            ),
            text=text,
            hovertemplate='<b>%{text}</b><br>' +
                         'X: %{x}<br>' +
                         'Y: %{y}<br>' +
                         'Z: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Interactive 3D Symbolic Protein Structure',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_slice_view(self, grid, slice_axis='z', slice_idx=None, figsize=(15, 5)):
        """Create 2D slice views along different axes."""
        if slice_idx is None:
            slice_idx = grid.shape[slice_axis_map[slice_axis]] // 2
        
        slice_axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_idx = slice_axis_map[slice_axis]
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Extract slices
        if axis_idx == 0:  # X slice
            slice_data = grid[slice_idx, :, :]
            axes[0].set_title(f'X={slice_idx} slice')
        elif axis_idx == 1:  # Y slice  
            slice_data = grid[:, slice_idx, :]
            axes[0].set_title(f'Y={slice_idx} slice')
        else:  # Z slice
            slice_data = grid[:, :, slice_idx]
            axes[0].set_title(f'Z={slice_idx} slice')
        
        # Main slice view
        im = axes[0].imshow(slice_data, cmap='viridis', vmin=-1, vmax=2)
        axes[0].set_aspect('equal')
        plt.colorbar(im, ax=axes[0])
        
        # Symbol distribution
        unique, counts = np.unique(slice_data, return_counts=True)
        symbol_labels = [self.symbol_names[s] for s in unique]
        axes[1].bar(range(len(unique)), counts, color=[self.symbol_colors[s] for s in unique])
        axes[1].set_xticks(range(len(unique)))
        axes[1].set_xticklabels(symbol_labels, rotation=45)
        axes[1].set_title('Symbol Distribution in Slice')
        axes[1].set_ylabel('Count')
        
        # Overall grid statistics
        full_unique, full_counts = np.unique(grid, return_counts=True)
        full_labels = [self.symbol_names[s] for s in full_unique]
        axes[2].bar(range(len(full_unique)), full_counts, 
                   color=[self.symbol_colors[s] for s in full_unique])
        axes[2].set_xticks(range(len(full_unique)))
        axes[2].set_xticklabels(full_labels, rotation=45)
        axes[2].set_title('Full Grid Distribution')
        axes[2].set_ylabel('Count')
        
        plt.tight_layout()
        return fig, axes
    
    def create_density_projection(self, grid, figsize=(15, 5)):
        """Create density projections along each axis."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Project along each axis (count non-empty voxels)
        proj_x = np.sum(grid != -1, axis=0)  # Sum along X
        proj_y = np.sum(grid != -1, axis=1)  # Sum along Y  
        proj_z = np.sum(grid != -1, axis=2)  # Sum along Z
        
        # Plot projections
        im1 = axes[0].imshow(proj_x, cmap='hot', aspect='equal')
        axes[0].set_title('Projection along X-axis (Y-Z view)')
        axes[0].set_xlabel('Z'); axes[0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(proj_y, cmap='hot', aspect='equal')
        axes[1].set_title('Projection along Y-axis (X-Z view)')
        axes[1].set_xlabel('Z'); axes[1].set_ylabel('X')
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(proj_z, cmap='hot', aspect='equal')
        axes[2].set_title('Projection along Z-axis (X-Y view)')
        axes[2].set_xlabel('Y'); axes[2].set_ylabel('X')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        return fig, axes
    
    def analyze_structure(self, grid):
        """Analyze structural properties of the symbolic grid."""
        print("=== Structural Analysis ===")
        
        # Basic statistics
        total_voxels = grid.size
        occupied_voxels = np.sum(grid != -1)
        occupancy_ratio = occupied_voxels / total_voxels
        
        print(f"Grid shape: {grid.shape}")
        print(f"Total voxels: {total_voxels:,}")
        print(f"Occupied voxels: {occupied_voxels:,} ({occupancy_ratio:.1%})")
        
        # Symbol distribution
        unique, counts = np.unique(grid, return_counts=True)
        print(f"\nSymbol distribution:")
        for symbol, count in zip(unique, counts):
            name = self.symbol_names.get(symbol, f"Unknown({symbol})")
            percentage = count / total_voxels * 100
            print(f"  {name}: {count:,} ({percentage:.1f}%)")
        
        # Spatial extent
        if occupied_voxels > 0:
            x, y, z, _ = self.get_occupied_voxels(grid)
            extents = {
                'X': (x.min(), x.max(), x.max() - x.min() + 1),
                'Y': (y.min(), y.max(), y.max() - y.min() + 1), 
                'Z': (z.min(), z.max(), z.max() - z.min() + 1)
            }
            
            print(f"\nSpatial extents:")
            for axis, (min_val, max_val, span) in extents.items():
                print(f"  {axis}: {min_val} to {max_val} (span: {span})")
        
        return {
            'total_voxels': total_voxels,
            'occupied_voxels': occupied_voxels,
            'occupancy_ratio': occupancy_ratio,
            'symbol_counts': dict(zip(unique, counts)),
            'extents': extents if occupied_voxels > 0 else None
        }

def main():
    """Example usage of the visualizer."""
    visualizer = SymbolicProteinVisualizer()
    
    # Load the processed file
    processed_file = "../processed/AF-A0A385XJ53-F1-model_v4.npy"
    
    if os.path.exists(processed_file):
        grid = visualizer.load_grid(processed_file)
        
        # Analyze structure
        analysis = visualizer.analyze_structure(grid)
        
        # Create visualizations
        print("\nCreating matplotlib 3D plot...")
        fig_3d, ax_3d = visualizer.create_3d_scatter_matplotlib(grid)
        plt.show()
        
        print("Creating slice view...")
        fig_slice, ax_slice = visualizer.create_slice_view(grid)
        plt.show()
        
        print("Creating density projections...")
        fig_proj, ax_proj = visualizer.create_density_projection(grid)
        plt.show()
        
        # Create interactive plotly plot
        print("Creating interactive plotly plot...")
        fig_interactive = visualizer.create_3d_scatter_plotly(grid)
        fig_interactive.show()
        
    else:
        print(f"Processed file not found: {processed_file}")
        print("Please run the batch processor first.")

if __name__ == "__main__":
    main()