import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
import os
import glob
from tqdm import tqdm
import gzip
import shutil
from process_protein import AMINO_ACID_MAP, get_protein_structure, voxelize_protein_to_symbolic_grid

class AlphaFoldBatchProcessor:
    """Batch processor for AlphaFold protein structures."""
    
    def __init__(self, raw_folder="raw", processed_folder="processed"):
        self.raw_folder = raw_folder
        self.processed_folder = processed_folder
        os.makedirs(processed_folder, exist_ok=True)
    
    def find_protein_files(self, extensions=['.cif', '.pdb', '.cif.gz', '.pdb.gz']):
        """Find all protein structure files in the raw folder."""
        files = []
        for ext in extensions:
            pattern = os.path.join(self.raw_folder, "**", f"*{ext}")
            files.extend(glob.glob(pattern, recursive=True))
        return sorted(files)
    
    def decompress_if_needed(self, file_path):
        """Decompress .gz files temporarily if needed."""
        if file_path.endswith('.gz'):
            temp_path = file_path[:-3]  # Remove .gz
            if not os.path.exists(temp_path):
                print(f"Decompressing {os.path.basename(file_path)}...")
                with gzip.open(file_path, 'rb') as f_in:
                    with open(temp_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            return temp_path, True  # Return path and whether we created temp file
        return file_path, False
    
    def cleanup_temp_file(self, file_path, was_temp):
        """Remove temporary decompressed file if we created it."""
        if was_temp and os.path.exists(file_path):
            os.remove(file_path)
    
    def process_single_file(self, file_path, grid_size=64, voxel_size=1.5):
        """Process a single protein file to symbolic grid."""
        try:
            # Handle compressed files
            working_path, is_temp = self.decompress_if_needed(file_path)
            
            # Get output filename
            base_name = os.path.basename(file_path)
            if base_name.endswith('.gz'):
                base_name = base_name[:-3]
            output_name = base_name.replace('.cif', '.npy').replace('.pdb', '.npy')
            output_path = os.path.join(self.processed_folder, output_name)
            
            # Skip if already processed
            if os.path.exists(output_path):
                print(f"Skipping {base_name} (already processed)")
                self.cleanup_temp_file(working_path, is_temp)
                return output_path, "skipped"
            
            # Load and process structure
            structure = get_protein_structure(working_path)
            symbolic_grid = voxelize_protein_to_symbolic_grid(structure, grid_size, voxel_size)
            
            # Save result
            np.save(output_path, symbolic_grid)
            
            # Cleanup
            self.cleanup_temp_file(working_path, is_temp)
            
            return output_path, "success"
            
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
            if 'working_path' in locals() and 'is_temp' in locals():
                self.cleanup_temp_file(working_path, is_temp)
            return None, f"error: {str(e)}"
    
    def process_batch(self, grid_size=64, voxel_size=1.5, max_files=None):
        """Process all protein files in batch."""
        files = self.find_protein_files()
        
        if max_files:
            files = files[:max_files]
        
        print(f"Found {len(files)} protein files to process")
        
        results = {
            'success': [],
            'skipped': [],
            'errors': []
        }
        
        for file_path in tqdm(files, desc="Processing proteins"):
            output_path, status = self.process_single_file(file_path, grid_size, voxel_size)
            
            if status == "success":
                results['success'].append((file_path, output_path))
            elif status == "skipped":
                results['skipped'].append(file_path)
            else:
                results['errors'].append((file_path, status))
        
        # Print summary
        print(f"\nBatch processing complete:")
        print(f"  Successfully processed: {len(results['success'])}")
        print(f"  Skipped (already done): {len(results['skipped'])}")
        print(f"  Errors: {len(results['errors'])}")
        
        if results['errors']:
            print("\nErrors encountered:")
            for file_path, error in results['errors'][:5]:  # Show first 5 errors
                print(f"  {os.path.basename(file_path)}: {error}")
            if len(results['errors']) > 5:
                print(f"  ... and {len(results['errors']) - 5} more errors")
        
        return results
    
    def get_processed_files(self):
        """Get list of all processed .npy files."""
        pattern = os.path.join(self.processed_folder, "*.npy")
        return sorted(glob.glob(pattern))
    
    def get_file_stats(self):
        """Get statistics about processed files."""
        processed_files = self.get_processed_files()
        
        if not processed_files:
            return {"total_files": 0}
        
        # Sample one file to get grid info
        sample_grid = np.load(processed_files[0])
        
        stats = {
            "total_files": len(processed_files),
            "grid_shape": sample_grid.shape,
            "grid_size_mb": sample_grid.nbytes / (1024 * 1024),
            "total_size_mb": len(processed_files) * sample_grid.nbytes / (1024 * 1024),
            "symbol_counts": {
                "hydrophobic (0)": 0,
                "polar (1)": 0, 
                "charged (2)": 0,
                "empty (-1)": 0
            }
        }
        
        # Count symbols across all files (sample first 10 for speed)
        sample_files = processed_files[:min(10, len(processed_files))]
        for file_path in sample_files:
            grid = np.load(file_path)
            unique, counts = np.unique(grid, return_counts=True)
            for symbol, count in zip(unique, counts):
                if symbol == -1:
                    stats["symbol_counts"]["empty (-1)"] += count
                elif symbol == 0:
                    stats["symbol_counts"]["hydrophobic (0)"] += count
                elif symbol == 1:
                    stats["symbol_counts"]["polar (1)"] += count
                elif symbol == 2:
                    stats["symbol_counts"]["charged (2)"] += count
        
        return stats

if __name__ == "__main__":
    # Example usage
    processor = AlphaFoldBatchProcessor()
    
    # Process first 5 files as test
    results = processor.process_batch(max_files=5)
    
    # Show statistics
    stats = processor.get_file_stats()
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")