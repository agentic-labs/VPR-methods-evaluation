#!/usr/bin/env python3
"""
Standalone script to create t-SNE visualizations from saved descriptors.

Usage:
    python visualize_saved_descriptors.py --db_descriptors path/to/database_descriptors.npy --query_descriptors path/to/queries_descriptors.npy
"""

import argparse
import numpy as np
from pathlib import Path
import visualizations

def main():
    parser = argparse.ArgumentParser(description="Create t-SNE visualization from saved descriptors")
    parser.add_argument("--db_descriptors", type=str, required=True, 
                        help="Path to database descriptors .npy file")
    parser.add_argument("--query_descriptors", type=str, required=True,
                        help="Path to query descriptors .npy file")
    parser.add_argument("--output_dir", type=str, default="tsne_output",
                        help="Directory to save visualization outputs")
    parser.add_argument("--perplexity", type=float, default=30,
                        help="t-SNE perplexity parameter")
    parser.add_argument("--n_iter", type=int, default=1000,
                        help="Number of t-SNE iterations")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load descriptors
    print(f"Loading database descriptors from {args.db_descriptors}")
    db_descriptors = np.load(args.db_descriptors)
    print(f"Database descriptors shape: {db_descriptors.shape}")
    
    print(f"Loading query descriptors from {args.query_descriptors}")
    query_descriptors = np.load(args.query_descriptors)
    print(f"Query descriptors shape: {query_descriptors.shape}")
    
    # Create basic t-SNE visualization
    print("\nCreating t-SNE visualization...")
    tsne_path = output_dir / "tsne_visualization.png"
    visualizations.plot_tsne(
        database_descriptors=db_descriptors,
        queries_descriptors=query_descriptors,
        save_path=tsne_path,
        perplexity=args.perplexity,
        n_iter=args.n_iter
    )
    
    print(f"\nVisualization saved to {tsne_path}")
    print(f"Embeddings saved to {output_dir / 'tsne_embeddings.npz'}")

if __name__ == "__main__":
    main() 