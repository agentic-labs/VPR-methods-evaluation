#!/usr/bin/env python3
"""
Test script to demonstrate k-means clustering with t-SNE visualization.

This script runs the VPR evaluation with k-means clustering enabled,
creating visualizations that show how images are grouped in the descriptor space.
"""

import subprocess
import sys
from pathlib import Path


def run_test():
    """Run the main script with k-means clustering and t-SNE visualization enabled."""
    
    # Check if toy dataset exists
    toy_dataset_path = Path("toy_dataset")
    if not toy_dataset_path.exists():
        print("Error: toy_dataset directory not found!")
        print("Please ensure the toy_dataset directory exists with database/ and queries/ subdirectories.")
        return 1
    
    # Command to run the main script with clustering
    cmd = [
        sys.executable,
        "main.py",
        "--database_folder", "toy_dataset/database",
        "--queries_folder", "toy_dataset/queries",
        "--method", "netvlad",
        "--descriptors_dimension", "4096",
        "--batch_size", "4",
        "--num_workers", "4",
        "--log_dir", "kmeans_tsne_test",
        "--save_descriptors",
        "--plot_tsne",
        "--perform_clustering",
        "--num_clusters", "5",
        "--num_preds_to_save", "5",
        "--no_labels"  # Using no_labels since it's a toy dataset
    ]
    
    print("Running VPR evaluation with k-means clustering and t-SNE visualization...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run the command
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("\n" + "="*80)
        print("SUCCESS! Check the following outputs:")
        print("1. logs/kmeans_tsne_test/*/tsne_visualization.png - Standard t-SNE plot")
        print("2. logs/kmeans_tsne_test/*/tsne_kmeans_visualization.png - t-SNE with k-means clusters")
        print("3. logs/kmeans_tsne_test/*/clusters/ - Images organized by cluster")
        print("4. logs/kmeans_tsne_test/*/clusters/cluster_summary.txt - Cluster statistics")
        print("5. logs/kmeans_tsne_test/*/preds/ - Query predictions with confidence scores")
        print("="*80)
    else:
        print(f"\nError: Command failed with return code {result.returncode}")
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(run_test()) 