#!/usr/bin/env python3
"""
Test script for HDBSCAN clustering on VPR descriptors.
This script demonstrates how to use HDBSCAN clustering with t-SNE visualization.
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Test HDBSCAN clustering on toy dataset
    cmd = [
        sys.executable, "main.py",
        "--database_folder", "toy_dataset/database",
        "--queries_folder", "toy_dataset/queries",
        "--method", "netvlad",
        "--descriptors_dimension", "4096",
        "--log_dir", "hdbscan_test",
        "--plot_tsne",
        "--perform_hdbscan",
        "--hdbscan_min_cluster_size", "3",
        "--hdbscan_min_samples", "2",
        "--save_descriptors",
        "--num_preds_to_save", "5",
        "--no_labels"
    ]
    
    print("Running HDBSCAN clustering test...")
    print("Command:", " ".join(cmd))
    print("-" * 80)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print("-" * 80)
        print("Test completed successfully!")
        print("Check the logs/hdbscan_test directory for results:")
        print("- tsne_hdbscan_visualization.png: t-SNE plot with HDBSCAN clusters")
        print("- hdbscan_clusters/: Directory with images organized by cluster")
        print("- hdbscan_cluster_summary.txt: Summary of cluster assignments")
        print("- hdbscan_cluster_labels_*.npy: Numpy arrays with cluster labels")
        print("- hdbscan_probabilities.npy: Cluster membership probabilities")
        print("- hdbscan_outlier_scores.npy: Outlier scores for each point")
    else:
        print("Test failed with return code:", result.returncode)

if __name__ == "__main__":
    main() 