#!/usr/bin/env python3
"""
Test script for hierarchical clustering with average linkage and cosine distance.
This script demonstrates the hierarchical clustering functionality with proper visualizations.
"""

import subprocess
import sys
from pathlib import Path

def run_hierarchical_clustering():
    """Run the VPR evaluation with hierarchical clustering."""
    
    # Base command
    cmd = [
        sys.executable, "main.py",
        "--database_folder", "assets/database",
        "--queries_folder", "assets/queries",
        "--method", "netvlad",
        "--descriptors_dimension", "4096",
        "--save_descriptors",
        "--plot_tsne",
        "--perform_hierarchical",
        "--hierarchical_distance_threshold", "0.5",  # Default cosine distance threshold
        "--num_preds_to_save", "5",
        "--log_dir", "hierarchical_clustering_test"
    ]
    
    print("Running hierarchical clustering with average linkage and cosine distance...")
    print(f"Command: {' '.join(cmd)}")
    print("\nThis will:")
    print("1. Extract normalized descriptors using NetVLAD")
    print("2. Perform hierarchical clustering with average linkage")
    print("3. Use cosine distance (threshold=0.5)")
    print("4. Generate a dendrogram visualization")
    print("5. Create t-SNE plots with cluster colors")
    print("6. Save images organized by clusters")
    print("-" * 50)
    
    # Run the command
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\n✅ Hierarchical clustering completed successfully!")
        print("\nCheck the following outputs:")
        print("- logs/hierarchical_clustering_test/*/hierarchical_dendrogram.png - Full dendrogram")
        print("- logs/hierarchical_clustering_test/*/tsne_hierarchical_visualization.png - t-SNE with clusters")
        print("- logs/hierarchical_clustering_test/*/hierarchical_clusters/ - Images organized by cluster")
        print("- logs/hierarchical_clustering_test/*/hierarchical_cluster_summary.txt - Cluster statistics")
    else:
        print("\n❌ Error running hierarchical clustering")
        return result.returncode
    
    return 0

def run_hierarchical_with_different_thresholds():
    """Run hierarchical clustering with different distance thresholds for comparison."""
    
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"Running with distance threshold = {threshold}")
        print(f"{'='*60}")
        
        cmd = [
            sys.executable, "main.py",
            "--database_folder", "assets/database",
            "--queries_folder", "assets/queries",
            "--method", "netvlad",
            "--descriptors_dimension", "4096",
            "--save_descriptors",
            "--plot_tsne",
            "--perform_hierarchical",
            "--hierarchical_distance_threshold", str(threshold),
            "--num_preds_to_save", "3",
            "--log_dir", f"hierarchical_threshold_{threshold}"
        ]
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode != 0:
            print(f"\n❌ Error with threshold {threshold}")
            return result.returncode
    
    print("\n✅ All threshold experiments completed!")
    print("\nCompare the results in:")
    for threshold in thresholds:
        print(f"- logs/hierarchical_threshold_{threshold}/")
    
    return 0

if __name__ == "__main__":
    print("Hierarchical Clustering Test Script")
    print("==================================\n")
    
    # Check if toy dataset exists
    if not Path("assets/database").exists():
        print("⚠️  Using toy_dataset instead of assets")
        # Update paths in the functions above
        import re
        import inspect
        
        # Get the source of both functions
        for func in [run_hierarchical_clustering, run_hierarchical_with_different_thresholds]:
            source = inspect.getsource(func)
            # Replace assets with toy_dataset
            new_source = source.replace('"assets/', '"toy_dataset/')
            # Execute the modified function
            exec(compile(new_source, '<string>', 'exec'), globals())
    
    # Run basic hierarchical clustering
    ret = run_hierarchical_clustering()
    
    if ret == 0:
        print("\n" + "="*60)
        print("Would you like to run experiments with different thresholds? (y/n)")
        response = input().strip().lower()
        
        if response == 'y':
            ret = run_hierarchical_with_different_thresholds()
    
    sys.exit(ret) 