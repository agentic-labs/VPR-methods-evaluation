#!/usr/bin/env python3
"""
Test script for t-SNE visualization of VPR descriptors.

Example usage:
    python test_tsne.py --database_folder toy_dataset/database --queries_folder toy_dataset/queries --plot_tsne --method netvlad
"""

import sys
import subprocess

# Example command to run the main script with t-SNE visualization
example_commands = [
    # Basic example with NetVLAD
    [
        "python", "main.py",
        "--database_folder", "toy_dataset/database",
        "--queries_folder", "toy_dataset/queries",
        "--method", "netvlad",
        "--plot_tsne",
        "--num_preds_to_save", "5",
        "--log_dir", "tsne_test"
    ],
    
    # Example with CosPlace and different backbone
    [
        "python", "main.py",
        "--database_folder", "toy_dataset/database",
        "--queries_folder", "toy_dataset/queries",
        "--method", "cosplace",
        "--backbone", "ResNet50",
        "--descriptors_dimension", "512",
        "--plot_tsne",
        "--save_descriptors",
        "--log_dir", "tsne_test_cosplace"
    ],
    
    # Example without labels
    [
        "python", "main.py",
        "--database_folder", "toy_dataset/database",
        "--queries_folder", "toy_dataset/queries",
        "--method", "mixvpr",
        "--plot_tsne",
        "--no_labels",
        "--log_dir", "tsne_test_no_labels"
    ]
]

if __name__ == "__main__":
    print("Running t-SNE visualization examples...")
    print("=" * 50)
    
    # Run the first example by default
    cmd = example_commands[0]
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("t-SNE visualization completed successfully!")
        print("Check the logs directory for the output files:")
        print("- tsne_visualization.png: Basic scatter plot")
        print("- tsne_visualization_enhanced.png: Plot with query-prediction connections")
        print("- tsne_embeddings.npz: Saved embeddings for further analysis")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1) 