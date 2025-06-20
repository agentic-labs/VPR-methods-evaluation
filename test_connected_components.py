#!/usr/bin/env python3
"""
Simple test script for connected components visualization.
This creates synthetic data to demonstrate the functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import visualizations
    print("✓ Successfully imported visualizations module")
except ImportError as e:
    print(f"✗ Failed to import visualizations: {e}")
    sys.exit(1)

def create_synthetic_data(n_database=20, n_queries=5, n_features=128):
    """Create synthetic descriptors and paths for testing."""
    # Create some clustered data
    np.random.seed(42)
    
    # Create 3 clusters in the database
    cluster_centers = np.random.randn(3, n_features) * 10
    
    database_descriptors = []
    for i in range(n_database):
        cluster_id = i % 3
        descriptor = cluster_centers[cluster_id] + np.random.randn(n_features) * 2
        database_descriptors.append(descriptor)
    
    database_descriptors = np.array(database_descriptors)
    
    # Create queries - some near clusters, some far
    queries_descriptors = []
    for i in range(n_queries):
        if i < 3:  # First 3 queries near clusters
            cluster_id = i % 3
            descriptor = cluster_centers[cluster_id] + np.random.randn(n_features) * 2
        else:  # Remaining queries are outliers
            descriptor = np.random.randn(n_features) * 15
        queries_descriptors.append(descriptor)
    
    queries_descriptors = np.array(queries_descriptors)
    
    # Create fake paths
    database_paths = [f"synthetic_db_{i}.jpg" for i in range(n_database)]
    queries_paths = [f"synthetic_query_{i}.jpg" for i in range(n_queries)]
    
    return database_descriptors, queries_descriptors, database_paths, queries_paths

def main():
    print("\n=== Testing Connected Components Visualization ===\n")
    
    # Create synthetic data
    print("Creating synthetic data...")
    db_desc, q_desc, db_paths, q_paths = create_synthetic_data()
    print(f"✓ Created {len(db_desc)} database descriptors and {len(q_desc)} query descriptors")
    
    # Create output directory
    output_dir = Path("test_output_connected_components")
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Created output directory: {output_dir}")
    
    # Test with different numbers of neighbors
    for n_neighbors in [1, 2, 3]:
        print(f"\nTesting with {n_neighbors} nearest neighbor(s)...")
        
        sub_dir = output_dir / f"nn_{n_neighbors}"
        
        try:
            graph, components = visualizations.create_nn_graph_and_visualize_components(
                database_descriptors=db_desc,
                queries_descriptors=q_desc,
                database_paths=db_paths,
                queries_paths=q_paths,
                output_dir=sub_dir,
                n_neighbors=n_neighbors
            )
            
            print(f"✓ Successfully created visualization in {sub_dir}")
            print(f"  - Found {len(components)} connected components")
            print(f"  - Component sizes: {[len(c) for c in components[:5]]}...")
            
        except Exception as e:
            print(f"✗ Error with {n_neighbors} neighbors: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ Test completed! Check the '{output_dir}' directory for results.")
    print("\nTo use with real data, run:")
    print("python main.py --database_folder <path> --queries_folder <path> --visualize_connected_components --nn_graph_neighbors 1")

if __name__ == "__main__":
    main() 