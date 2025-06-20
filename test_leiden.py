#!/usr/bin/env python3
"""
Test script for hierarchical Leiden community detection.
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

def create_hierarchical_synthetic_data(n_database=50, n_queries=15, n_features=128):
    """Create synthetic descriptors with hierarchical cluster structure."""
    np.random.seed(42)
    
    # Create hierarchical structure:
    # 2 super-clusters, each with 2 sub-clusters
    super_centers = np.random.randn(2, n_features) * 20
    
    database_descriptors = []
    for i in range(n_database):
        # Determine super-cluster
        super_cluster = i % 2
        # Determine sub-cluster within super-cluster
        sub_cluster = (i // 2) % 2
        
        # Create descriptor with hierarchical noise
        descriptor = super_centers[super_cluster].copy()
        sub_offset = np.random.randn(n_features) * 5
        descriptor += sub_offset * (1 if sub_cluster == 0 else -1)
        descriptor += np.random.randn(n_features) * 1  # Local noise
        
        database_descriptors.append(descriptor)
    
    database_descriptors = np.array(database_descriptors)
    
    # Create queries - some matching clusters, some outliers
    queries_descriptors = []
    for i in range(n_queries):
        if i < 8:  # First 8 queries match existing clusters
            super_cluster = i % 2
            sub_cluster = (i // 2) % 2
            
            descriptor = super_centers[super_cluster].copy()
            sub_offset = np.random.randn(n_features) * 5
            descriptor += sub_offset * (1 if sub_cluster == 0 else -1)
            descriptor += np.random.randn(n_features) * 1
        else:  # Remaining queries are outliers
            descriptor = np.random.randn(n_features) * 30
        queries_descriptors.append(descriptor)
    
    queries_descriptors = np.array(queries_descriptors)
    
    # Create fake paths
    database_paths = [f"synthetic_db_{i:03d}.jpg" for i in range(n_database)]
    queries_paths = [f"synthetic_query_{i:02d}.jpg" for i in range(n_queries)]
    
    return database_descriptors, queries_descriptors, database_paths, queries_paths

def main():
    print("\n=== Testing Hierarchical Leiden Community Detection ===\n")
    
    # Create synthetic data with hierarchical structure
    print("Creating synthetic data with hierarchical cluster structure...")
    db_desc, q_desc, db_paths, q_paths = create_hierarchical_synthetic_data()
    print(f"✓ Created {len(db_desc)} database descriptors and {len(q_desc)} query descriptors")
    
    # Create output directory
    output_dir = Path("test_output_leiden")
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Created output directory: {output_dir}")
    
    # Test with 2 nearest neighbors (as requested)
    n_neighbors = 2
    print(f"\nPerforming Leiden community detection with {n_neighbors} nearest neighbors...")
    
    try:
        graph, level_data = visualizations.create_nn_graph_with_leiden(
            database_descriptors=db_desc,
            queries_descriptors=q_desc,
            database_paths=db_paths,
            queries_paths=q_paths,
            output_dir=output_dir,
            n_neighbors=n_neighbors,
            n_iterations=2
        )
        
        print(f"\n✓ Successfully completed Leiden analysis")
        print(f"  - Graph has {graph.vcount()} nodes and {graph.ecount()} edges")
        print(f"  - Analyzed {len(level_data)} resolution levels:")
        
        for level_info in level_data:
            print(f"    Resolution {level_info['resolution']:>4}: "
                  f"{level_info['n_communities']:>3} communities, "
                  f"modularity = {level_info['modularity']:.3f}")
        
        # Find best resolution (highest modularity)
        best_level = max(level_data, key=lambda x: x['modularity'])
        print(f"\n  - Best resolution: {best_level['resolution']} "
              f"(modularity = {best_level['modularity']:.3f})")
        
    except Exception as e:
        print(f"✗ Error during Leiden analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n✓ Test completed! Check the '{output_dir}' directory for results.")
    print("\nKey files to examine:")
    print("  - leiden_communities_summary.txt: Overview of all results")
    print("  - leiden_hierarchy_comparison.png: Visual comparison of all resolutions")
    print("  - leiden_resolution_analysis.png: Modularity and community count plots")
    print("  - resolution_*/: Detailed results for each resolution level")
    
    print("\nTo use with real data, run:")
    print("python main.py --database_folder <path> --queries_folder <path> --perform_leiden --nn_graph_neighbors 2")

if __name__ == "__main__":
    main() 