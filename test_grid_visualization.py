#!/usr/bin/env python3
"""Test script to demonstrate improved grid visualization for clustering methods."""

import argparse
import sys
from pathlib import Path
import numpy as np
from loguru import logger

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from visualizations import create_cluster_visualization, create_component_visualization


def test_grid_visualization():
    """Test the improved grid visualization with different cluster sizes."""
    
    # Get sample image paths
    database_paths = list(Path("toy_dataset/database").glob("*.jpg"))
    queries_paths = list(Path("toy_dataset/queries").glob("*.jpg"))
    
    if not database_paths or not queries_paths:
        logger.error("No images found in toy_dataset. Please ensure toy_dataset exists with database and queries folders.")
        return
    
    # Create output directory
    output_dir = Path("test_grid_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Found {len(database_paths)} database images and {len(queries_paths)} query images")
    
    # Test 1: Small cluster (all images)
    logger.info("Test 1: Creating visualization for small cluster with all images")
    small_cluster_dir = output_dir / "small_cluster"
    small_cluster_dir.mkdir(exist_ok=True)
    
    db_indices = list(range(min(5, len(database_paths))))
    query_indices = list(range(min(3, len(queries_paths))))
    
    create_cluster_visualization(
        database_paths, queries_paths, 
        db_indices, query_indices, 
        cluster_id=0, 
        cluster_dir=small_cluster_dir,
        max_images=None  # Show all images
    )
    logger.info(f"Small cluster visualization saved to {small_cluster_dir}")
    
    # Test 2: Medium cluster
    logger.info("Test 2: Creating visualization for medium cluster")
    medium_cluster_dir = output_dir / "medium_cluster"
    medium_cluster_dir.mkdir(exist_ok=True)
    
    db_indices = list(range(min(15, len(database_paths))))
    query_indices = list(range(min(10, len(queries_paths))))
    
    create_cluster_visualization(
        database_paths, queries_paths, 
        db_indices, query_indices, 
        cluster_id=1, 
        cluster_dir=medium_cluster_dir,
        max_images=None
    )
    logger.info(f"Medium cluster visualization saved to {medium_cluster_dir}")
    
    # Test 3: Large cluster (if enough images available)
    if len(database_paths) >= 10:
        logger.info("Test 3: Creating visualization for large cluster")
        large_cluster_dir = output_dir / "large_cluster"
        large_cluster_dir.mkdir(exist_ok=True)
        
        db_indices = list(range(len(database_paths)))
        query_indices = list(range(len(queries_paths)))
        
        create_cluster_visualization(
            database_paths, queries_paths, 
            db_indices, query_indices, 
            cluster_id=2, 
            cluster_dir=large_cluster_dir,
            max_images=None
        )
        logger.info(f"Large cluster visualization saved to {large_cluster_dir}")
    
    # Test 4: Component visualization with different sizes
    logger.info("Test 4: Creating component visualizations with different sizes")
    
    # Small component
    small_comp_dir = output_dir / "small_component"
    small_comp_dir.mkdir(exist_ok=True)
    create_component_visualization(
        database_paths[:3], queries_paths[:2],
        comp_idx=0, comp_size=5, output_dir=small_comp_dir
    )
    
    # Medium component
    medium_comp_dir = output_dir / "medium_component"
    medium_comp_dir.mkdir(exist_ok=True)
    create_component_visualization(
        database_paths[:10], queries_paths[:5],
        comp_idx=1, comp_size=15, output_dir=medium_comp_dir
    )
    
    # Large component
    if len(database_paths) >= 15:
        large_comp_dir = output_dir / "large_component"
        large_comp_dir.mkdir(exist_ok=True)
        create_component_visualization(
            database_paths, queries_paths,
            comp_idx=2, comp_size=len(database_paths) + len(queries_paths), 
            output_dir=large_comp_dir
        )
    
    logger.info(f"\nAll test visualizations have been saved to: {output_dir}")
    logger.info("The visualizations now support:")
    logger.info("  - Adaptive image sizing based on cluster size")
    logger.info("  - Adaptive grid layout (columns adjust based on number of images)")
    logger.info("  - No hard limit on cluster size for visualization")
    logger.info("  - Better handling of large grids with legends and scaling")
    logger.info("  - Optimized for both small and large clusters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test improved grid visualization for clustering")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    
    logger.remove()
    logger.add(sys.stdout, level=args.log_level)
    
    test_grid_visualization() 