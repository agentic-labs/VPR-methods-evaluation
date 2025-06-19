# HDBSCAN Clustering for Visual Place Recognition

This document describes the HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) clustering feature added to the VPR methods evaluation framework.

## Overview

HDBSCAN is a density-based clustering algorithm that:
- Automatically determines the number of clusters
- Identifies outliers/noise points
- Works well with clusters of varying densities
- Provides soft clustering (membership probabilities)

## Installation

First, install the required package:

```bash
pip install hdbscan
# or
uv pip install hdbscan
```

## Usage

### Basic HDBSCAN Clustering

To perform HDBSCAN clustering on your VPR descriptors:

```bash
python main.py \
    --database_folder path/to/database \
    --queries_folder path/to/queries \
    --method netvlad \
    --perform_hdbscan \
    --plot_tsne \
    --save_descriptors
```

### HDBSCAN Parameters

- `--perform_hdbscan`: Enable HDBSCAN clustering
- `--hdbscan_min_cluster_size`: Minimum size of clusters (default: 5)
  - Smaller values = more clusters, more fine-grained
  - Larger values = fewer clusters, more coarse-grained
- `--hdbscan_min_samples`: Minimum samples in neighborhood (default: 5)
  - Controls how conservative clustering is
  - Higher values = more points labeled as noise

### Example with Custom Parameters

```bash
python main.py \
    --database_folder toy_dataset/database \
    --queries_folder toy_dataset/queries \
    --method netvlad \
    --descriptors_dimension 4096 \
    --perform_hdbscan \
    --hdbscan_min_cluster_size 3 \
    --hdbscan_min_samples 2 \
    --plot_tsne \
    --save_descriptors \
    --num_preds_to_save 5
```

## Output Files

When HDBSCAN clustering is enabled, the following files are generated:

### Cluster Assignments
- `hdbscan_cluster_labels_db.npy`: Cluster labels for database images (-1 indicates noise)
- `hdbscan_cluster_labels_queries.npy`: Cluster labels for query images
- `hdbscan_probabilities.npy`: Soft cluster membership probabilities
- `hdbscan_outlier_scores.npy`: Outlier scores (higher = more outlier-like)

### Visualizations
- `tsne_hdbscan_visualization.png`: t-SNE scatter plot colored by HDBSCAN clusters
  - Left subplot: Scatter plot with cluster colors
  - Right subplot: Bar chart showing cluster distribution

### Cluster Organization
- `hdbscan_clusters/`: Directory containing images organized by cluster
  - `cluster_XX/`: One directory per cluster
  - `noise/`: Directory for outlier points
  - `hdbscan_cluster_summary.txt`: Summary statistics

## Comparison with K-Means

You can run both K-Means and HDBSCAN clustering simultaneously:

```bash
python main.py \
    --database_folder path/to/database \
    --queries_folder path/to/queries \
    --method netvlad \
    --perform_clustering \
    --num_clusters 5 \
    --perform_hdbscan \
    --hdbscan_min_cluster_size 5 \
    --plot_tsne
```

This will generate visualizations for both methods, allowing comparison.

## Understanding Results

### Cluster Labels
- Non-negative integers (0, 1, 2, ...): Cluster assignments
- -1: Noise/outlier points that don't belong to any cluster

### Interpreting t-SNE Visualizations
- **Circles**: Database images
- **Triangles**: Query images
- **Colors**: Different clusters
- **Gray points**: Noise/outliers
- **Proximity**: Similar descriptors appear closer in t-SNE space

### Quality Metrics
- **Number of clusters**: Automatically determined by HDBSCAN
- **Noise percentage**: Indicates data density/separation
- **Cluster sizes**: Distribution of points across clusters

## Tips for Parameter Tuning

1. **Start with defaults**: min_cluster_size=5, min_samples=5
2. **Too many noise points?** Decrease min_cluster_size or min_samples
3. **Too many small clusters?** Increase min_cluster_size
4. **Want more conservative clustering?** Increase min_samples

## Use Cases

1. **Data Exploration**: Understand natural groupings in your visual data
2. **Quality Assessment**: Identify outlier images or unusual scenes
3. **Dataset Analysis**: Discover visual patterns and similarities
4. **Retrieval Improvement**: Use cluster information to refine search

## Test Script

Run the included test script to see HDBSCAN in action:

```bash
python test_hdbscan.py
```

This will process the toy dataset and generate all outputs in `logs/hdbscan_test/`. 