# K-Means Clustering with t-SNE Visualization

This document describes the k-means clustering functionality that has been added to the VPR methods evaluation framework.

## Overview

The new functionality allows you to:
1. Perform k-means clustering on image descriptors using FAISS
2. Visualize clusters in t-SNE plots with color coding
3. Save images organized by their cluster assignments
4. View cluster statistics and sample images

## Usage

### Command Line Arguments

Two new arguments have been added:

- `--perform_clustering`: Enable k-means clustering
- `--num_clusters`: Number of clusters (default: 5)

### Example Command

```bash
python main.py \
    --database_folder toy_dataset/database \
    --queries_folder toy_dataset/queries \
    --method netvlad \
    --save_descriptors \
    --plot_tsne \
    --perform_clustering \
    --num_clusters 5 \
    --num_preds_to_save 5
```

### Quick Test

Run the provided test script:

```bash
python test_kmeans_tsne.py
```

## Output Files

When clustering is enabled, the following additional files are created:

### 1. Cluster Assignments
- `cluster_labels_db.npy`: Cluster labels for database images
- `cluster_labels_queries.npy`: Cluster labels for query images

### 2. t-SNE Visualization with Clusters
- `tsne_kmeans_visualization.png`: Two-panel plot showing:
  - Left: t-SNE scatter plot with points colored by cluster
  - Right: Bar chart showing cluster size distribution

### 3. Organized Images Directory (`clusters/`)
```
clusters/
├── cluster_summary.txt          # Overview of all clusters
├── cluster_00/
│   ├── database_paths.txt       # Paths to database images in this cluster
│   ├── query_paths.txt          # Paths to query images in this cluster
│   └── cluster_00_samples.jpg   # Visual samples from this cluster
├── cluster_01/
│   └── ...
└── ...
```

### 4. Cluster Summary File
Contains statistics like:
```
Cluster 0:
  Database images: 3
  Query images: 1

Cluster 1:
  Database images: 5
  Query images: 2
...
```

## Implementation Details

### K-Means Clustering
- Uses FAISS's efficient k-means implementation
- Clusters are computed on combined database and query descriptors
- 300 iterations for convergence

### Visualization Features
- Database images: Circular markers
- Query images: Triangle markers
- Colors: Automatically assigned using matplotlib's tab20 colormap
- Cluster sample images: Shows up to 10 examples per cluster

### Integration with Existing Features
- Works with all VPR methods (NetVLAD, MixVPR, etc.)
- Compatible with confidence score logging
- Can be used with or without ground truth labels

## Use Cases

1. **Dataset Analysis**: Understand the distribution of your image dataset
2. **Method Evaluation**: See how different VPR methods cluster images
3. **Outlier Detection**: Identify unusual images in small clusters
4. **Retrieval Analysis**: Check if queries and their matches fall in the same clusters

## Tips

- Start with a small number of clusters (3-10) for interpretability
- Use `--save_descriptors` to save descriptors for later analysis
- Combine with `--num_preds_to_save` to see retrieval results per cluster
- Adjust perplexity in the code if t-SNE results look too spread out or compressed 