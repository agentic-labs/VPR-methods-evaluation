# Hierarchical Clustering with Average Linkage and Cosine Distance

This implementation adds hierarchical clustering capabilities to the VPR-methods-evaluation framework, using average linkage with cosine distance for clustering image descriptors.

## Features

### 1. **Average Linkage with Cosine Distance**
- Uses cosine distance (1 - cosine similarity) to measure similarity between normalized descriptors
- Average linkage ensures balanced cluster formation
- Automatically normalizes descriptors using L2 normalization

### 2. **Flexible Clustering Options**
- **Distance Threshold** (default: 0.5): Cut the dendrogram at a specific distance level
- **Number of Clusters**: Optionally specify exact number of clusters instead of threshold

### 3. **Comprehensive Visualizations**

#### Dendrogram
- Full hierarchical structure showing how images group together
- Red line indicates the cutting threshold
- Color-coded clusters below the threshold
- Helps identify natural clustering points

#### t-SNE with Clusters
- 2D visualization of descriptor space
- Points colored by cluster assignment
- Separate markers for database (circles) and query (triangles) images
- Includes mini-dendrogram for reference

#### Cluster Distribution
- Bar chart showing number of database and query images per cluster
- Helps identify cluster balance and size

#### Image Organization
- Images saved in folders by cluster
- Separate subfolders for database and query images
- Sample visualization grid for each cluster
- Text files with image paths for programmatic access

## Usage

### Basic Usage

```bash
python main.py \
    --database_folder toy_dataset/database \
    --queries_folder toy_dataset/queries \
    --method netvlad \
    --save_descriptors \
    --perform_hierarchical \
    --plot_tsne
```

### Custom Distance Threshold

```bash
python main.py \
    --database_folder toy_dataset/database \
    --queries_folder toy_dataset/queries \
    --method netvlad \
    --save_descriptors \
    --perform_hierarchical \
    --hierarchical_distance_threshold 0.3 \
    --plot_tsne
```

### Specify Number of Clusters

```bash
python main.py \
    --database_folder toy_dataset/database \
    --queries_folder toy_dataset/queries \
    --method netvlad \
    --save_descriptors \
    --perform_hierarchical \
    --hierarchical_num_clusters 10 \
    --hierarchical_distance_threshold None \
    --plot_tsne
```

## Understanding the Distance Threshold

The cosine distance ranges from 0 to 2:
- **0**: Identical vectors (perfect similarity)
- **1**: Orthogonal vectors (no similarity)
- **2**: Opposite vectors (maximum dissimilarity)

Common threshold values:
- **0.3**: Very similar images only (tight clusters)
- **0.5**: Moderately similar images (default, balanced)
- **0.7**: Loosely similar images (broad clusters)

## Output Files

After running hierarchical clustering, you'll find:

```
logs/<experiment_name>/<timestamp>/
├── hierarchical_dendrogram.png          # Full dendrogram visualization
├── tsne_hierarchical_visualization.png  # t-SNE plot with cluster colors
├── hierarchical_cluster_labels_db.npy   # Cluster assignments for database
├── hierarchical_cluster_labels_queries.npy # Cluster assignments for queries
├── hierarchical_linkage_matrix.npy      # Linkage matrix for further analysis
├── hierarchical_clusters/               # Images organized by cluster
│   ├── hierarchical_cluster_summary.txt # Statistics for each cluster
│   ├── cluster_00/
│   │   ├── database/                    # Database images in this cluster
│   │   ├── queries/                     # Query images in this cluster
│   │   ├── database_paths.txt           # List of database image paths
│   │   ├── query_paths.txt              # List of query image paths
│   │   └── cluster_00_visualization.jpg # Sample images from cluster
│   └── cluster_01/
│       └── ...
└── tsne_hierarchical_embeddings.npz     # t-SNE coordinates and labels
```

## Test Script

Run the included test script to see hierarchical clustering in action:

```bash
python test_hierarchical.py
```

This will:
1. Run clustering with default threshold (0.5)
2. Optionally run with multiple thresholds (0.3, 0.5, 0.7) for comparison

## Algorithm Details

1. **Descriptor Normalization**: All descriptors are L2-normalized to unit length
2. **Distance Computation**: Pairwise cosine distances computed as (1 - cosine_similarity)
3. **Linkage**: Average linkage used to build hierarchy
4. **Cluster Assignment**: Dendrogram cut at specified distance threshold

## Advantages over K-means

- **No need to specify K**: Distance threshold provides more natural clustering
- **Hierarchical structure**: See relationships at multiple scales
- **Deterministic**: Same input always produces same output
- **Interpretable**: Dendrogram shows why images are grouped together
- **Flexible**: Can extract clusters at any level of the hierarchy

## Tips for Best Results

1. **Examine the dendrogram** to identify natural clustering points (large jumps in distance)
2. **Start with default threshold** (0.5) and adjust based on results
3. **Use t-SNE visualization** to verify cluster quality in 2D space
4. **Check cluster sizes** to ensure balanced distribution
5. **Inspect sample images** from each cluster to verify semantic coherence 