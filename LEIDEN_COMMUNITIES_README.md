# Hierarchical Leiden Community Detection

This feature creates a nearest neighbor graph and performs hierarchical Leiden community detection at multiple resolution levels to identify communities of visually similar images.

## Why Leiden over Louvain?

The Leiden algorithm is an improved version of the Louvain algorithm that:
- Guarantees well-connected communities (no internally disconnected communities)
- Produces higher quality partitions with better modularity
- Is more robust and consistent across runs
- Can find communities at multiple resolution scales

## Usage

Add the following flags to your command:

```bash
python main.py \
    --database_folder path/to/database \
    --queries_folder path/to/queries \
    --perform_leiden \
    --nn_graph_neighbors 2 \
    --leiden_iterations 2
```

### Parameters

- `--perform_leiden`: Enable hierarchical Leiden community detection
- `--nn_graph_neighbors`: Number of nearest neighbors to connect in the graph (default: 2)
  - Use 2-3 for a well-connected graph that captures local structure
  - Higher values create denser graphs but may lose fine-grained community structure
- `--leiden_iterations`: Number of iterations for the Leiden algorithm (default: 2)
  - More iterations can improve quality but take longer

## Resolution Levels

The algorithm automatically analyzes the graph at 6 different resolution levels:
- **0.1**: Very large communities (coarse clustering)
- **0.5**: Large communities
- **1.0**: Default resolution (balanced)
- **2.0**: Smaller communities
- **5.0**: Fine-grained communities
- **10.0**: Very small communities (fine clustering)

## Output

The visualization creates a directory `leiden_communities/` in your log directory with:

1. **Summary File** (`leiden_communities_summary.txt`):
   - Graph statistics (nodes, edges)
   - Results for each resolution level
   - Community counts and modularity scores
   - Database vs query distribution per community

2. **Resolution Directories** (`resolution_X.X_communities_Y/`):
   - One directory per resolution level
   - Contains subdirectories for each community
   - `database_paths.txt` and `query_paths.txt` for each community
   - Visual grids for small communities (≤20 images)

3. **Visualizations**:
   - `resolution_X.X_visualization.png`: t-SNE plot and community size distribution for each level
   - `leiden_hierarchy_comparison.png`: Grid showing all resolution levels side-by-side
   - `leiden_resolution_analysis.png`: Plots showing how modularity and community count change with resolution

4. **Graph Files**:
   - `leiden_graph.graphml`: Graph in GraphML format (compatible with Gephi, Cytoscape)
   - `leiden_graph.gexf`: Graph in GEXF format for further analysis

## Interpretation

### Resolution Effects
- **Low resolution (0.1-0.5)**: Few large communities, good for understanding major visual themes
- **Medium resolution (1.0-2.0)**: Balanced view, good for most applications
- **High resolution (5.0-10.0)**: Many small communities, good for finding specific visual patterns

### Community Quality Indicators
- **High modularity**: Well-separated communities (typically > 0.3 is good)
- **Mixed communities**: Communities with both database and query images indicate good retrieval potential
- **Pure communities**: Only database or only query images may indicate gaps in the dataset

### Visual Patterns
- Communities group visually similar images based on their feature descriptors
- The hierarchical nature shows how communities merge/split at different scales
- Outliers often form singleton communities at high resolutions

## Example Use Cases

1. **Dataset Understanding**: See how your images naturally cluster at different scales
2. **Retrieval Analysis**: Identify which queries have good matches (mixed communities)
3. **Quality Assessment**: Find visual gaps between database and queries
4. **Multi-scale Analysis**: Understand both coarse and fine-grained visual similarities

## Tips

- Start with `--nn_graph_neighbors 2` for a good balance
- Look at the resolution analysis plot to find the "elbow" where modularity stabilizes
- Focus on resolution levels with high modularity for the best community structure
- Use the GraphML file to visualize the graph in tools like Gephi for interactive exploration 