# Connected Components Visualization

This feature creates a nearest neighbor graph where each point (image descriptor) is connected to its k nearest neighbors, then identifies and visualizes the connected components in this graph.

## Usage

Add the following flags to your command:

```bash
python main.py \
    --database_folder path/to/database \
    --queries_folder path/to/queries \
    --visualize_connected_components \
    --nn_graph_neighbors 1
```

### Parameters

- `--visualize_connected_components`: Enable connected components visualization
- `--nn_graph_neighbors`: Number of nearest neighbors to connect in the graph (default: 1)
  - Use 1 for single nearest neighbor (creates more, smaller components)
  - Use higher values (e.g., 3-5) for more connected graphs

## Output

The visualization creates a directory `connected_components/` in your log directory with:

1. **Summary File** (`connected_components_summary.txt`):
   - Total number of nodes and edges
   - Number of connected components
   - Size distribution of components
   - Database vs query image counts per component

2. **Component Directories** (`component_XXX_size_YY/`):
   - Each connected component gets its own directory
   - Contains `database_paths.txt` and `query_paths.txt` with image paths
   - For small components (≤20 images), includes a visual grid showing all images

3. **Visualizations**:
   - `connected_components_visualization.png`: t-SNE plot colored by component membership
   - Component size distribution histogram
   - Visual grids for small components showing actual images

4. **Graph File** (`nearest_neighbor_graph.gexf`):
   - The graph structure in GEXF format for further analysis

## Interpretation

- **Large Components**: Indicate groups of visually similar images that form chains through nearest neighbor connections
- **Small Components/Singletons**: Images that are visually distinct from others in the dataset
- **Mixed Components**: Components containing both database and query images suggest good retrieval potential
- **Query-only Components**: Queries that might not have good matches in the database

## Example Use Cases

1. **Dataset Analysis**: Understand the visual diversity and clustering structure of your image dataset
2. **Retrieval Quality**: Components mixing queries and database images indicate good retrieval possibilities
3. **Outlier Detection**: Singleton components or very small components may represent outliers
4. **Visual Similarity**: Images in the same component are connected through a chain of visual similarity

## Tips

- Start with `--nn_graph_neighbors 1` for the most granular view
- Increase neighbors for more connected components (useful for finding larger visual groups)
- The visualization automatically shows image grids for components with ≤20 images
- Database images have blue borders, query images have red borders in visualizations 