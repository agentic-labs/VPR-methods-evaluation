import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch
from PIL import Image, ImageOps
import torchvision.transforms as tfm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import faiss
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import squareform
import networkx as nx
from collections import defaultdict
import sys

# Height and width of a single image for visualization
IMG_HW = 512
TEXT_H = 175
FONTSIZE = 50
SPACE = 50  # Space between two images


def write_labels_to_image(labels=["text1", "text2"]):
    """Creates an image with text"""
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONTSIZE)
    img = Image.new("RGB", ((IMG_HW * len(labels)) + 50 * (len(labels) - 1), TEXT_H), (1, 1, 1))
    d = ImageDraw.Draw(img)
    for i, text in enumerate(labels):
        lines = text.split('\n')
        y_offset = 1
        for line in lines:
            _, _, w, h = d.textbbox((0, 0), line, font=font)
            x_pos = (IMG_HW + SPACE) * i + IMG_HW // 2 - w // 2
            d.text((x_pos, y_offset), line, fill=(0, 0, 0), font=font)
            y_offset += h + 5  # Add some spacing between lines
    return Image.fromarray(np.array(img)[:150] * 255)  # Increased height for multi-line text


def draw_box(img, c=(0, 1, 0), thickness=20):
    """Draw a colored box around an image. Image should be a PIL.Image."""
    assert isinstance(img, Image.Image)
    img = tfm.ToTensor()(img)
    assert len(img.shape) >= 2, f"{img.shape=}"
    c = torch.tensor(c).type(torch.float).reshape(3, 1, 1)
    img[..., :thickness, :] = c
    img[..., -thickness:, :] = c
    img[..., :, -thickness:] = c
    img[..., :, :thickness] = c
    return tfm.ToPILImage()(img)


def build_prediction_image(images_paths, preds_correct, confidences=None):
    """Build a row of images, where the first is the query and the rest are predictions.
    For each image, if is_correct then draw a green/red box.
    """
    assert len(images_paths) == len(preds_correct)
    labels = ["Query"]
    for i, is_correct in enumerate(preds_correct[1:]):
        label = f"Pred{i}"
        if confidences is not None and i < len(confidences) - 1:
            # Add confidence score to label
            conf = confidences[i + 1]  # +1 because first is query
            label += f"\nConf: {conf:.3f}"
        if is_correct is not None:
            label += f" - {is_correct}"
        labels.append(label)

    images = [Image.open(path).convert("RGB") for path in images_paths]
    for img_idx, (img, is_correct) in enumerate(zip(images, preds_correct)):
        if is_correct is None:
            continue
        color = (0, 1, 0) if is_correct else (1, 0, 0)
        img = draw_box(img, color)
        images[img_idx] = img

    resized_images = [tfm.Resize(510, max_size=IMG_HW, antialias=True)(img) for img in images]
    resized_images = [ImageOps.pad(img, (IMG_HW, IMG_HW), color='white') for img in images]  # Apply padding to make them squared

    total_h = len(resized_images)*IMG_HW + max(0,len(resized_images)-1)*SPACE # 2
    concat_image = Image.new('RGB', (total_h, IMG_HW), (255, 255, 255))
    y=0
    for img in resized_images:
        concat_image.paste(img, (y, 0))
        y += IMG_HW + SPACE

    try:
        labels_image = write_labels_to_image(labels)
        # Transform the images to np arrays for concatenation
        final_image = Image.fromarray(np.concatenate((np.array(labels_image), np.array(concat_image)), axis=0))
    except OSError:  # Handle error in case of missing PIL ImageFont
        final_image = concat_image

    return final_image


def save_file_with_paths(query_path, preds_paths, positives_paths, output_path, use_labels=True, confidences=None):
    file_content = []
    file_content.append("Query path:")
    file_content.append(query_path + "\n")
    
    if confidences is not None:
        file_content.append("Confidence scores:")
        for i, conf in enumerate(confidences):
            if conf is not None:  # Skip None values (query)
                file_content.append(f"Prediction {i-1}: {conf:.4f}")
        file_content.append("")  # Empty line
    
    file_content.append("Predictions paths:")
    file_content.append("\n".join(preds_paths) + "\n")
    if use_labels:
        file_content.append("Positives paths:")
        file_content.append("\n".join(positives_paths) + "\n")
    with open(output_path, "w") as file:
        _ = file.write("\n".join(file_content))


def save_preds(predictions, eval_ds, log_dir, save_only_wrong_preds=None, use_labels=True, confidences=None):
    """For each query, save an image containing the query and its predictions,
    and a file with the paths of the query, its predictions and its positives.

    Parameters
    ----------
    predictions : np.array of shape [num_queries x num_preds_to_viz], with the preds
        for each query
    eval_ds : TestDataset
    log_dir : Path with the path to save the predictions
    save_only_wrong_preds : bool, if True save only the wrongly predicted queries,
        i.e. the ones where the first pred is uncorrect (further than 25 m)
    confidences : np.array of shape [num_queries x num_preds_to_viz], with confidence
        scores for each prediction (optional)
    """
    if use_labels:
        positives_per_query = eval_ds.get_positives()

    viz_dir = log_dir / "preds"
    viz_dir.mkdir()
    for query_index, preds in enumerate(tqdm(predictions, desc=f"Saving preds in {viz_dir}")):
        query_path = eval_ds.queries_paths[query_index]
        list_of_images_paths = [query_path]
        # List of None (query), True (correct preds) or False (wrong preds)
        preds_correct = [None]
        # Get confidence scores for this query if available
        query_confidences = [None] if confidences is None else [None]  # None for query itself
        
        for pred_index, pred in enumerate(preds):
            pred_path = eval_ds.database_paths[pred]
            list_of_images_paths.append(pred_path)
            if use_labels:
                is_correct = pred in positives_per_query[query_index]
            else:
                is_correct = None
            preds_correct.append(is_correct)
            
            if confidences is not None:
                query_confidences.append(confidences[query_index, pred_index])

        if save_only_wrong_preds and preds_correct[1]:
            continue

        prediction_image = build_prediction_image(
            list_of_images_paths, 
            preds_correct,
            query_confidences if confidences is not None else None
        )
        pred_image_path = viz_dir / f"{query_index:03d}.jpg"
        prediction_image.save(pred_image_path)

        if use_labels:
            positives_paths = [eval_ds.database_paths[idx] for idx in positives_per_query[query_index]]
        else:
            positives_paths = None
        
        save_file_with_paths(
            query_path=list_of_images_paths[0],
            preds_paths=list_of_images_paths[1:],
            positives_paths=positives_paths,
            output_path=viz_dir / f"{query_index:03d}.txt",
            use_labels=use_labels,
            confidences=query_confidences if confidences is not None else None
        )


def plot_tsne(database_descriptors, queries_descriptors, save_path, perplexity=30, n_iter=1000, random_state=42):
    """Create a t-SNE visualization of database and query descriptors.
    
    Parameters
    ----------
    database_descriptors : np.array of shape [num_database x descriptor_dim]
    queries_descriptors : np.array of shape [num_queries x descriptor_dim]
    save_path : Path or str, where to save the plot
    perplexity : float, t-SNE perplexity parameter
    n_iter : int, number of iterations for t-SNE
    random_state : int, random seed for reproducibility
    """
    # Combine all descriptors
    all_descriptors = np.vstack([database_descriptors, queries_descriptors])
    
    # Create labels (0 for database, 1 for queries)
    labels = np.concatenate([
        np.zeros(len(database_descriptors)),
        np.ones(len(queries_descriptors))
    ])
    
    print(f"Running t-SNE on {len(all_descriptors)} descriptors...")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                random_state=random_state, verbose=1)
    embeddings = tsne.fit_transform(all_descriptors)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot database points
    db_embeddings = embeddings[labels == 0]
    plt.scatter(db_embeddings[:, 0], db_embeddings[:, 1], 
                c='blue', alpha=0.6, s=50, label='Database', edgecolors='none')
    
    # Plot query points
    query_embeddings = embeddings[labels == 1]
    plt.scatter(query_embeddings[:, 0], query_embeddings[:, 1], 
                c='red', alpha=0.8, s=100, label='Queries', edgecolors='black', linewidths=1)
    
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.title('t-SNE Visualization of Image Descriptors', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE plot saved to {save_path}")
    
    # Also save the embeddings for potential further analysis
    embeddings_path = Path(save_path).parent / "tsne_embeddings.npz"
    np.savez(embeddings_path, 
             embeddings=embeddings, 
             labels=labels,
             db_embeddings=db_embeddings,
             query_embeddings=query_embeddings)
    print(f"t-SNE embeddings saved to {embeddings_path}")


def plot_tsne_with_connections(database_descriptors, queries_descriptors, predictions, 
                               save_path, num_connections=3, perplexity=30, n_iter=1000, 
                               random_state=42):
    """Create an enhanced t-SNE visualization showing connections between queries and predictions.
    
    Parameters
    ----------
    database_descriptors : np.array of shape [num_database x descriptor_dim]
    queries_descriptors : np.array of shape [num_queries x descriptor_dim]
    predictions : np.array of shape [num_queries x num_predictions]
    save_path : Path or str, where to save the plot
    num_connections : int, number of top predictions to show connections for
    perplexity : float, t-SNE perplexity parameter
    n_iter : int, number of iterations for t-SNE
    random_state : int, random seed for reproducibility
    """
    # Combine all descriptors
    all_descriptors = np.vstack([database_descriptors, queries_descriptors])
    
    # Create labels (0 for database, 1 for queries)
    labels = np.concatenate([
        np.zeros(len(database_descriptors)),
        np.ones(len(queries_descriptors))
    ])
    
    print(f"Running t-SNE on {len(all_descriptors)} descriptors...")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                random_state=random_state, verbose=1)
    embeddings = tsne.fit_transform(all_descriptors)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # First subplot: Basic scatter plot
    db_embeddings = embeddings[labels == 0]
    query_embeddings = embeddings[labels == 1]
    
    ax1.scatter(db_embeddings[:, 0], db_embeddings[:, 1], 
                c='blue', alpha=0.6, s=50, label='Database', edgecolors='none')
    ax1.scatter(query_embeddings[:, 0], query_embeddings[:, 1], 
                c='red', alpha=0.8, s=100, label='Queries', edgecolors='black', linewidths=1)
    
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.set_title('t-SNE Visualization of Image Descriptors', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Second subplot: With connections
    ax2.scatter(db_embeddings[:, 0], db_embeddings[:, 1], 
                c='blue', alpha=0.3, s=30, label='Database', edgecolors='none')
    ax2.scatter(query_embeddings[:, 0], query_embeddings[:, 1], 
                c='red', alpha=0.8, s=100, label='Queries', edgecolors='black', linewidths=1)
    
    # Draw connections between queries and their top predictions
    for query_idx in range(len(queries_descriptors)):
        query_embedding = query_embeddings[query_idx]
        
        # Get top predictions for this query
        top_preds = predictions[query_idx, :num_connections]
        
        for i, pred_idx in enumerate(top_preds):
            pred_embedding = db_embeddings[pred_idx]
            
            # Draw line with decreasing alpha for lower-ranked predictions
            alpha = 0.6 * (1 - i / num_connections)
            ax2.plot([query_embedding[0], pred_embedding[0]], 
                    [query_embedding[1], pred_embedding[1]], 
                    'gray', alpha=alpha, linewidth=1)
    
    ax2.set_xlabel('t-SNE Component 1', fontsize=12)
    ax2.set_ylabel('t-SNE Component 2', fontsize=12)
    ax2.set_title(f'Query-Prediction Connections (Top {num_connections})', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced t-SNE plot saved to {save_path}")
    
    # Save embeddings
    embeddings_path = Path(save_path).parent / "tsne_embeddings_enhanced.npz"
    np.savez(embeddings_path, 
             embeddings=embeddings, 
             labels=labels,
             db_embeddings=db_embeddings,
             query_embeddings=query_embeddings,
             predictions=predictions)
    print(f"t-SNE embeddings saved to {embeddings_path}")


def plot_tsne_with_kmeans(database_descriptors, queries_descriptors, cluster_labels_db, 
                          cluster_labels_queries, save_path, num_clusters, 
                          perplexity=30, n_iter=1000, random_state=42):
    """Create a t-SNE visualization with k-means clustering results.
    
    Parameters
    ----------
    database_descriptors : np.array of shape [num_database x descriptor_dim]
    queries_descriptors : np.array of shape [num_queries x descriptor_dim]
    cluster_labels_db : np.array of shape [num_database], cluster assignments for database
    cluster_labels_queries : np.array of shape [num_queries], cluster assignments for queries
    save_path : Path or str, where to save the plot
    num_clusters : int, number of clusters
    perplexity : float, t-SNE perplexity parameter
    n_iter : int, number of iterations for t-SNE
    random_state : int, random seed for reproducibility
    """
    # Combine all descriptors
    all_descriptors = np.vstack([database_descriptors, queries_descriptors])
    
    # Combine cluster labels
    all_cluster_labels = np.concatenate([cluster_labels_db, cluster_labels_queries])
    
    # Create labels (0 for database, 1 for queries)
    data_type_labels = np.concatenate([
        np.zeros(len(database_descriptors)),
        np.ones(len(queries_descriptors))
    ])
    
    print(f"Running t-SNE on {len(all_descriptors)} descriptors...")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                random_state=random_state, verbose=1)
    embeddings = tsne.fit_transform(all_descriptors)
    
    # Create the plot with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Get colors for clusters
    colors = cm.tab20(np.linspace(0, 1, num_clusters))
    
    # First subplot: Color by cluster
    for cluster_id in range(num_clusters):
        # Database points in this cluster
        mask_db = (data_type_labels == 0) & (all_cluster_labels == cluster_id)
        if np.any(mask_db):
            ax1.scatter(embeddings[mask_db, 0], embeddings[mask_db, 1], 
                       c=[colors[cluster_id]], alpha=0.6, s=50, 
                       label=f'DB Cluster {cluster_id}', edgecolors='none')
        
        # Query points in this cluster
        mask_query = (data_type_labels == 1) & (all_cluster_labels == cluster_id)
        if np.any(mask_query):
            ax1.scatter(embeddings[mask_query, 0], embeddings[mask_query, 1], 
                       c=[colors[cluster_id]], alpha=0.8, s=100, 
                       marker='^', label=f'Query Cluster {cluster_id}', 
                       edgecolors='black', linewidths=1)
    
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.set_title(f't-SNE Visualization with K-Means Clustering (K={num_clusters})', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Second subplot: Show cluster distribution
    cluster_counts_db = np.bincount(cluster_labels_db, minlength=num_clusters)
    cluster_counts_queries = np.bincount(cluster_labels_queries, minlength=num_clusters)
    
    x = np.arange(num_clusters)
    width = 0.35
    
    ax2.bar(x - width/2, cluster_counts_db, width, label='Database', alpha=0.7)
    ax2.bar(x + width/2, cluster_counts_queries, width, label='Queries', alpha=0.7)
    
    ax2.set_xlabel('Cluster ID', fontsize=12)
    ax2.set_ylabel('Number of Images', fontsize=12)
    ax2.set_title('Distribution of Images across Clusters', fontsize=14)
    ax2.set_xticks(x)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE with k-means plot saved to {save_path}")
    
    # Save embeddings and cluster assignments
    embeddings_path = Path(save_path).parent / "tsne_kmeans_embeddings.npz"
    np.savez(embeddings_path, 
             embeddings=embeddings, 
             data_type_labels=data_type_labels,
             cluster_labels=all_cluster_labels,
             cluster_labels_db=cluster_labels_db,
             cluster_labels_queries=cluster_labels_queries)
    print(f"t-SNE embeddings and cluster labels saved to {embeddings_path}")


def save_images_by_cluster(database_paths, queries_paths, cluster_labels_db, 
                          cluster_labels_queries, num_clusters, output_dir):
    """Save images organized by their cluster assignments.
    
    Parameters
    ----------
    database_paths : list of paths to database images
    queries_paths : list of paths to query images
    cluster_labels_db : np.array of cluster assignments for database images
    cluster_labels_queries : np.array of cluster assignments for query images
    num_clusters : int, number of clusters
    output_dir : Path, directory to save the cluster directories
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create a summary file
    summary_file = output_dir / "cluster_summary.txt"
    summary_lines = []
    
    for cluster_id in range(num_clusters):
        cluster_dir = output_dir / f"cluster_{cluster_id:02d}"
        cluster_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for database and queries
        db_dir = cluster_dir / "database"
        query_dir = cluster_dir / "queries"
        db_dir.mkdir(exist_ok=True)
        query_dir.mkdir(exist_ok=True)
        
        # Get indices for this cluster
        db_indices = np.where(cluster_labels_db == cluster_id)[0]
        query_indices = np.where(cluster_labels_queries == cluster_id)[0]
        
        summary_lines.append(f"Cluster {cluster_id}:")
        summary_lines.append(f"  Database images: {len(db_indices)}")
        summary_lines.append(f"  Query images: {len(query_indices)}")
        summary_lines.append("")
        
        # Save database image paths
        db_paths_file = cluster_dir / "database_paths.txt"
        with open(db_paths_file, 'w') as f:
            for idx in db_indices:
                f.write(f"{database_paths[idx]}\n")
        
        # Save query image paths
        query_paths_file = cluster_dir / "query_paths.txt"
        with open(query_paths_file, 'w') as f:
            for idx in query_indices:
                f.write(f"{queries_paths[idx]}\n")
        
        # Create visualization of sample images from this cluster
        create_cluster_visualization(database_paths, queries_paths, db_indices, 
                                   query_indices, cluster_id, cluster_dir)
    
    # Write summary file
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"Images organized by cluster saved to {output_dir}")
    print(f"Cluster summary saved to {summary_file}")


def create_cluster_visualization(database_paths, queries_paths, db_indices, 
                                query_indices, cluster_id, cluster_dir, max_images=10):
    """Create a visualization showing sample images from a cluster.
    
    Parameters
    ----------
    database_paths : list of all database image paths
    queries_paths : list of all query image paths
    db_indices : indices of database images in this cluster
    query_indices : indices of query images in this cluster
    cluster_id : int, cluster identifier
    cluster_dir : Path, directory for this cluster
    max_images : int, maximum number of images to show per category
    """
    # Limit the number of images to visualize
    db_sample = db_indices[:max_images]
    query_sample = query_indices[:max_images]
    
    # Calculate grid dimensions
    n_db = len(db_sample)
    n_query = len(query_sample)
    
    if n_db == 0 and n_query == 0:
        return
    
    # Create the visualization
    fig_width = 15
    fig_height = 6
    fig, axes = plt.subplots(2, max(n_db, n_query), figsize=(fig_width, fig_height))
    
    # Handle case where axes is 1D
    if max(n_db, n_query) == 1:
        axes = axes.reshape(-1, 1)
    
    # Display database images
    for i in range(max(n_db, n_query)):
        # Database row
        if i < n_db:
            img_path = database_paths[db_sample[i]]
            img = Image.open(img_path).convert('RGB')
            # Use Resize with only size parameter to avoid the max_size conflict
            img_resized = tfm.Resize((256, 256), antialias=True)(img)
            axes[0, i].imshow(img_resized)
            axes[0, i].set_title(f'DB {db_sample[i]}', fontsize=8)
        else:
            axes[0, i].axis('off')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        # Query row
        if i < n_query:
            img_path = queries_paths[query_sample[i]]
            img = Image.open(img_path).convert('RGB')
            # Use Resize with only size parameter to avoid the max_size conflict
            img_resized = tfm.Resize((256, 256), antialias=True)(img)
            axes[1, i].imshow(img_resized)
            axes[1, i].set_title(f'Query {query_sample[i]}', fontsize=8)
        else:
            axes[1, i].axis('off')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    
    # Add row labels
    if n_db > 0:
        axes[0, 0].set_ylabel('Database', fontsize=10, rotation=90, va='center')
    if n_query > 0:
        axes[1, 0].set_ylabel('Queries', fontsize=10, rotation=90, va='center')
    
    plt.suptitle(f'Cluster {cluster_id} Sample Images\n(DB: {len(db_indices)} images, Queries: {len(query_indices)} images)', 
                 fontsize=12)
    plt.tight_layout()
    
    # Save the visualization
    viz_path = cluster_dir / f'cluster_{cluster_id}_samples.jpg'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_tsne_with_hdbscan(database_descriptors, queries_descriptors, cluster_labels_db, 
                           cluster_labels_queries, save_path, 
                           perplexity=30, n_iter=1000, random_state=42):
    """Create a t-SNE visualization with HDBSCAN clustering results.
    
    Parameters
    ----------
    database_descriptors : np.array of shape [num_database x descriptor_dim]
    queries_descriptors : np.array of shape [num_queries x descriptor_dim]
    cluster_labels_db : np.array of shape [num_database], cluster assignments for database
    cluster_labels_queries : np.array of shape [num_queries], cluster assignments for queries
    save_path : Path or str, where to save the plot
    perplexity : float, t-SNE perplexity parameter
    n_iter : int, number of iterations for t-SNE
    random_state : int, random seed for reproducibility
    """
    # Combine all descriptors
    all_descriptors = np.vstack([database_descriptors, queries_descriptors])
    
    # Combine cluster labels
    all_cluster_labels = np.concatenate([cluster_labels_db, cluster_labels_queries])
    
    # Create labels (0 for database, 1 for queries)
    data_type_labels = np.concatenate([
        np.zeros(len(database_descriptors)),
        np.ones(len(queries_descriptors))
    ])
    
    print(f"Running t-SNE on {len(all_descriptors)} descriptors...")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                random_state=random_state, verbose=1)
    embeddings = tsne.fit_transform(all_descriptors)
    
    # Create the plot with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Get unique cluster labels (excluding noise points -1)
    unique_labels = np.unique(all_cluster_labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    
    # Get colors for clusters
    colors = cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    
    # First subplot: Color by cluster
    for i, cluster_id in enumerate(unique_labels):
        if cluster_id == -1:
            # Noise points in gray
            color = 'gray'
            label_prefix = 'Noise'
            alpha_db = 0.3
            alpha_query = 0.5
        else:
            color = colors[i % len(colors)]
            label_prefix = f'Cluster {cluster_id}'
            alpha_db = 0.6
            alpha_query = 0.8
        
        # Database points in this cluster
        mask_db = (data_type_labels == 0) & (all_cluster_labels == cluster_id)
        if np.any(mask_db):
            ax1.scatter(embeddings[mask_db, 0], embeddings[mask_db, 1], 
                       c=[color], alpha=alpha_db, s=50, 
                       label=f'DB {label_prefix}', edgecolors='none')
        
        # Query points in this cluster
        mask_query = (data_type_labels == 1) & (all_cluster_labels == cluster_id)
        if np.any(mask_query):
            ax1.scatter(embeddings[mask_query, 0], embeddings[mask_query, 1], 
                       c=[color], alpha=alpha_query, s=100, 
                       marker='^', label=f'Query {label_prefix}', 
                       edgecolors='black', linewidths=1)
    
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.set_title(f't-SNE Visualization with HDBSCAN Clustering ({n_clusters} clusters found)', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Second subplot: Show cluster distribution
    unique_labels_no_noise = unique_labels[unique_labels != -1]
    
    # Count points in each cluster (including noise)
    cluster_counts_db = []
    cluster_counts_queries = []
    cluster_labels_for_plot = []
    
    for cluster_id in unique_labels:
        db_count = np.sum((cluster_labels_db == cluster_id))
        query_count = np.sum((cluster_labels_queries == cluster_id))
        if db_count > 0 or query_count > 0:
            cluster_counts_db.append(db_count)
            cluster_counts_queries.append(query_count)
            cluster_labels_for_plot.append(f'Noise' if cluster_id == -1 else f'C{cluster_id}')
    
    x = np.arange(len(cluster_labels_for_plot))
    width = 0.35
    
    ax2.bar(x - width/2, cluster_counts_db, width, label='Database', alpha=0.7)
    ax2.bar(x + width/2, cluster_counts_queries, width, label='Queries', alpha=0.7)
    
    ax2.set_xlabel('Cluster', fontsize=12)
    ax2.set_ylabel('Number of Images', fontsize=12)
    ax2.set_title('Distribution of Images across HDBSCAN Clusters', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cluster_labels_for_plot, rotation=45 if len(cluster_labels_for_plot) > 10 else 0)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE with HDBSCAN plot saved to {save_path}")
    
    # Save embeddings and cluster assignments
    embeddings_path = Path(save_path).parent / "tsne_hdbscan_embeddings.npz"
    np.savez(embeddings_path, 
             embeddings=embeddings, 
             data_type_labels=data_type_labels,
             cluster_labels=all_cluster_labels,
             cluster_labels_db=cluster_labels_db,
             cluster_labels_queries=cluster_labels_queries)
    print(f"t-SNE embeddings and cluster labels saved to {embeddings_path}")


def save_hdbscan_images_by_cluster(database_paths, queries_paths, cluster_labels_db, 
                                   cluster_labels_queries, output_dir):
    """Save images organized by their HDBSCAN cluster assignments.
    
    Parameters
    ----------
    database_paths : list of paths to database images
    queries_paths : list of paths to query images
    cluster_labels_db : np.array of cluster assignments for database images
    cluster_labels_queries : np.array of cluster assignments for query images
    output_dir : Path, directory to save the cluster directories
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get unique cluster labels
    all_labels = np.concatenate([cluster_labels_db, cluster_labels_queries])
    unique_labels = np.unique(all_labels)
    
    # Create a summary file
    summary_file = output_dir / "hdbscan_cluster_summary.txt"
    summary_lines = []
    summary_lines.append(f"HDBSCAN Clustering Results")
    summary_lines.append(f"Total clusters found: {len(unique_labels[unique_labels != -1])}")
    summary_lines.append("")
    
    for cluster_id in unique_labels:
        if cluster_id == -1:
            cluster_name = "noise"
        else:
            cluster_name = f"cluster_{cluster_id:02d}"
            
        cluster_dir = output_dir / cluster_name
        cluster_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for database and queries
        db_dir = cluster_dir / "database"
        query_dir = cluster_dir / "queries"
        db_dir.mkdir(exist_ok=True)
        query_dir.mkdir(exist_ok=True)
        
        # Get indices for this cluster
        db_indices = np.where(cluster_labels_db == cluster_id)[0]
        query_indices = np.where(cluster_labels_queries == cluster_id)[0]
        
        if cluster_id == -1:
            summary_lines.append(f"Noise points:")
        else:
            summary_lines.append(f"Cluster {cluster_id}:")
        summary_lines.append(f"  Database images: {len(db_indices)}")
        summary_lines.append(f"  Query images: {len(query_indices)}")
        summary_lines.append("")
        
        # Save database image paths
        db_paths_file = cluster_dir / "database_paths.txt"
        with open(db_paths_file, 'w') as f:
            for idx in db_indices:
                f.write(f"{database_paths[idx]}\n")
        
        # Save query image paths
        query_paths_file = cluster_dir / "query_paths.txt"
        with open(query_paths_file, 'w') as f:
            for idx in query_indices:
                f.write(f"{queries_paths[idx]}\n")
        
        # Create visualization of sample images from this cluster
        create_cluster_visualization(database_paths, queries_paths, db_indices, 
                                   query_indices, cluster_id, cluster_dir)
    
    # Write summary file
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"Images organized by HDBSCAN clusters saved to {output_dir}")
    print(f"Cluster summary saved to {summary_file}")


def plot_hierarchical_dendrogram(linkage_matrix, save_path, distance_threshold=0.5, 
                                 labels=None, title_suffix=""):
    """Create a dendrogram visualization for hierarchical clustering.
    
    Parameters
    ----------
    linkage_matrix : np.array, linkage matrix from scipy hierarchical clustering
    save_path : Path or str, where to save the plot
    distance_threshold : float, distance threshold for cutting the dendrogram
    labels : list, optional labels for leaves
    title_suffix : str, additional text for the title
    """
    plt.figure(figsize=(20, 10))
    
    # Create dendrogram
    dendrogram_result = dendrogram(
        linkage_matrix,
        labels=labels,
        color_threshold=distance_threshold,
        above_threshold_color='gray',
        leaf_rotation=90,
        leaf_font_size=8
    )
    
    # Add threshold line
    plt.axhline(y=distance_threshold, c='red', linestyle='--', 
                label=f'Distance threshold = {distance_threshold}')
    
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Cosine Distance', fontsize=12)
    plt.title(f'Hierarchical Clustering Dendrogram{title_suffix}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dendrogram saved to {save_path}")
    return dendrogram_result


def plot_tsne_with_hierarchical(database_descriptors, queries_descriptors, 
                                cluster_labels_db, cluster_labels_queries, 
                                save_path, linkage_matrix=None, distance_threshold=0.5,
                                perplexity=30, n_iter=1000, random_state=42):
    """Create a t-SNE visualization with hierarchical clustering results.
    
    Parameters
    ----------
    database_descriptors : np.array of shape [num_database x descriptor_dim]
    queries_descriptors : np.array of shape [num_queries x descriptor_dim]
    cluster_labels_db : np.array of shape [num_database], cluster assignments for database
    cluster_labels_queries : np.array of shape [num_queries], cluster assignments for queries
    save_path : Path or str, where to save the plot
    linkage_matrix : np.array, optional linkage matrix for showing dendrogram
    distance_threshold : float, distance threshold used for clustering
    perplexity : float, t-SNE perplexity parameter
    n_iter : int, number of iterations for t-SNE
    random_state : int, random seed for reproducibility
    """
    # Combine all descriptors
    all_descriptors = np.vstack([database_descriptors, queries_descriptors])
    
    # Combine cluster labels
    all_cluster_labels = np.concatenate([cluster_labels_db, cluster_labels_queries])
    
    # Create labels (0 for database, 1 for queries)
    data_type_labels = np.concatenate([
        np.zeros(len(database_descriptors)),
        np.ones(len(queries_descriptors))
    ])
    
    print(f"Running t-SNE on {len(all_descriptors)} descriptors...")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                random_state=random_state, verbose=1)
    embeddings = tsne.fit_transform(all_descriptors)
    
    # Determine number of clusters
    unique_clusters = np.unique(all_cluster_labels)
    n_clusters = len(unique_clusters)
    
    # Create figure with subplots
    if linkage_matrix is not None:
        fig = plt.figure(figsize=(36, 10))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Get colors for clusters
    if n_clusters <= 20:
        colors = cm.tab20(np.linspace(0, 1, n_clusters))
    else:
        colors = cm.gist_rainbow(np.linspace(0, 1, n_clusters))
    
    # First subplot: Color by cluster
    for i, cluster_id in enumerate(unique_clusters):
        # Database points in this cluster
        mask_db = (data_type_labels == 0) & (all_cluster_labels == cluster_id)
        if np.any(mask_db):
            ax1.scatter(embeddings[mask_db, 0], embeddings[mask_db, 1], 
                       c=[colors[i]], alpha=0.6, s=50, 
                       label=f'DB Cluster {cluster_id}', edgecolors='none')
        
        # Query points in this cluster
        mask_query = (data_type_labels == 1) & (all_cluster_labels == cluster_id)
        if np.any(mask_query):
            ax1.scatter(embeddings[mask_query, 0], embeddings[mask_query, 1], 
                       c=[colors[i]], alpha=0.8, s=100, 
                       marker='^', label=f'Query Cluster {cluster_id}', 
                       edgecolors='black', linewidths=1)
    
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.set_title(f't-SNE with Hierarchical Clustering\n(threshold={distance_threshold}, {n_clusters} clusters)', 
                  fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Second subplot: Show cluster distribution
    cluster_counts_db = np.bincount(cluster_labels_db, minlength=n_clusters)
    cluster_counts_queries = np.bincount(cluster_labels_queries, minlength=n_clusters)
    
    x = np.arange(n_clusters)
    width = 0.35
    
    ax2.bar(x - width/2, cluster_counts_db, width, label='Database', alpha=0.7)
    ax2.bar(x + width/2, cluster_counts_queries, width, label='Queries', alpha=0.7)
    
    ax2.set_xlabel('Cluster ID', fontsize=12)
    ax2.set_ylabel('Number of Images', fontsize=12)
    ax2.set_title('Distribution of Images across Clusters', fontsize=14)
    ax2.set_xticks(x)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Third subplot: Mini dendrogram if linkage matrix provided
    if linkage_matrix is not None:
        dendrogram(
            linkage_matrix,
            ax=ax3,
            color_threshold=distance_threshold,
            above_threshold_color='gray',
            no_labels=True
        )
        ax3.axhline(y=distance_threshold, c='red', linestyle='--', 
                    label=f'Threshold = {distance_threshold}')
        ax3.set_xlabel('Sample Index', fontsize=10)
        ax3.set_ylabel('Cosine Distance', fontsize=10)
        ax3.set_title('Hierarchical Clustering Dendrogram', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE with hierarchical clustering plot saved to {save_path}")
    
    # Save embeddings and cluster assignments
    embeddings_path = Path(save_path).parent / "tsne_hierarchical_embeddings.npz"
    np.savez(embeddings_path, 
             embeddings=embeddings, 
             data_type_labels=data_type_labels,
             cluster_labels=all_cluster_labels,
             cluster_labels_db=cluster_labels_db,
             cluster_labels_queries=cluster_labels_queries)
    print(f"t-SNE embeddings and cluster labels saved to {embeddings_path}")


def save_hierarchical_images_by_cluster(database_paths, queries_paths, cluster_labels_db, 
                                        cluster_labels_queries, output_dir, 
                                        distance_threshold=0.5):
    """Save images organized by their hierarchical cluster assignments.
    
    Parameters
    ----------
    database_paths : list of paths to database images
    queries_paths : list of paths to query images
    cluster_labels_db : np.array of cluster assignments for database images
    cluster_labels_queries : np.array of cluster assignments for query images
    output_dir : Path, directory to save the cluster directories
    distance_threshold : float, distance threshold used for clustering
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get unique cluster IDs
    unique_clusters = np.unique(np.concatenate([cluster_labels_db, cluster_labels_queries]))
    n_clusters = len(unique_clusters)
    
    # Create a summary file
    summary_file = output_dir / "hierarchical_cluster_summary.txt"
    summary_lines = [
        f"Hierarchical Clustering Results",
        f"Distance threshold: {distance_threshold}",
        f"Number of clusters: {n_clusters}",
        f"Clustering method: Average Linkage with Cosine Distance",
        "",
    ]
    
    for cluster_id in unique_clusters:
        cluster_dir = output_dir / f"cluster_{cluster_id:02d}"
        cluster_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for database and queries
        db_dir = cluster_dir / "database"
        query_dir = cluster_dir / "queries"
        db_dir.mkdir(exist_ok=True)
        query_dir.mkdir(exist_ok=True)
        
        # Get indices for this cluster
        db_indices = np.where(cluster_labels_db == cluster_id)[0]
        query_indices = np.where(cluster_labels_queries == cluster_id)[0]
        
        summary_lines.append(f"Cluster {cluster_id}:")
        summary_lines.append(f"  Database images: {len(db_indices)}")
        summary_lines.append(f"  Query images: {len(query_indices)}")
        summary_lines.append("")
        
        # Save database image paths
        db_paths_file = cluster_dir / "database_paths.txt"
        with open(db_paths_file, 'w') as f:
            for idx in db_indices:
                f.write(f"{database_paths[idx]}\n")
        
        # Save query image paths
        query_paths_file = cluster_dir / "query_paths.txt"
        with open(query_paths_file, 'w') as f:
            for idx in query_indices:
                f.write(f"{queries_paths[idx]}\n")
        
        # Create visualization of sample images from this cluster
        create_cluster_visualization(database_paths, queries_paths, db_indices, 
                                   query_indices, cluster_id, cluster_dir)
    
    # Write summary file
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"Images organized by hierarchical clusters saved to {output_dir}")
    print(f"Cluster summary saved to {summary_file}")


def plot_tsne_with_all_nn_connections(database_descriptors, queries_descriptors, save_path, 
                                      n_neighbors=2, perplexity=30, n_iter=1000, random_state=42):
    """Create a t-SNE visualization showing nearest neighbor connections for all points.
    
    Parameters
    ----------
    database_descriptors : np.array of shape [num_database x descriptor_dim]
    queries_descriptors : np.array of shape [num_queries x descriptor_dim]
    save_path : Path or str, where to save the plot
    n_neighbors : int, number of nearest neighbors to connect (default: 2)
    perplexity : float, t-SNE perplexity parameter
    n_iter : int, number of iterations for t-SNE
    random_state : int, random seed for reproducibility
    """
    # Combine all descriptors
    all_descriptors = np.vstack([database_descriptors, queries_descriptors])
    
    # Create labels (0 for database, 1 for queries)
    labels = np.concatenate([
        np.zeros(len(database_descriptors)),
        np.ones(len(queries_descriptors))
    ])
    
    print(f"Running t-SNE on {len(all_descriptors)} descriptors...")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                random_state=random_state, verbose=1)
    embeddings = tsne.fit_transform(all_descriptors)
    
    print(f"Finding {n_neighbors} nearest neighbors in descriptor space...")
    
    # Find nearest neighbors in the original descriptor space
    # Using FAISS for efficient nearest neighbor search
    faiss_index = faiss.IndexFlatL2(all_descriptors.shape[1])
    faiss_index.add(all_descriptors.astype(np.float32))
    
    # Search for n_neighbors + 1 nearest neighbors (first one is the point itself)
    distances, neighbors = faiss_index.search(all_descriptors.astype(np.float32), n_neighbors + 1)
    nearest_neighbors = neighbors[:, 1:n_neighbors+1]  # Skip the first column (self)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # First subplot: Basic scatter plot
    db_embeddings = embeddings[labels == 0]
    query_embeddings = embeddings[labels == 1]
    
    ax1.scatter(db_embeddings[:, 0], db_embeddings[:, 1], 
                c='blue', alpha=0.6, s=50, label='Database', edgecolors='none')
    ax1.scatter(query_embeddings[:, 0], query_embeddings[:, 1], 
                c='red', alpha=0.8, s=100, label='Queries', edgecolors='black', linewidths=1)
    
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.set_title('t-SNE Visualization of Image Descriptors', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Second subplot: With nearest neighbor connections
    ax2.scatter(db_embeddings[:, 0], db_embeddings[:, 1], 
                c='blue', alpha=0.4, s=30, label='Database', edgecolors='none')
    ax2.scatter(query_embeddings[:, 0], query_embeddings[:, 1], 
                c='red', alpha=0.6, s=80, label='Queries', edgecolors='black', linewidths=1)
    
    # Draw connections to nearest neighbors
    print(f"Drawing connections to {n_neighbors} nearest neighbors...")
    for i in range(len(embeddings)):
        for neighbor_rank in range(n_neighbors):
            nn_idx = nearest_neighbors[i, neighbor_rank]
            
            # Determine color based on point types
            if labels[i] == 0 and labels[nn_idx] == 0:  # DB to DB
                color = 'blue'
                alpha = 0.2 * (1 - neighbor_rank / n_neighbors)  # Fade for further neighbors
            elif labels[i] == 1 and labels[nn_idx] == 1:  # Query to Query
                color = 'red'
                alpha = 0.3 * (1 - neighbor_rank / n_neighbors)
            else:  # Cross-type connection (DB to Query or Query to DB)
                color = 'green'
                alpha = 0.4 * (1 - neighbor_rank / n_neighbors)
            
            # Make lines thinner for further neighbors
            linewidth = 0.5 * (1 - neighbor_rank / (n_neighbors + 1))
            
            ax2.plot([embeddings[i, 0], embeddings[nn_idx, 0]], 
                    [embeddings[i, 1], embeddings[nn_idx, 1]], 
                    color=color, alpha=alpha, linewidth=linewidth)
    
    ax2.set_xlabel('t-SNE Component 1', fontsize=12)
    ax2.set_ylabel('t-SNE Component 2', fontsize=12)
    ax2.set_title(f't-SNE with {n_neighbors} Nearest Neighbor Connections', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add a custom legend for connection types
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='blue', alpha=0.5, lw=2),
        Line2D([0], [0], color='red', alpha=0.5, lw=2),
        Line2D([0], [0], color='green', alpha=0.5, lw=2)
    ]
    ax2.legend(custom_lines, ['DB→DB', 'Query→Query', 'Cross-type'], 
               loc='upper right', fontsize=10, title='Connection Types')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE with {n_neighbors} nearest neighbor connections saved to {save_path}")
    
    # Save embeddings and nearest neighbor information
    embeddings_path = Path(save_path).parent / "tsne_nn_embeddings.npz"
    np.savez(embeddings_path, 
             embeddings=embeddings, 
             labels=labels,
             db_embeddings=db_embeddings,
             query_embeddings=query_embeddings,
             nearest_neighbors=nearest_neighbors,
             distances=distances[:, 1:n_neighbors+1])  # Save distances to nearest neighbors
    print(f"t-SNE embeddings and NN info saved to {embeddings_path}")


def create_nn_graph_and_visualize_components(database_descriptors, queries_descriptors, 
                                            database_paths, queries_paths, output_dir,
                                            n_neighbors=1, perplexity=30, n_iter=1000, 
                                            random_state=42):
    """Create a nearest neighbor graph and visualize photos in each connected component.
    
    Parameters
    ----------
    database_descriptors : np.array of shape [num_database x descriptor_dim]
    queries_descriptors : np.array of shape [num_queries x descriptor_dim]
    database_paths : list of paths to database images
    queries_paths : list of paths to query images
    output_dir : Path, directory to save the component visualizations
    n_neighbors : int, number of nearest neighbors to connect (default=1 for single NN)
    perplexity : float, t-SNE perplexity parameter
    n_iter : int, number of iterations for t-SNE
    random_state : int, random seed for reproducibility
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Combine all descriptors and paths
    all_descriptors = np.vstack([database_descriptors, queries_descriptors])
    all_paths = database_paths + queries_paths
    num_database = len(database_descriptors)
    
    # Create labels (0 for database, 1 for queries)
    data_type_labels = np.concatenate([
        np.zeros(len(database_descriptors)),
        np.ones(len(queries_descriptors))
    ])
    
    print(f"Building nearest neighbor index for {len(all_descriptors)} descriptors...")
    
    # Build FAISS index for efficient NN search
    dimension = all_descriptors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(all_descriptors.astype(np.float32))
    
    # Find k+1 nearest neighbors (including self)
    distances, neighbors = index.search(all_descriptors.astype(np.float32), n_neighbors + 1)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(len(all_descriptors)):
        node_type = 'database' if i < num_database else 'query'
        G.add_node(i, type=node_type, path=all_paths[i])
    
    # Add edges (skip self-connections)
    edge_count = 0
    for i in range(len(all_descriptors)):
        for j in range(1, n_neighbors + 1):  # Skip j=0 which is self
            neighbor_idx = neighbors[i, j]
            if i != neighbor_idx:  # Double check no self-loops
                G.add_edge(i, neighbor_idx, weight=float(distances[i, j]))
                edge_count += 1
    
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Find connected components
    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)  # Sort by size, largest first
    
    print(f"Found {len(components)} connected components")
    print(f"Component sizes: {[len(c) for c in components[:10]]}...")  # Show first 10
    
    # Create summary file
    summary_file = output_dir / "connected_components_summary.txt"
    summary_lines = []
    summary_lines.append(f"Total nodes: {G.number_of_nodes()}")
    summary_lines.append(f"Total edges: {G.number_of_edges()}")
    summary_lines.append(f"Number of connected components: {len(components)}")
    summary_lines.append(f"Nearest neighbors used: {n_neighbors}")
    summary_lines.append("")
    
    # Analyze and visualize each component
    for comp_idx, component in enumerate(components):
        component_nodes = list(component)
        comp_size = len(component_nodes)
        
        # Count database vs query nodes
        db_count = sum(1 for node in component_nodes if G.nodes[node]['type'] == 'database')
        query_count = comp_size - db_count
        
        summary_lines.append(f"Component {comp_idx} (size={comp_size}):")
        summary_lines.append(f"  Database images: {db_count}")
        summary_lines.append(f"  Query images: {query_count}")
        
        # Create directory for this component
        comp_dir = output_dir / f"component_{comp_idx:03d}_size_{comp_size}"
        comp_dir.mkdir(exist_ok=True)
        
        # Save paths for images in this component
        db_paths_in_comp = []
        query_paths_in_comp = []
        
        for node in component_nodes:
            if G.nodes[node]['type'] == 'database':
                db_paths_in_comp.append(G.nodes[node]['path'])
            else:
                query_paths_in_comp.append(G.nodes[node]['path'])
        
        # Save database paths
        if db_paths_in_comp:
            with open(comp_dir / "database_paths.txt", 'w') as f:
                for path in db_paths_in_comp:
                    f.write(f"{path}\n")
        
        # Save query paths
        if query_paths_in_comp:
            with open(comp_dir / "query_paths.txt", 'w') as f:
                for path in query_paths_in_comp:
                    f.write(f"{path}\n")
        
        # Create visualization for small components
        if comp_size <= 20:  # Visualize components with 20 or fewer images
            create_component_visualization(db_paths_in_comp, query_paths_in_comp, 
                                         comp_idx, comp_size, comp_dir)
        
        summary_lines.append("")
    
    # Write summary file
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    # Create t-SNE visualization with component coloring
    print("Creating t-SNE visualization with connected components...")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                random_state=random_state, verbose=1)
    embeddings = tsne.fit_transform(all_descriptors)
    
    # Create component labels
    component_labels = np.zeros(len(all_descriptors), dtype=int)
    for comp_idx, component in enumerate(components):
        for node in component:
            component_labels[node] = comp_idx
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # First subplot: Color by component (show only largest components)
    num_components_to_show = min(20, len(components))
    colors = cm.tab20(np.linspace(0, 1, num_components_to_show))
    
    for comp_idx in range(num_components_to_show):
        mask = component_labels == comp_idx
        comp_embeddings = embeddings[mask]
        comp_types = data_type_labels[mask]
        
        # Plot database points
        db_mask = comp_types == 0
        if np.any(db_mask):
            ax1.scatter(comp_embeddings[db_mask, 0], comp_embeddings[db_mask, 1],
                       c=[colors[comp_idx]], s=50, alpha=0.6,
                       edgecolors='none')
        
        # Plot query points
        query_mask = comp_types == 1
        if np.any(query_mask):
            ax1.scatter(comp_embeddings[query_mask, 0], comp_embeddings[query_mask, 1],
                       c=[colors[comp_idx]], s=100, alpha=0.8, marker='^',
                       edgecolors='black', linewidths=1)
    
    # Plot remaining components in gray
    if len(components) > num_components_to_show:
        mask = component_labels >= num_components_to_show
        if np.any(mask):
            ax1.scatter(embeddings[mask, 0], embeddings[mask, 1],
                       c='gray', s=30, alpha=0.3, label='Other components')
    
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.set_title(f'Connected Components (Top {num_components_to_show})', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Second subplot: Show component size distribution
    component_sizes = [len(c) for c in components]
    ax2.hist(component_sizes, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Component Size', fontsize=12)
    ax2.set_ylabel('Number of Components', fontsize=12)
    ax2.set_title('Distribution of Connected Component Sizes', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')  # Log scale for better visibility
    
    # Add text with statistics
    ax2.text(0.95, 0.95, f'Total components: {len(components)}\nLargest: {max(component_sizes)}\nSingletons: {sum(1 for s in component_sizes if s == 1)}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / "connected_components_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Connected components visualization saved to {output_dir}")
    
    # Save graph structure
    nx.write_gexf(G, output_dir / "nearest_neighbor_graph.gexf")
    print(f"Graph structure saved to {output_dir / 'nearest_neighbor_graph.gexf'}")
    
    return G, components


def create_nn_graph_with_leiden(database_descriptors, queries_descriptors, 
                               database_paths, queries_paths, output_dir,
                               n_neighbors=2, resolution=1.0, n_iterations=2,
                               perplexity=30, n_iter=1000, random_state=42):
    """Create a nearest neighbor graph and detect communities using hierarchical Leiden algorithm.
    
    Parameters
    ----------
    database_descriptors : np.array of shape [num_database x descriptor_dim]
    queries_descriptors : np.array of shape [num_queries x descriptor_dim]
    database_paths : list of paths to database images
    queries_paths : list of paths to query images
    output_dir : Path, directory to save the community visualizations
    n_neighbors : int, number of nearest neighbors to connect (default=2)
    resolution : float, resolution parameter for Leiden (higher = smaller communities)
    n_iterations : int, number of iterations for Leiden algorithm
    perplexity : float, t-SNE perplexity parameter
    n_iter : int, number of iterations for t-SNE
    random_state : int, random seed for reproducibility
    """
    try:
        import leidenalg
        import igraph as ig
    except ImportError:
        print("Installing python-igraph and leidenalg for Leiden community detection...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-igraph", "leidenalg"])
        import leidenalg
        import igraph as ig
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Combine all descriptors and paths
    all_descriptors = np.vstack([database_descriptors, queries_descriptors])
    all_paths = database_paths + queries_paths
    num_database = len(database_descriptors)
    
    # Create labels (0 for database, 1 for queries)
    data_type_labels = np.concatenate([
        np.zeros(len(database_descriptors)),
        np.ones(len(queries_descriptors))
    ])
    
    print(f"Building nearest neighbor index for {len(all_descriptors)} descriptors...")
    
    # Build FAISS index for efficient NN search
    dimension = all_descriptors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(all_descriptors.astype(np.float32))
    
    # Find k+1 nearest neighbors (including self)
    distances, neighbors = index.search(all_descriptors.astype(np.float32), n_neighbors + 1)
    
    # Create igraph Graph
    edges = []
    weights = []
    
    for i in range(len(all_descriptors)):
        for j in range(1, n_neighbors + 1):  # Skip j=0 which is self
            neighbor_idx = neighbors[i, j]
            if i != neighbor_idx:
                # Use inverse distance as weight (similarity)
                weight = 1.0 / (distances[i, j] + 1e-6)
                edges.append((i, neighbor_idx))
                weights.append(weight)
    
    # Create igraph
    g = ig.Graph(n=len(all_descriptors))
    g.add_edges(edges)
    g.es['weight'] = weights
    
    # Add node attributes
    g.vs['type'] = ['database' if i < num_database else 'query' for i in range(len(all_descriptors))]
    g.vs['path'] = all_paths
    
    print(f"Created graph with {g.vcount()} nodes and {g.ecount()} edges")
    
    # Create summary file
    summary_file = output_dir / "leiden_communities_summary.txt"
    summary_lines = []
    summary_lines.append(f"Total nodes: {g.vcount()}")
    summary_lines.append(f"Total edges: {g.ecount()}")
    summary_lines.append(f"Nearest neighbors used: {n_neighbors}")
    summary_lines.append(f"Resolution parameter: {resolution}")
    summary_lines.append("")
    
    # Run t-SNE once for all visualizations
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                random_state=random_state, verbose=1)
    embeddings = tsne.fit_transform(all_descriptors)
    
    # Perform hierarchical Leiden clustering with different resolutions
    print("Performing hierarchical Leiden community detection...")
    
    resolutions = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # Different resolution levels
    level_data = []
    
    for level, res in enumerate(resolutions):
        print(f"\nAnalyzing resolution {res}...")
        
        # Run Leiden algorithm
        partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, 
                                           weights='weight', resolution_parameter=res,
                                           n_iterations=n_iterations, seed=random_state)
        
        # Get community assignments
        community_labels = np.array(partition.membership)
        n_communities = len(set(community_labels))
        
        # Calculate modularity
        modularity = partition.modularity
        
        summary_lines.append(f"Resolution {res}:")
        summary_lines.append(f"  Number of communities: {n_communities}")
        summary_lines.append(f"  Modularity: {modularity:.4f}")
        
        # Create level directory
        level_dir = output_dir / f"resolution_{res:.1f}_communities_{n_communities}"
        level_dir.mkdir(exist_ok=True)
        
        # Analyze each community
        community_info = []
        for comm_id in range(n_communities):
            nodes_in_comm = np.where(community_labels == comm_id)[0]
            db_count = sum(1 for node in nodes_in_comm if g.vs[node]['type'] == 'database')
            query_count = len(nodes_in_comm) - db_count
            
            community_info.append({
                'id': comm_id,
                'size': len(nodes_in_comm),
                'db_count': db_count,
                'query_count': query_count,
                'nodes': nodes_in_comm
            })
            
            summary_lines.append(f"    Community {comm_id}: {len(nodes_in_comm)} nodes ({db_count} DB, {query_count} Q)")
        
        summary_lines.append("")
        
        # Sort communities by size
        community_info.sort(key=lambda x: x['size'], reverse=True)
        
        # Save community assignments for this level
        for comm_data in community_info:
            comm_id = comm_data['id']
            comm_dir = level_dir / f"community_{comm_id:03d}_size_{comm_data['size']}"
            comm_dir.mkdir(exist_ok=True)
            
            # Get paths for this community
            db_paths_in_comm = []
            query_paths_in_comm = []
            
            for node in comm_data['nodes']:
                if g.vs[node]['type'] == 'database':
                    db_paths_in_comm.append(g.vs[node]['path'])
                else:
                    query_paths_in_comm.append(g.vs[node]['path'])
            
            # Save paths
            if db_paths_in_comm:
                with open(comm_dir / "database_paths.txt", 'w') as f:
                    for path in db_paths_in_comm:
                        f.write(f"{path}\n")
            
            if query_paths_in_comm:
                with open(comm_dir / "query_paths.txt", 'w') as f:
                    for path in query_paths_in_comm:
                        f.write(f"{path}\n")
            
            # Create visualization for small communities
            if comm_data['size'] <= 20:
                create_component_visualization(db_paths_in_comm, query_paths_in_comm,
                                             comm_id, comm_data['size'], comm_dir)
        
        # Create visualization for this resolution level
        create_leiden_level_visualization(embeddings, community_labels, data_type_labels,
                                        res, n_communities, modularity, level_dir)
        
        level_data.append({
            'resolution': res,
            'n_communities': n_communities,
            'modularity': modularity,
            'membership': community_labels
        })
    
    # Write summary file
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    # Create comparison visualization across resolutions
    create_leiden_hierarchy_visualization(embeddings, level_data, data_type_labels, output_dir)
    
    # Save graph structure in multiple formats
    g.write_graphml(str(output_dir / "leiden_graph.graphml"))
    
    # Also create NetworkX graph for compatibility
    G = nx.Graph()
    for i in range(g.vcount()):
        G.add_node(i, type=g.vs[i]['type'], path=g.vs[i]['path'])
    for edge in g.es:
        G.add_edge(edge.source, edge.target, weight=edge['weight'])
    nx.write_gexf(G, output_dir / "leiden_graph.gexf")
    
    print(f"\nHierarchical Leiden community detection completed!")
    print(f"Results saved in {output_dir}")
    
    return g, level_data


def create_leiden_level_visualization(embeddings, community_labels, data_type_labels,
                                    resolution, n_communities, modularity, output_dir):
    """Create visualization for a single resolution level of Leiden algorithm."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # First subplot: t-SNE colored by community
    # Use a colormap that can handle many communities
    if n_communities <= 20:
        colors = cm.tab20(np.linspace(0, 1, n_communities))
    else:
        colors = cm.gist_rainbow(np.linspace(0, 1, n_communities))
    
    for comm_id in range(n_communities):
        mask = community_labels == comm_id
        
        # Separate database and query points
        db_mask = mask & (data_type_labels == 0)
        query_mask = mask & (data_type_labels == 1)
        
        # Plot database points
        if np.any(db_mask):
            ax1.scatter(embeddings[db_mask, 0], embeddings[db_mask, 1],
                       c=[colors[comm_id]], s=50, alpha=0.6,
                       edgecolors='none')
        
        # Plot query points
        if np.any(query_mask):
            ax1.scatter(embeddings[query_mask, 0], embeddings[query_mask, 1],
                       c=[colors[comm_id]], s=100, alpha=0.8, marker='^',
                       edgecolors='black', linewidths=1)
    
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.set_title(f'Resolution {resolution}: {n_communities} Communities (Modularity: {modularity:.3f})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add legend for database vs query
    db_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                         markersize=8, alpha=0.6, label='Database')
    query_patch = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                            markersize=10, alpha=0.8, markeredgecolor='black', label='Query')
    ax1.legend(handles=[db_patch, query_patch], loc='upper right')
    
    # Second subplot: Community size distribution
    community_sizes = np.bincount(community_labels)
    
    # Separate counts for database and queries
    db_counts = np.zeros(n_communities, dtype=int)
    query_counts = np.zeros(n_communities, dtype=int)
    
    for comm_id in range(n_communities):
        mask = community_labels == comm_id
        db_counts[comm_id] = np.sum(mask & (data_type_labels == 0))
        query_counts[comm_id] = np.sum(mask & (data_type_labels == 1))
    
    # Sort by total size
    sorted_indices = np.argsort(community_sizes)[::-1]
    
    x = np.arange(min(n_communities, 30))  # Show max 30 communities
    width = 0.35
    
    ax2.bar(x - width/2, db_counts[sorted_indices[:30]], width, label='Database', alpha=0.7, color='blue')
    ax2.bar(x + width/2, query_counts[sorted_indices[:30]], width, label='Queries', alpha=0.7, color='red')
    
    ax2.set_xlabel('Community (sorted by size)', fontsize=12)
    ax2.set_ylabel('Number of Images', fontsize=12)
    ax2.set_title('Community Size Distribution (Top 30)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis labels
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(sorted_indices[i]) for i in range(min(n_communities, 30))], rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"resolution_{resolution:.1f}_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_leiden_hierarchy_visualization(embeddings, level_data, data_type_labels, output_dir):
    """Create a comparison visualization across all resolution levels."""
    n_levels = len(level_data)
    
    # Create grid layout
    n_cols = 3
    n_rows = (n_levels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, level_info in enumerate(level_data):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Get community assignments for this level
        community_labels = level_info['membership']
        n_communities = level_info['n_communities']
        resolution = level_info['resolution']
        
        # Use appropriate colormap
        if n_communities <= 20:
            colors = cm.tab20(np.linspace(0, 1, n_communities))
        else:
            colors = cm.gist_rainbow(np.linspace(0, 1, n_communities))
        
        # Plot each community
        for comm_id in range(n_communities):
            mask = community_labels == comm_id
            
            # Plot all points in this community with the same color
            if np.any(mask):
                # Database points
                db_mask = mask & (data_type_labels == 0)
                if np.any(db_mask):
                    ax.scatter(embeddings[db_mask, 0], embeddings[db_mask, 1],
                              c=[colors[comm_id]], s=30, alpha=0.6, edgecolors='none')
                
                # Query points
                query_mask = mask & (data_type_labels == 1)
                if np.any(query_mask):
                    ax.scatter(embeddings[query_mask, 0], embeddings[query_mask, 1],
                              c=[colors[comm_id]], s=60, alpha=0.8, marker='^',
                              edgecolors='black', linewidths=0.5)
        
        ax.set_title(f"Resolution {resolution}: {n_communities} communities\nModularity: {level_info['modularity']:.3f}",
                    fontsize=10)
        ax.set_xlabel('t-SNE 1', fontsize=8)
        ax.set_ylabel('t-SNE 2', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_levels, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('Hierarchical Leiden Community Detection - Different Resolutions', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "leiden_hierarchy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create resolution analysis plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    resolutions = [info['resolution'] for info in level_data]
    modularities = [info['modularity'] for info in level_data]
    n_communities_list = [info['n_communities'] for info in level_data]
    
    # Modularity vs resolution
    ax1.plot(resolutions, modularities, 'o-', color='blue', linewidth=2, markersize=8)
    ax1.set_xlabel('Resolution Parameter', fontsize=12)
    ax1.set_ylabel('Modularity', fontsize=12)
    ax1.set_title('Modularity vs Resolution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Number of communities vs resolution
    ax2.plot(resolutions, n_communities_list, 's-', color='red', linewidth=2, markersize=8)
    ax2.set_xlabel('Resolution Parameter', fontsize=12)
    ax2.set_ylabel('Number of Communities', fontsize=12)
    ax2.set_title('Community Count vs Resolution', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / "leiden_resolution_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_nn_graph_with_louvain(database_descriptors, queries_descriptors, 
                                 database_paths, queries_paths, output_dir,
                                 n_neighbors=2, resolution=1.0, perplexity=30, 
                                 n_iter=1000, random_state=42):
    """Create a nearest neighbor graph and detect communities using Louvain algorithm.
    
    Parameters
    ----------
    database_descriptors : np.array of shape [num_database x descriptor_dim]
    queries_descriptors : np.array of shape [num_queries x descriptor_dim]
    database_paths : list of paths to database images
    queries_paths : list of paths to query images
    output_dir : Path, directory to save the community visualizations
    n_neighbors : int, number of nearest neighbors to connect (default=2)
    resolution : float, resolution parameter for Louvain (higher = smaller communities)
    perplexity : float, t-SNE perplexity parameter
    n_iter : int, number of iterations for t-SNE
    random_state : int, random seed for reproducibility
    """
    try:
        import community as community_louvain
    except ImportError:
        print("Installing python-louvain for community detection...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-louvain"])
        import community as community_louvain
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Combine all descriptors and paths
    all_descriptors = np.vstack([database_descriptors, queries_descriptors])
    all_paths = database_paths + queries_paths
    num_database = len(database_descriptors)
    
    # Create labels (0 for database, 1 for queries)
    data_type_labels = np.concatenate([
        np.zeros(len(database_descriptors)),
        np.ones(len(queries_descriptors))
    ])
    
    print(f"Building nearest neighbor index for {len(all_descriptors)} descriptors...")
    
    # Build FAISS index for efficient NN search
    dimension = all_descriptors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(all_descriptors.astype(np.float32))
    
    # Find k+1 nearest neighbors (including self)
    distances, neighbors = index.search(all_descriptors.astype(np.float32), n_neighbors + 1)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for i in range(len(all_descriptors)):
        node_type = 'database' if i < num_database else 'query'
        G.add_node(i, type=node_type, path=all_paths[i])
    
    # Add edges with weights (using similarity = 1/distance)
    for i in range(len(all_descriptors)):
        for j in range(1, n_neighbors + 1):  # Skip j=0 which is self
            neighbor_idx = neighbors[i, j]
            if i != neighbor_idx:
                # Use inverse distance as weight (similarity)
                weight = 1.0 / (distances[i, j] + 1e-6)
                G.add_edge(i, neighbor_idx, weight=weight)
    
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Detect communities using Louvain algorithm
    print("Detecting communities using Louvain algorithm...")
    
    # Get dendrogram (hierarchical communities)
    dendrogram = community_louvain.generate_dendrogram(G, weight='weight', resolution=resolution)
    
    # Create summary file
    summary_file = output_dir / "louvain_communities_summary.txt"
    summary_lines = []
    summary_lines.append(f"Total nodes: {G.number_of_nodes()}")
    summary_lines.append(f"Total edges: {G.number_of_edges()}")
    summary_lines.append(f"Nearest neighbors used: {n_neighbors}")
    summary_lines.append(f"Resolution parameter: {resolution}")
    summary_lines.append(f"Number of hierarchical levels: {len(dendrogram)}")
    summary_lines.append("")
    
    # Run t-SNE once for all visualizations
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                random_state=random_state, verbose=1)
    embeddings = tsne.fit_transform(all_descriptors)
    
    # Visualize communities at each level
    level_data = []
    for level in range(len(dendrogram)):
        print(f"\nAnalyzing level {level}...")
        partition = community_louvain.partition_at_level(dendrogram, level)
        
        # Convert partition dict to array
        community_labels = np.array([partition[i] for i in range(len(all_descriptors))])
        n_communities = len(set(community_labels))
        
        # Calculate modularity
        modularity = community_louvain.modularity(partition, G, weight='weight')
        
        summary_lines.append(f"Level {level}:")
        summary_lines.append(f"  Number of communities: {n_communities}")
        summary_lines.append(f"  Modularity: {modularity:.4f}")
        
        # Create level directory
        level_dir = output_dir / f"level_{level:02d}_communities_{n_communities}"
        level_dir.mkdir(exist_ok=True)
        
        # Analyze each community
        community_info = []
        for comm_id in range(n_communities):
            nodes_in_comm = np.where(community_labels == comm_id)[0]
            db_count = sum(1 for node in nodes_in_comm if G.nodes[node]['type'] == 'database')
            query_count = len(nodes_in_comm) - db_count
            
            community_info.append({
                'id': comm_id,
                'size': len(nodes_in_comm),
                'db_count': db_count,
                'query_count': query_count,
                'nodes': nodes_in_comm
            })
            
            summary_lines.append(f"    Community {comm_id}: {len(nodes_in_comm)} nodes ({db_count} DB, {query_count} Q)")
        
        summary_lines.append("")
        
        # Sort communities by size
        community_info.sort(key=lambda x: x['size'], reverse=True)
        
        # Save community assignments for this level
        for comm_data in community_info:
            comm_id = comm_data['id']
            comm_dir = level_dir / f"community_{comm_id:03d}_size_{comm_data['size']}"
            comm_dir.mkdir(exist_ok=True)
            
            # Get paths for this community
            db_paths_in_comm = []
            query_paths_in_comm = []
            
            for node in comm_data['nodes']:
                if G.nodes[node]['type'] == 'database':
                    db_paths_in_comm.append(G.nodes[node]['path'])
                else:
                    query_paths_in_comm.append(G.nodes[node]['path'])
            
            # Save paths
            if db_paths_in_comm:
                with open(comm_dir / "database_paths.txt", 'w') as f:
                    for path in db_paths_in_comm:
                        f.write(f"{path}\n")
            
            if query_paths_in_comm:
                with open(comm_dir / "query_paths.txt", 'w') as f:
                    for path in query_paths_in_comm:
                        f.write(f"{path}\n")
            
            # Create visualization for small communities
            if comm_data['size'] <= 20:
                create_component_visualization(db_paths_in_comm, query_paths_in_comm, 
                                             comm_id, comm_data['size'], comm_dir)
        
        # Create visualization for this level
        create_louvain_level_visualization(embeddings, community_labels, data_type_labels,
                                         level, n_communities, modularity, level_dir)
        
        level_data.append({
            'level': level,
            'n_communities': n_communities,
            'modularity': modularity,
            'partition': partition
        })
    
    # Write summary file
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    # Create comparison visualization across levels
    create_louvain_hierarchy_visualization(embeddings, level_data, data_type_labels, output_dir)
    
    # Save graph structure
    nx.write_gexf(G, output_dir / "louvain_graph.gexf")
    print(f"Graph structure saved to {output_dir / 'louvain_graph.gexf'}")
    
    print(f"\nLouvain community detection completed!")
    print(f"Results saved in {output_dir}")
    
    return G, dendrogram, level_data


def create_louvain_level_visualization(embeddings, community_labels, data_type_labels,
                                     level, n_communities, modularity, output_dir):
    """Create visualization for a single level of Louvain hierarchy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # First subplot: t-SNE colored by community
    # Use a colormap that can handle many communities
    if n_communities <= 20:
        colors = cm.tab20(np.linspace(0, 1, n_communities))
    else:
        colors = cm.gist_rainbow(np.linspace(0, 1, n_communities))
    
    for comm_id in range(n_communities):
        mask = community_labels == comm_id
        
        # Separate database and query points
        db_mask = mask & (data_type_labels == 0)
        query_mask = mask & (data_type_labels == 1)
        
        # Plot database points
        if np.any(db_mask):
            ax1.scatter(embeddings[db_mask, 0], embeddings[db_mask, 1], 
                       c=[colors[comm_id]], alpha=0.6, s=50, 
                       label=f'DB Cluster {comm_id}', edgecolors='none')
        
        # Plot query points
        query_mask = mask & (data_type_labels == 1)
        if np.any(query_mask):
            ax1.scatter(embeddings[query_mask, 0], embeddings[query_mask, 1], 
                       c=[colors[comm_id]], alpha=0.8, s=100, 
                       marker='^', label=f'Query Cluster {comm_id}', 
                       edgecolors='black', linewidths=1)
    
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.set_title(f'Level {level}: {n_communities} Communities (Modularity: {modularity:.3f})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add legend for database vs query
    db_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                         markersize=8, alpha=0.6, label='Database')
    query_patch = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                            markersize=10, alpha=0.8, markeredgecolor='black', label='Query')
    ax1.legend(handles=[db_patch, query_patch], loc='upper right')
    
    # Second subplot: Community size distribution
    community_sizes = np.bincount(community_labels)
    
    # Separate counts for database and queries
    db_counts = np.zeros(n_communities, dtype=int)
    query_counts = np.zeros(n_communities, dtype=int)
    
    for comm_id in range(n_communities):
        mask = community_labels == comm_id
        db_counts[comm_id] = np.sum(mask & (data_type_labels == 0))
        query_counts[comm_id] = np.sum(mask & (data_type_labels == 1))
    
    # Sort by total size
    sorted_indices = np.argsort(community_sizes)[::-1]
    
    x = np.arange(n_communities)
    width = 0.35
    
    ax2.bar(x - width/2, db_counts[sorted_indices], width, label='Database', alpha=0.7, color='blue')
    ax2.bar(x + width/2, query_counts[sorted_indices], width, label='Queries', alpha=0.7, color='red')
    
    ax2.set_xlabel('Community (sorted by size)', fontsize=12)
    ax2.set_ylabel('Number of Images', fontsize=12)
    ax2.set_title('Community Size Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Only show labels for first few communities
    if n_communities > 20:
        ax2.set_xticks(x[:20])
        ax2.set_xticklabels([str(sorted_indices[i]) for i in range(20)])
    else:
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(sorted_indices[i]) for i in range(n_communities)])
    
    plt.tight_layout()
    plt.savefig(output_dir / f"level_{level:02d}_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_louvain_hierarchy_visualization(embeddings, level_data, data_type_labels, output_dir):
    """Create a comparison visualization across all hierarchy levels."""
    n_levels = len(level_data)
    
    if n_levels <= 4:
        fig, axes = plt.subplots(1, n_levels, figsize=(6*n_levels, 5))
        if n_levels == 1:
            axes = [axes]
    else:
        n_rows = (n_levels + 3) // 4
        fig, axes = plt.subplots(n_rows, 4, figsize=(24, 5*n_rows))
        axes = axes.flatten()
    
    for idx, level_info in enumerate(level_data):
        ax = axes[idx]
        
        # Get partition for this level
        partition = level_info['partition']
        community_labels = np.array([partition[i] for i in range(len(embeddings))])
        n_communities = level_info['n_communities']
        
        # Use appropriate colormap
        if n_communities <= 20:
            colors = cm.tab20(np.linspace(0, 1, n_communities))
        else:
            colors = cm.gist_rainbow(np.linspace(0, 1, n_communities))
        
        # Plot each community
        for comm_id in range(n_communities):
            mask = community_labels == comm_id
            
            # Plot all points in this community with the same color
            if np.any(mask):
                # Database points
                db_mask = mask & (data_type_labels == 0)
                if np.any(db_mask):
                    ax.scatter(embeddings[db_mask, 0], embeddings[db_mask, 1],
                              c=[colors[comm_id]], s=30, alpha=0.6, edgecolors='none')
                
                # Query points
                query_mask = mask & (data_type_labels == 1)
                if np.any(query_mask):
                    ax.scatter(embeddings[query_mask, 0], embeddings[query_mask, 1],
                              c=[colors[comm_id]], s=60, alpha=0.8, marker='^',
                              edgecolors='black', linewidths=0.5)
        
        ax.set_title(f"Level {idx}: {n_communities} communities\nModularity: {level_info['modularity']:.3f}",
                    fontsize=10)
        ax.set_xlabel('t-SNE 1', fontsize=8)
        ax.set_ylabel('t-SNE 2', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots if any
    if n_levels > 1 and n_levels % 4 != 0:
        for idx in range(n_levels, len(axes)):
            axes[idx].set_visible(False)
    
    plt.suptitle('Louvain Community Detection - Hierarchy Levels', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "louvain_hierarchy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create modularity plot
    plt.figure(figsize=(10, 6))
    levels = [info['level'] for info in level_data]
    modularities = [info['modularity'] for info in level_data]
    n_communities_list = [info['n_communities'] for info in level_data]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Hierarchy Level', fontsize=12)
    ax1.set_ylabel('Modularity', color=color, fontsize=12)
    ax1.plot(levels, modularities, 'o-', color=color, linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Number of Communities', color=color, fontsize=12)
    ax2.plot(levels, n_communities_list, 's-', color=color, linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Modularity and Community Count by Hierarchy Level', fontsize=14)
    fig.tight_layout()
    plt.savefig(output_dir / "louvain_modularity_plot.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_component_visualization(db_paths, query_paths, comp_idx, comp_size, output_dir):
    """Create a visualization grid for a single connected component."""
    from PIL import Image, ImageDraw, ImageFont
    import math
    
    all_paths = db_paths + query_paths
    if not all_paths:
        return
    
    # Calculate grid dimensions
    n_images = len(all_paths)
    grid_cols = min(5, n_images)  # Max 5 columns
    grid_rows = math.ceil(n_images / grid_cols)
    
    # Image dimensions
    img_size = 200
    padding = 10
    
    # Create canvas
    canvas_width = grid_cols * img_size + (grid_cols + 1) * padding
    canvas_height = grid_rows * img_size + (grid_rows + 1) * padding + 50  # Extra space for title
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Add title
    title = f"Component {comp_idx} - Size: {comp_size} (DB: {len(db_paths)}, Q: {len(query_paths)})"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = None
    
    text_bbox = draw.textbbox((0, 0), title, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    draw.text(((canvas_width - text_width) // 2, 10), title, fill='black', font=font)
    
    # Place images
    for idx, path in enumerate(all_paths):
        row = idx // grid_cols
        col = idx % grid_cols
        
        x = col * img_size + (col + 1) * padding
        y = row * img_size + (row + 1) * padding + 50
        
        try:
            img = Image.open(path).convert('RGB')
            img.thumbnail((img_size, img_size), Image.Resampling.LANCZOS)
            
            # Add border color based on type
            is_query = path in query_paths
            border_color = (255, 0, 0) if is_query else (0, 0, 255)  # Red for query, blue for database
            
            # Create bordered image
            bordered = Image.new('RGB', (img_size, img_size), border_color)
            img_x = (img_size - img.width) // 2
            img_y = (img_size - img.height) // 2
            bordered.paste(img, (img_x, img_y))
            
            # Draw border
            draw_border = ImageDraw.Draw(bordered)
            draw_border.rectangle([0, 0, img_size-1, img_size-1], outline=border_color, width=3)
            
            canvas.paste(bordered, (x, y))
            
            # Add label
            label = "Q" if is_query else "DB"
            draw.text((x + 5, y + 5), label, fill='white', font=font)
            
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Draw placeholder
            draw.rectangle([x, y, x + img_size, y + img_size], fill='lightgray')
            draw.text((x + img_size//2 - 20, y + img_size//2), "Error", fill='red')
    
    # Save visualization
    canvas.save(output_dir / f"component_{comp_idx:03d}_visualization.jpg", quality=85)
