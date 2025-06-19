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
