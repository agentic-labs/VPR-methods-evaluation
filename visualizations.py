import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch
from PIL import Image, ImageOps
import torchvision.transforms as tfm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path

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
