import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch
from PIL import Image, ImageOps
import torchvision.transforms as tfm

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
