import torch
import numpy as np

def compute_class_weights(dataset):
    """
    Compute class weights based on pixel frequency in masks.
    Returns weights that are inversely proportional to class frequency.
    """
    print("Starting class weight computation...")
    class_counts = torch.zeros(33)
    
    for idx in range(len(dataset)):
        if idx % 10 == 0:  # Print progress every 10 images
            print(f"Processing image {idx}/{len(dataset)}")
        _, mask = dataset[idx]
        mask_array = torch.as_tensor(np.array(mask))
        unique_values = torch.unique(mask_array)
        print(f"Unique values in mask {idx}: {unique_values}")  # Debug print
        
        for class_id in range(33):
            pixel_count = (mask_array == class_id).sum().item()
            class_counts[class_id] += pixel_count
            
    print("Class counts:", class_counts)
    
    total_pixels = class_counts.sum()
    class_frequencies = class_counts / total_pixels
    print("Class frequencies:", class_frequencies)
    
    weights = 1.0 / (class_frequencies + 1e-6)
    weights = weights / weights.sum()
    print("Computed weights:", weights)
    
    return weights


def dice_metric(pred, target, num_classes=33):
    """
    Compute per-class Dice scores.
    """
    pred = torch.argmax(pred, dim=1)  # Shape: (batch_size, H, W)
    dice_scores = []

    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        if union == 0:  # Avoid NaN for empty classes
            dice_scores.append(torch.tensor(1.0))  # Perfect score for empty classes
        else:
            dice_scores.append((2.0 * intersection) / (union + 1e-6))

    return dice_scores