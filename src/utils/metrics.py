import torch

def compute_class_weights(dataset):
    """
    Compute class weights based on pixel frequency in masks.
    Returns weights that are inversely proportional to class frequency.
    """
    # Initialize pixel counts for each class
    class_counts = torch.zeros(33)  # 0 for background, 1-32 for teeth
    
    # Count pixels of each class
    for idx in range(len(dataset)):
        _, mask = dataset[idx]
        mask_array = torch.as_tensor(mask)
        for class_id in range(33):
            class_counts[class_id] += (mask_array == class_id).sum()
    
    # Compute weights (inverse of frequency)
    total_pixels = class_counts.sum()
    class_frequencies = class_counts / total_pixels
    weights = 1.0 / (class_frequencies + 1e-6)  # add small epsilon to avoid division by zero
    
    # Normalize weights
    weights = weights / weights.sum()
    
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