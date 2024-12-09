import torch

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