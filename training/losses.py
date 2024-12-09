# src/training/losses.py
import torch
from torch.nn import functional as F
from torch import nn
from typing import Optional

# src/training/losses.py
class SegmentationLoss(nn.Module):
    """Base class for segmentation losses."""
    def __init__(self, num_classes: int, weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_classes = num_classes
        self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DiceLoss(SegmentationLoss):
    def __init__(
        self,
        num_classes: int,
        weights: Optional[torch.Tensor] = None,
        smooth: float = 1e-6,
        ignore_index: Optional[int] = None
    ):
        """
        DiceLoss with optional class weights.

        Args:
            num_classes (int): Number of classes (including background)
            weights (torch.Tensor, optional): Weights for each class. Shape: (num_classes,)
            smooth (float): Smoothing factor to avoid division by zero
            ignore_index (int, optional): Index to ignore in loss calculation (e.g., for ignore_index=-1)
        """
        super(DiceLoss, self).__init__(num_classes=num_classes, weights=weights)  # Pass parameters to parent
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice Loss.

        Args:
            pred (torch.Tensor): Predicted logits of shape (B, C, H, W)
            target (torch.Tensor): Ground truth labels of shape (B, H, W)

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Convert predictions to probabilities
        pred = torch.softmax(pred, dim=1)

        # Convert target to one-hot encoding
        target = F.one_hot(target, num_classes=self.num_classes)
        target = target.permute(0, 3, 1, 2).float()

        # Handle ignore index if specified
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float()
            pred = pred * mask
            target = target * mask

        # Calculate intersection and union
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        # Calculate Dice score
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Apply class weights if specified
        if self.weights is not None:
            weights = self.weights.to(pred.device)
            dice_score = dice_score * weights.view(1, -1)

        # Return mean Dice loss
        return 1.0 - dice_score.mean()

    def __repr__(self) -> str:
        """String representation of the loss function."""
        return (f"DiceLoss(num_classes={self.num_classes}, "
                f"weights={'None' if self.weights is None else 'specified'}, "
                f"smooth={self.smooth}, "
                f"ignore_index={self.ignore_index})")
    

class CombinedLoss(SegmentationLoss):
    """Combine multiple losses with weights."""
    def __init__(self, losses: list, weights: list):
        super().__init__()
        self.losses = losses
        self.loss_weights = weights

    def forward(self, pred, target):
        return sum(w * loss(pred, target) 
                  for loss, w in zip(self.losses, self.loss_weights))