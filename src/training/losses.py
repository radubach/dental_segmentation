import torch
from torch.nn import functional as F
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self, weights=None):
        """
        DiceLoss with optional class weights.

        Args:
            weights (torch.Tensor): Weights for each class. Shape: (num_classes,)
        """
        super(DiceLoss, self).__init__()
        self.weights = weights

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)  # Class probabilities
        target = F.one_hot(target, num_classes=33).permute(0, 3, 1, 2).float()  # One-hot encode target

        intersection = (pred * target).sum(dim=(2, 3))  # Per class intersection
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))  # Per class union

        dice_score = 2.0 * intersection / (union + 1e-6)  # Per class Dice score

        # Apply weights
        if self.weights is not None:
            dice_score = dice_score * self.weights.view(1, -1)

        return 1.0 - dice_score.mean()  # Mean weighted Dice loss