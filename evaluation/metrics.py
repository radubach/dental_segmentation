import numpy as np
import torch
import matplotlib.pyplot as plt

# mapping after adding 1 to category_id
TOOTH_TYPE_MAPPING = {
    # Quadrant 1 (Upper Right)
    1: 'incisor', 
    2: 'incisor', 
    3: 'canine',  
    4: 'premolar',
    5: 'premolar',
    6: 'molar',   
    7: 'molar',   
    8: 'molar',   

    # Quadrant 2 (Upper Left)
    9: 'incisor', 
    10: 'incisor', 
    11: 'canine', 
    12: 'premolar',
    13: 'premolar',
    14: 'molar',   
    15: 'molar',   
    16: 'molar',   

    # Quadrant 3 (Lower Left)
    17: 'incisor', 
    18: 'incisor', 
    19: 'canine',  
    20: 'premolar',
    21: 'premolar',
    22: 'molar',   
    23: 'molar',   
    24: 'molar',   

    # Quadrant 4 (Lower Right)
    25: 'incisor', 
    26: 'incisor', 
    27: 'canine',  
    28: 'premolar',
    29: 'premolar',
    30: 'molar',   
    31: 'molar',   
    32: 'molar',   
}


class SegmentationMetrics:
    """Base class for computing segmentation metrics."""
    def __init__(self, model, device, val_loader, num_classes=33):
        self.model = model
        self.device = device
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.tooth_types = ['incisor', 'canine', 'premolar', 'molar']

    def compute_all_metrics(self):
        """Compute all available metrics."""
        predictions, targets = self._get_predictions()
        return {
            'dice_scores': self.compute_dice_scores(predictions, targets),
            'precision_recall': self.compute_precision_recall(predictions, targets)
        }

    def _get_predictions(self):
        """Get model predictions on validation set."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.append(predictions.cpu())
                all_targets.append(targets)
        return torch.cat(all_predictions), torch.cat(all_targets)

    def _dice_coefficient(self, pred, target):
        """Compute dice coefficient between two binary masks."""
        intersection = torch.logical_and(pred, target).sum()
        union = pred.sum() + target.sum()
        if union == 0:
            return torch.tensor(1.0)  # Both pred and target are empty
        return 2.0 * intersection / union
    
    def _compute_overall_dice(self, predictions, targets, exclude_background=True):
        """Compute overall dice score."""
        if exclude_background:
            mask = targets > 0
            return self._dice_coefficient(predictions[mask] > 0, targets[mask] > 0)
        return self._dice_coefficient(predictions > 0, targets > 0)
    
    def _compute_per_tooth_dice(self, predictions, targets):
        """Compute dice score for each individual tooth."""
        tooth_dice = {}
        for tooth_id in range(1, self.num_classes):  # Skip background
            pred_mask = predictions == tooth_id
            target_mask = targets == tooth_id
            # Only compute dice if this tooth exists in ground truth
            if target_mask.sum() > 0:
                tooth_dice[tooth_id] = self._dice_coefficient(pred_mask, target_mask)
        return tooth_dice
    
    def _compute_per_type_dice(self, predictions, targets):
        """Compute dice score for each tooth type (incisor, canine, etc.)."""
        type_predictions = {}
        type_targets = {}
        type_dice = {}

        # Initialize tensors for each tooth type
        for tooth_type in self.tooth_types:
            type_predictions[tooth_type] = torch.zeros_like(predictions, dtype=torch.bool)
            type_targets[tooth_type] = torch.zeros_like(targets, dtype=torch.bool)

        # Aggregate teeth by type
        for tooth_id, tooth_type in TOOTH_TYPE_MAPPING.items():
            type_predictions[tooth_type] |= (predictions == tooth_id)
            type_targets[tooth_type] |= (targets == tooth_id)

        # Compute dice for each type
        for tooth_type in self.tooth_types:
            type_dice[tooth_type] = self._dice_coefficient(
                type_predictions[tooth_type],
                type_targets[tooth_type]
            )

        return type_dice

    def compute_dice_scores(self, predictions, targets, exclude_background=True):
        """Compute various Dice scores."""
        return {
            'overall': self._compute_overall_dice(predictions, targets, exclude_background).item(),
            'per_tooth': {k: v.item() for k, v in self._compute_per_tooth_dice(predictions, targets).items()},
            'per_type': {k: v.item() for k, v in self._compute_per_type_dice(predictions, targets).items()}
        }
    


    def compute_precision_recall(self, predictions, targets):
        """Compute precision and recall using bbox overlap."""
        pred_boxes = self._masks_to_boxes(predictions)
        target_boxes = self._masks_to_boxes(targets)
        
        precision = self._compute_precision(pred_boxes, target_boxes)
        recall = self._compute_recall(pred_boxes, target_boxes)
        
        return {
            'precision': precision,
            'recall': recall
        }

# src/evaluation/visualization.py
class SegmentationVisualizer:
    """Class for visualizing segmentation results."""
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def visualize_prediction(self, image, target=None):
        """Visualize model prediction with optional ground truth."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(image.unsqueeze(0).to(self.device))
            pred = torch.argmax(output, dim=1)[0]

        fig, axes = plt.subplots(1, 3 if target is not None else 2, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image.squeeze(), cmap='gray')
        axes[0].set_title('Original Image')
        
        # Prediction
        axes[1].imshow(pred.cpu(), cmap='tab20')
        axes[1].set_title('Prediction')
        
        # Ground truth (if provided)
        if target is not None:
            axes[2].imshow(target, cmap='tab20')
            axes[2].set_title('Ground Truth')
            
        plt.show()

# Usage example:
# metrics = SegmentationMetrics(model, device, val_loader)
# results = metrics.compute_all_metrics()

# print("Overall Dice Score:", results['dice_scores']['overall'])
# print("\nDice Scores by Type:")
# for tooth_type, score in results['dice_scores']['per_type'].items():
#     print(f"{tooth_type}: {score:.4f}")

# print("\nPrecision:", results['precision_recall']['precision'])
# print("Recall:", results['precision_recall']['recall'])

# # For visualization
# visualizer = SegmentationVisualizer(model, device)
# image, target = next(iter(val_loader))
# visualizer.visualize_prediction(image[0], target[0])