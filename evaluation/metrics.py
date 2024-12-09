import numpy as np
import torch
import matplotlib.pyplot as plt

TOOTH_TYPE_MAPPING = {
    # Universal Numbering System (1-32)
    # Incisors
    1: 'incisor', 2: 'incisor', 7: 'incisor', 8: 'incisor',
    9: 'incisor', 10: 'incisor', 15: 'incisor', 16: 'incisor',
    # Canines
    3: 'canine', 6: 'canine', 11: 'canine', 14: 'canine',
    # Premolars
    4: 'premolar', 5: 'premolar', 12: 'premolar', 13: 'premolar',
    20: 'premolar', 21: 'premolar', 28: 'premolar', 29: 'premolar',
    # Molars
    17: 'molar', 18: 'molar', 19: 'molar',
    30: 'molar', 31: 'molar', 32: 'molar'
}

# src/evaluation/metrics.py
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

    def compute_dice_scores(self, predictions, targets):
        """Compute various Dice scores."""
        # Overall dice (excluding background)
        overall_dice = self._compute_overall_dice(predictions, targets)
        
        # Per-tooth dice
        tooth_dice = self._compute_per_tooth_dice(predictions, targets)
        
        # Per-type dice
        type_dice = self._compute_per_type_dice(tooth_dice)
        
        return {
            'overall': overall_dice,
            'per_tooth': tooth_dice,
            'per_type': type_dice
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