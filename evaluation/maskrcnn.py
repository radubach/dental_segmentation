from .base import BaseEvaluator
import torch
import numpy as np
from PIL import Image
import torchvision
from typing import Tuple, List, Dict, Any
from data.transforms import SegmentationTransforms
from configs.config import Config

class MaskRCNNEvaluator(BaseEvaluator):
    """Evaluator for Mask R-CNN instance segmentation model predictions.
    
    This class handles the evaluation pipeline for a Mask R-CNN model, including:
    - Loading and preprocessing images
    - Generating model predictions with confidence scores
    - Post-processing predictions into instance masks and bounding boxes
    - Non-maximum suppression (NMS) to remove overlapping predictions
    - Resizing outputs back to original image dimensions
    
    Args:
        model: Trained Mask R-CNN model for generating predictions
        config: Configuration object containing model and data parameters
        confidence_threshold: Minimum confidence score for predictions (default: 0.5)
        nms_threshold: IoU threshold for non-maximum suppression (default: 0.5)
    
    Example:
        >>> evaluator = MaskRCNNEvaluator(model, config)
        >>> predictions = evaluator.get_predictions(image_id)
        >>> # predictions contains 'boxes', 'masks', 'labels', and 'scores'
    """
    def __init__(
        self,
        model,
        config: Config,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.model.device)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        self.model.to(self.device)
        self.model.eval()
        
    def get_predictions(self, image_id: int) -> Dict[str, torch.Tensor]:
        """Get predictions for a single image.
        
        Args:
            image_id: ID of the image to process
            
        Returns:
            dict containing:
                - boxes: Tensor[N, 4] - Bounding boxes in (x1, y1, x2, y2) format
                - masks: Tensor[N, H, W] - Binary masks for each instance
                - labels: Tensor[N] - Classification labels for each instance
                - scores: Tensor[N] - Confidence scores for each instance
        """
        # Get original image and size
        original_image = self.val_dataset.load_image(image_id)
        original_size = original_image.size  # (width, height)
        
        # Use inference transforms to prepare image for model
        transform = SegmentationTransforms.get_inference_transforms(
            self.config.data,
            is_instance_segmentation=True
        )
        transformed = transform(image=np.array(original_image))
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            predictions = self.model(image_tensor)[0]  # Get first (only) image predictions
            
        # Filter predictions by confidence threshold
        keep = predictions['scores'] > self.confidence_threshold
        filtered_predictions = {
            'boxes': predictions['boxes'][keep],
            'masks': predictions['masks'][keep],
            'labels': predictions['labels'][keep],
            'scores': predictions['scores'][keep]
        }
        
        # Apply NMS if there are any predictions
        if filtered_predictions['boxes'].shape[0] > 0:
            keep_indices = torchvision.ops.nms(
                filtered_predictions['boxes'],
                filtered_predictions['scores'],
                self.nms_threshold
            )
            filtered_predictions = {
                k: v[keep_indices] for k, v in filtered_predictions.items()
            }
        
        # Convert masks to binary and resize to original dimensions
        if filtered_predictions['masks'].shape[0] > 0:
            # Convert mask logits to binary masks
            masks = (filtered_predictions['masks'] > 0).squeeze(1)  # Remove channel dim
            
            # Resize masks to original dimensions
            resize_transform = SegmentationTransforms.get_resize_transform(
                output_height=original_size[1],
                output_width=original_size[0]
            )
            
            resized_masks = []
            for mask in masks:
                transformed = resize_transform(
                    image=np.zeros_like(mask.cpu().numpy()),  # Dummy image
                    mask=mask.cpu().numpy()
                )
                resized_masks.append(transformed['mask'])
            
            filtered_predictions['masks'] = torch.tensor(
                np.stack(resized_masks),
                device=self.device,
                dtype=torch.bool
            )
            
            # Scale boxes to original dimensions
            scale_x = original_size[0] / self.config.data.input_size[1]
            scale_y = original_size[1] / self.config.data.input_size[0]
            boxes = filtered_predictions['boxes']
            scaled_boxes = torch.stack([
                boxes[:, 0] * scale_x,
                boxes[:, 1] * scale_y,
                boxes[:, 2] * scale_x,
                boxes[:, 3] * scale_y
            ], dim=1)
            filtered_predictions['boxes'] = scaled_boxes
            
        return filtered_predictions
    
    def visualize_predictions(
        self,
        image: Image.Image,
        predictions: Dict[str, torch.Tensor],
        score_threshold: float = 0.5
    ) -> Image.Image:
        """Visualize predictions on the image.
        
        Args:
            image: PIL Image to draw predictions on
            predictions: Dictionary of predictions from get_predictions
            score_threshold: Minimum confidence score for visualization
            
        Returns:
            PIL Image with predictions drawn on it
        """
        import cv2
        
        # Convert PIL to cv2
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Filter by score threshold
        keep = predictions['scores'] > score_threshold
        boxes = predictions['boxes'][keep].cpu().numpy()
        masks = predictions['masks'][keep].cpu().numpy()
        labels = predictions['labels'][keep].cpu().numpy()
        scores = predictions['scores'][keep].cpu().numpy()
        
        # Colors for visualization
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Add more colors if needed
        
        # Draw each instance
        for box, mask, label, score, color in zip(
            boxes, masks, labels, scores,
            [colors[i % len(colors)] for i in range(len(boxes))]
        ):
            # Draw mask
            mask_color = (*color, 127)  # Semi-transparent
            mask_overlay = np.zeros_like(image_cv, dtype=np.uint8)
            mask_overlay[mask] = mask_color
            cv2.addWeighted(mask_overlay, 0.5, image_cv, 1, 0, image_cv)
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
            
            # Draw label and score
            label_text = f"Class {label}: {score:.2f}"
            cv2.putText(
                image_cv, label_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)) 