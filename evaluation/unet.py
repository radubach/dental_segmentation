from .base import BaseEvaluator
import torch
import numpy as np
from PIL import Image
from data.transforms import SegmentationTransforms
from configs.config import Config
from typing import List

class UNetEvaluator(BaseEvaluator):
    """Evaluator for UNet segmentation model predictions.
    
    This class handles the evaluation pipeline for a UNet model, including:
    - Loading and preprocessing images
    - Generating model predictions
    - Post-processing predictions into instance masks and bounding boxes
    - Resizing outputs back to original image dimensions
    
    Args:
        model: Trained UNet model for generating predictions
        config: Configuration object containing model and data parameters
    
    Example:
        >>> evaluator = UNetEvaluator(model, config)
        >>> masks, boxes = evaluator.get_predictions(image_id)
        >>> # masks.shape: (32, H, W) boolean array of instance masks
        >>> # boxes.shape: (32, 4) array of [x1, y1, x2, y2] coordinates
    """
    def __init__(self, model, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device(config.model.device)
        self.model.to(self.device)
        self.model.eval()
        
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a single image for model input."""
        transform = SegmentationTransforms.get_inference_transforms(
            self.config.data,
            is_instance_segmentation=False
        )
        transformed = transform(image=np.array(image))
        return transformed['image'].unsqueeze(0).to(self.device)
        
    def postprocess_output(self, output: torch.Tensor) -> np.ndarray:
        """Convert model output to segmentation mask."""
        return output.argmax(dim=1).squeeze().cpu().numpy()
        
    def predict(self, image: Image.Image, use_sliding_window: bool = False) -> np.ndarray:
        """Generate predictions for a single image.
        
        Args:
            image: PIL Image to process
            use_sliding_window: Whether to use sliding window inference
            
        Returns:
            numpy array of shape (H, W) containing class predictions
        """
        # Get original size for resizing back
        original_size = image.size[::-1]  # PIL size is (W, H)
        
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Get model prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            
        # Convert output to numpy array
        pred = self.postprocess_output(output)
        
        # Resize back to original size if needed
        if pred.shape != original_size:
            transform = SegmentationTransforms.get_resize_transform(
                output_height=original_size[0],
                output_width=original_size[1]
            )
            transformed = transform(image=pred)
            pred = transformed['image']
            
        return pred
        
    def predict_batch(self, images: List[Image.Image]) -> List[np.ndarray]:
        """Generate predictions for a batch of images."""
        return [self.predict(image) for image in images]
        
    def infer_boxes(self, masks: np.ndarray) -> np.ndarray:
        """Infer bounding boxes from instance masks."""
        boxes = np.zeros((masks.shape[0], 4), dtype=float)
        
        for i in range(masks.shape[0]):
            mask = masks[i]
            if mask.any():  # If mask contains any True values
                # Find the bounding box of the mask
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                y1, y2 = np.where(rows)[0][[0, -1]]
                x1, x2 = np.where(cols)[0][[0, -1]]
                boxes[i] = [x1, y1, x2, y2]
                
        return boxes 