from PIL import Image, ImageDraw
import numpy as np
import torch
import torchvision.transforms.functional as TF
import albumentations as A
from typing import Dict, Any, Tuple, List
import cv2

from .base import BaseDataset
from ..transforms import SegmentationTransforms
from configs.config import Config

class MaskRCNNDataset(BaseDataset):
    """Dataset for Mask R-CNN model.
    
    This dataset class is specifically designed to work with Mask R-CNN, which requires
    instance segmentation data in a specific format. Unlike regular segmentation datasets,
    it provides:
    
    1. Instance Masks: Binary masks for each individual object instance, rather than
       a single semantic segmentation mask. Each mask has shape (H, W) where pixels
       belonging to the instance are 1 and background is 0.
       
    2. Bounding Boxes: Coordinates in [x_min, y_min, x_max, y_max] format for each
       instance, used by the Region Proposal Network (RPN) in Mask R-CNN.
       
    3. Labels: Class IDs for each instance, offset by +1 to reserve 0 for background
       class as required by Mask R-CNN.
    
    The dataset returns a tuple of (image, target) where target is a dictionary containing:
        - boxes: Tensor[N, 4] - Bounding boxes in (x1, y1, x2, y2) format
        - labels: Tensor[N] - Classification labels for each instance
        - masks: Tensor[N, H, W] - Binary masks for each instance
        - image_id: Tensor[1] - ID of the image
        
    Args:
        config: Configuration object containing dataset parameters
        is_training: Whether this dataset is for training. Default: True
    """
    def __init__(self, config: Config, is_training: bool = True):
        super().__init__(config, is_training)
        
        self.config = config
        self.transform = (
            SegmentationTransforms.get_training_transforms(config.data, is_instance_segmentation=True)
            if is_training else
            SegmentationTransforms.get_inference_transforms(config.data, is_instance_segmentation=True)
        )

    def get_image_info(self, image_id: int) -> Dict[str, Any]:
        """Get image information from COCO dataset."""
        return self.coco.imgs[image_id]

    def load_mask(self, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load instance masks and class IDs for an image.
        
        Returns:
            tuple: (masks, class_ids) where:
                - masks: [height, width, num_instances] binary array
                - class_ids: [num_instances] array of class IDs
        """
        image_info = self.get_image_info(image_id)
        annotations = self.get_annotations(image_id)
        
        # Initialize empty masks array
        masks = np.zeros((image_info['height'], image_info['width'], len(annotations)), dtype=np.uint8)
        class_ids = np.zeros(len(annotations), dtype=np.int32)
        
        for i, ann in enumerate(annotations):
            try:
                # Create binary mask
                mask = Image.new("L", (image_info['width'], image_info['height']), 0)
                draw = ImageDraw.Draw(mask)
                
                segmentation = ann["segmentation"]
                if isinstance(segmentation[0], list):  # Multi-polygon case
                    for poly in segmentation:
                        points = np.array(poly).reshape(-1, 2)
                        draw.polygon([tuple(p) for p in points], fill=1)
                else:  # Single polygon case
                    points = np.array(segmentation).reshape(-1, 2)
                    draw.polygon([tuple(p) for p in points], fill=1)
                
                masks[:, :, i] = np.array(mask, dtype=np.uint8)
                class_ids[i] = ann["category_id"]  # Use category_id as is
                
            except Exception as e:
                print(f"Error processing annotation for image {image_id}: {e}")
                continue
                
        return masks, class_ids

    def get_targets(self, image_id: int) -> Dict[str, Any]:
        """Create binary instance masks and other targets for an image."""
        image_info = self.get_image_info(image_id)
        annotations = self.get_annotations(image_id)
        
        boxes = []
        labels = []
        masks = []
        
        for ann in annotations:
            try:
                # Bounding box in [x_min, y_min, x_max, y_max] format
                bbox = ann["bbox"]
                x_min, y_min, width, height = bbox
                boxes.append([x_min, y_min, x_min + width, y_min + height])
                
                # Class label (add 1 to reserve 0 for background)
                labels.append(ann["category_id"] + 1)
                
                # Create binary mask
                mask = Image.new("L", (image_info['width'], image_info['height']), 0)
                draw = ImageDraw.Draw(mask)
                
                segmentation = ann["segmentation"]
                if isinstance(segmentation[0], list):  # Multi-polygon case
                    for poly in segmentation:
                        points = np.array(poly).reshape(-1, 2)
                        draw.polygon([tuple(p) for p in points], fill=1)
                else:  # Single polygon case
                    points = np.array(segmentation).reshape(-1, 2)
                    draw.polygon([tuple(p) for p in points], fill=1)
                
                masks.append(np.array(mask, dtype=np.uint8))
                
            except Exception as e:
                print(f"Error processing annotation for image {image_id}: {e}")
                continue

        # Return as dictionary
        return {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id
        }
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset.
        
        Returns:
            dict containing:
                - image: torch.Tensor of shape (C, H, W)
                - masks: torch.Tensor of shape (N, H, W) where N is number of instances
                - labels: torch.Tensor of shape (N,)
                - boxes: torch.Tensor of shape (N, 4)
        """
        image_id = self.image_ids[idx]
        image = self.load_image(image_id)
        
        # Get instance masks as (N, H, W) array
        instance_masks, labels = self.load_masks(image_id)
        
        # Apply transforms
        transformed = self.transform(
            image=np.array(image),
            masks=instance_masks  # plural because it's multiple binary masks
        )
        
        return {
            'image': transformed['image'],
            'masks': transformed['masks'],  # Keep plural for instance segmentation
            'labels': torch.as_tensor(labels),
            'boxes': self.masks_to_boxes(transformed['masks'])
        }

    def load_masks(self, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load instance masks for an image.
        
        Returns:
            tuple: (masks, labels) where:
                - masks: np.ndarray of shape (N, H, W) containing binary masks
                - labels: np.ndarray of shape (N,) containing class labels
        """
        annotations = self.get_annotations(image_id)
        image_info = self.get_image_info(image_id)
        
        # Initialize arrays
        masks = np.zeros((len(annotations), 
                         image_info['height'], 
                         image_info['width']), 
                        dtype=np.uint8)
        labels = np.zeros(len(annotations), dtype=np.int64)
        
        # Fill arrays
        for idx, ann in enumerate(annotations):
            masks[idx] = self.coco.annToMask(ann)
            labels[idx] = ann['category_id']
            
        return masks, labels