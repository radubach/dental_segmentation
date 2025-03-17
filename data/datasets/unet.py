from PIL import Image, ImageDraw
import numpy as np
import os
import cv2
import torch
from typing import Tuple, Dict, Any

from .base import BaseDataset
from ..transforms import SegmentationTransforms
from configs.config import Config

class UNetDataset(BaseDataset):
    """A dataset class specifically designed for training U-Net segmentation models.

    This dataset handles semantic segmentation data where each pixel in the mask is assigned
    a category ID. It's particularly suited for U-Net architecture as it:
    - Generates one-hot encoded masks where background is 0 and other classes start from 1
    - Maintains aspect ratio and handles both single and multi-polygon segmentation masks
    - Supports caching of processed masks for improved performance
    - Provides synchronized transformations for both images and their corresponding masks

    Args:
        config: Configuration object containing dataset parameters
        use_cache: Whether to cache processed masks in memory. Default: True
        is_training: Whether this dataset is for training. Default: True

    Note:
        The class expects annotations in COCO format and converts segmentation polygons
        into pixel-wise masks. Category IDs in the output masks are offset by +1 to
        reserve 0 for background, which is typical in U-Net implementations.
    """
    def __init__(self, config: Config, use_cache: bool = True, is_training: bool = True):
        super().__init__(config, is_training)
        
        # Single cache initialization
        self._mask_cache = {} if use_cache else None
        self.config = config
        
        # Get appropriate transforms
        self.transform = (
            SegmentationTransforms.get_training_transforms(config.data, is_instance_segmentation=False)
            if is_training else
            SegmentationTransforms.get_inference_transforms(config.data, is_instance_segmentation=False)
        )

    def create_mask(self, image_id):
        """Create mask from annotations at original size."""
        image_info = self.coco.imgs[image_id]
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

        annotations = self.get_annotations(image_id)
        
        if not annotations:
            print(f"Warning: No annotations found for image {image_id}")

        for ann in annotations:
            try:
                segmentation = ann["segmentation"]
                category_id = ann["category_id"] + 1  # Add 1 to reserve 0 for background

                if isinstance(segmentation, list) and segmentation:
                    if isinstance(segmentation[0], list):  # Multi-polygon case
                        for poly in segmentation:
                            if len(poly) >= 6:
                                points = np.array(poly).reshape(-1, 2).astype(np.int32)
                                cv2.fillPoly(mask, [points], color=category_id)
                    else:  # Single polygon case
                        if len(segmentation) >= 6:
                            points = np.array(segmentation).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(mask, [points], color=category_id)
            except Exception as e:
                print(f"Error processing annotation for image {image_id}: {e}")
                continue

        return mask

    def get_mask(self, image_id):
        """Get mask from cache or create new."""
        if self._mask_cache is not None and image_id in self._mask_cache:
            return self._mask_cache[image_id]
            
        mask = self.create_mask(image_id)
        
        if self._mask_cache is not None:
            self._mask_cache[image_id] = mask
            
        return mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset.
        
        Returns:
            tuple: (image, mask) where:
                - image: torch.Tensor of shape (C, H, W)
                - mask: torch.Tensor of shape (H, W) with values 0-32
                  representing class IDs (singular because it's one semantic mask)
        """
        image_id = self.image_ids[idx]
        image = self.load_image(image_id)
        mask = self.get_mask(image_id)  # singular because it's one semantic mask
        
        transformed = self.transform(
            image=np.array(image),
            mask=mask  # singular for semantic segmentation
        )
        
        return transformed['image'], transformed['mask']