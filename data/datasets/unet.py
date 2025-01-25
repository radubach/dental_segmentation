from PIL import Image, ImageDraw
import numpy as np
import os
import cv2
import torch

from .base import BaseDataset
from ..transforms import SegmentationTransforms

class UNetDataset(BaseDataset):
    def __init__(self, image_dir, mask_dir, coco_json, input_size=(256, 256), augment=False, use_cache=True):
        super().__init__(image_dir, coco_json, mask_dir, input_size, augment)
        
        # Validate input size
        if not isinstance(input_size, tuple) or len(input_size) != 2:
            raise ValueError("input_size must be a tuple of (height, width)")    

        # Single cache initialization
        self._mask_cache = {} if use_cache else None
        
        self.transform = SegmentationTransforms.get_training_transforms(
            input_height=input_size[0],
            input_width=input_size[1],
            augment=augment
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

    def __getitem__(self, idx):
        """Get image and corresponding segmentation mask."""
        image_id = self.image_ids[idx]
        
        try:
            # Load original size image and mask
            image = self.load_image(image_id)
            mask = self.get_mask(image_id)

            # Convert to numpy arrays
            image_array = np.array(image)
            mask_array = np.array(mask)

            # Add channel dimension to image if needed
            if len(image_array.shape) == 2:
                image_array = np.expand_dims(image_array, axis=-1)

            # Apply transforms (including resize)
            transformed = self.transform(image=image_array, mask=mask_array)
            
            # # Convert to tensors of correct type
            image_tensor = transformed['image']  # Should already be float
            mask_tensor = transformed['mask'].long()  # Convert to LongTensor
            
            return image_tensor, mask_tensor
            
        except Exception as e:
            print(f"Error processing item {idx} (image_id: {image_id}): {e}")
            raise