import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class SegmentationTransforms:
    """Transforms for segmentation tasks."""
    
    @staticmethod
    def get_training_transforms(
        input_height: int = 256,
        input_width: int = 256,
        augment: bool = True
    ) -> A.Compose:
        """Get transforms for training."""
        transforms_list = [
            # Always resize first
            A.Resize(height=input_height, width=input_width),
            
            # Add augmentations if requested
            *([
                # A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.2
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,  # Only use shift
                    scale_limit=0.0,   # Disable scaling
                    rotate_limit=0,    # Disable rotation
                    p=0.5
                ),
            ] if augment else []),
            
            # Always end with tensor conversion
            A.Normalize(
                mean=[0.5],  
                std=[0.5]
            ),
            ToTensorV2()
        ]
        
        return A.Compose(
            transforms_list,
            is_check_shapes=True
        )
    
    @staticmethod
    def get_inference_transforms(
        input_height: int = 256,
        input_width: int = 256
    ) -> A.Compose:
        """Get transforms for inference (no augmentation)."""
        return A.Compose([
            A.Resize(height=input_height, width=input_width),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_sliding_window_transforms() -> A.Compose:
        """Get transforms for sliding window inference on full-size images."""
        return A.Compose([
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])