import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Tuple, Union, Optional, Dict, Any
from configs.config import DataConfig


class SegmentationTransforms:
    """Transforms for segmentation tasks supporting both UNet and Mask R-CNN."""
    
    @staticmethod
    def get_training_transforms(
        config: DataConfig,
        is_instance_segmentation: bool = False
    ) -> A.Compose:
        """Get transforms for training.
        
        Args:
            config: Data configuration object containing transform parameters
            is_instance_segmentation: If True, configures transforms for Mask R-CNN style instance segmentation.
                                    If False, configures for UNet style semantic segmentation.
        """
        # Validate config parameters
        if not isinstance(config.input_size, tuple) or len(config.input_size) != 2:
            raise ValueError("input_size must be a tuple of (height, width)")
        
        if config.normalize_mean is None or config.normalize_std is None:
            raise ValueError("normalize_mean and normalize_std must be specified")

        transforms_list = [
            # Always resize first
            A.Resize(height=config.input_size[0], width=config.input_size[1]),
            
            # Add augmentations if requested
            *([
                A.RandomBrightnessContrast(
                    brightness_limit=config.brightness_limit,
                    contrast_limit=config.contrast_limit,
                    p=config.augment_prob
                ),
                A.GaussNoise(var_limit=config.noise_limit, p=config.augment_prob),
                A.Rotate(limit=config.rotation_limit, p=config.augment_prob),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.0,
                    rotate_limit=0,
                    p=config.augment_prob
                ),
            ] if config.augment else []),
            
            A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
            ToTensorV2()
        ]
        
        # Configure for either instance or semantic segmentation
        if is_instance_segmentation:
            return A.Compose(
                transforms_list,
                bbox_params=A.BboxParams(
                    format='pascal_voc',
                    min_area=0,
                    min_visibility=0,
                    label_fields=['labels']
                ),
                additional_targets={'masks': 'masks'}
            )
        else:
            # For semantic segmentation, convert masks to float32
            transforms_list.insert(-1, A.Lambda(
                image=None,
                masks=lambda x, **kwargs: x.astype(np.float32)
            ))
            return A.Compose(transforms_list)
    
    @staticmethod
    def get_inference_transforms(
        config: DataConfig,
        is_instance_segmentation: bool = False
    ) -> A.Compose:
        """Get transforms for inference."""
        transforms_list = [
            A.Resize(height=config.input_size[0], width=config.input_size[1]),
            A.Normalize(mean=config.normalize_mean, std=config.normalize_std),
            ToTensorV2()
        ]
        
        # Configure for either instance or semantic segmentation
        if is_instance_segmentation:
            return A.Compose(
                transforms_list,
                bbox_params=A.BboxParams(
                    format='pascal_voc',
                    min_area=0,
                    min_visibility=0,
                    label_fields=['labels']
                ),
                additional_targets={'masks': 'masks'}
            )
        else:
            # For semantic segmentation, convert masks to float32
            transforms_list.insert(-1, A.Lambda(
                image=None,
                masks=lambda x, **kwargs: x.astype(np.float32)
            ))
            return A.Compose(transforms_list)
    
    @staticmethod
    def get_resize_transform(
        output_height: int,
        output_width: int
    ) -> A.Compose:
        """Get transform for resizing outputs back to original size."""
        return A.Compose([
            A.Resize(height=output_height, width=output_width)
        ])
    
    @staticmethod
    def get_sliding_window_transforms() -> A.Compose:
        """Get transforms for sliding window inference on full-size images."""
        return A.Compose([
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])