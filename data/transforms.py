# src/data/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.2),
                A.RandomRotate90(p=0.2),
            ] if augment else []),
            
            # Always end with tensor conversion
            A.Normalize(
                mean=[0.5],  # For grayscale
                std=[0.5]
            ),
            ToTensorV2()
        ]
        
        return A.Compose(transforms_list)
    
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