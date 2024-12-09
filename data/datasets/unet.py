from PIL import Image, ImageDraw
import numpy as np
import os

from .base import BaseDataset
from ..transforms import SegmentationTransforms

class UNetDataset(BaseDataset):
    """Dataset for UNet segmentation model."""
    def __init__(self, image_dir, mask_dir, coco_json, input_size=(256, 256), augment=False):
        super().__init__(image_dir, coco_json, mask_dir, input_size, augment)
        
        # Create transforms using SegmentationTransforms
        self.transform = SegmentationTransforms.get_training_transforms(
            input_height=input_size[0],
            input_width=input_size[1],
            augment=augment
        )

    def load_mask(self, image_id):
        """Load existing mask file."""
        image_info = self.coco.imgs[image_id]
        mask_name = os.path.splitext(image_info['file_name'])[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        return Image.open(mask_path)

    def __getitem__(self, idx):
        """Get image and corresponding segmentation mask."""
        image_id = self.image_ids[idx]
        
        # Load image and mask
        image = self.load_image(image_id)
        mask = self.load_mask(image_id)

        # Convert PIL images to numpy arrays for albumentations
        image_array = np.array(image)
        mask_array = np.array(mask)

        # Apply transforms
        transformed = self.transform(image=image_array, mask=mask_array)
        
        # Convert to tensors of correct type
        image_tensor = transformed['image']  # Should already be float
        mask_tensor = transformed['mask'].long()  # Convert to LongTensor
        
        return image_tensor, mask_tensor