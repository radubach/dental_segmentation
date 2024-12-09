from PIL import Image, ImageDraw
import numpy as np
import os

from .base import BaseDataset

class UNetDataset(BaseDataset):
    """Dataset for UNet segmentation model."""
    def __init__(self, image_dir, mask_dir, coco_json, transform=None):
        super().__init__(image_dir, coco_json, mask_dir)
        self.transform = transform

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

        # Apply transforms if specified
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask