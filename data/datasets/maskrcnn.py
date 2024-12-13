from PIL import Image, ImageDraw
import numpy as np
import torch
import torchvision.transforms.functional as TF

from .base import BaseDataset


class MaskRCNNDataset(BaseDataset):
    """Dataset for Mask R-CNN model."""
    def __init__(self, image_dir, coco_json, transform=None, input_size=(256, 256), augment=False):
        super().__init__(image_dir, coco_json, mask_dir=None, input_size=input_size, augment=augment)
        self.transform = transform
        
    def create_instance_masks(self, image_id):
        """Create binary instance masks for each annotation."""
        image_info = self.coco.imgs[image_id]
        annotations = self.get_annotations(image_id)
        
        boxes = []
        labels = []
        masks = []
        
        image_size = (image_info['width'], image_info['height'])
        
        for ann in annotations:
            try:
                # Bounding box in [x_min, y_min, x_max, y_max] format
                bbox = ann["bbox"]
                x_min, y_min, width, height = bbox
                boxes.append([x_min, y_min, x_min + width, y_min + height])
                
                # Class label (add 1 for background)
                labels.append(ann["category_id"] + 1)
                
                # Create binary mask
                mask = Image.new("L", image_size, 0)
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
                
        return boxes, labels, masks
        
    def __getitem__(self, idx):
        """Get image and corresponding target dictionary."""
        image_id = self.image_ids[idx]
        
        try:
            # Load image
            image = self.load_image(image_id)
            image = image.convert("RGB")  # Convert to RGB for Mask R-CNN
            
            # Get instance masks and annotations
            boxes, labels, masks = self.create_instance_masks(image_id)
            
            # Handle empty annotations
            if not boxes:
                boxes = torch.empty((0, 4), dtype=torch.float32)
                labels = torch.empty((0,), dtype=torch.int64)
                masks = torch.empty((0, *image.size[::-1]), dtype=torch.uint8)
            else:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)
                masks = torch.tensor(np.stack(masks), dtype=torch.uint8)
            
            # Create target dictionary
            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": torch.tensor([image_id], dtype=torch.int64)
            }
            
            # Apply transformations
            if self.transform:
                image, target = self.transform(image, target)
            else:
                image = TF.to_tensor(image)
            
            return image, target
            
        except Exception as e:
            print(f"Error processing item {idx} (image_id: {image_id}): {e}")
            raise