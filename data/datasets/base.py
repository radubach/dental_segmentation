from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import os

class BaseDataset(Dataset):
    """Base class for dental image datasets."""
    def __init__(self, image_dir, coco_json, mask_dir=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.coco = COCO(coco_json)
        
        # Get all image IDs
        self.image_ids = list(sorted(self.coco.imgs.keys()))
        print(f"Dataset initialized with {len(self.image_ids)} images.")
        
    def load_image(self, image_id):
        """Load image given image ID."""
        image_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        return Image.open(image_path).convert("L")  # Grayscale
    
    def get_annotations(self, image_id):
        """Get annotations for a specific image."""
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        return self.coco.loadAnns(ann_ids)
    
    def __len__(self):
        return len(self.image_ids)