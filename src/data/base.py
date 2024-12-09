from torch.utils.data import Dataset
from pycocotools.coco import COCO

# src/data/datasets/base.py
class BaseDataset(Dataset):
    """Base class for dental image datasets."""
    def __init__(self, image_dir, coco_json):
        self.image_dir = image_dir
        self.coco = COCO(coco_json)
        
    def load_image(self, image_id):
        """Common image loading logic"""
        pass