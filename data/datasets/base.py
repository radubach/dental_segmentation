from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import os
from configs.config import Config

class BaseDataset(Dataset):
    """Base class for dental image datasets using COCO format annotations.
    
    This class provides core functionality for loading images and annotations in COCO format,
    which is a standardized format for object detection, segmentation, and keypoint annotations.
    COCO (Common Objects in Context) format stores annotations in JSON files with specific structure
    for images, categories, and annotations.
    
    Child classes must implement:
        - __getitem__(self, idx): Returns tuple of (image, target) for training
            - image: Usually a transformed PIL Image or tensor
            - target: Dictionary containing annotations in format required by model
    
    Args:
        config: Configuration object containing dataset parameters
        is_training: Whether this dataset is for training. Default: True
    
    The COCO annotation format expects:
        - images: List of dicts with keys: 'id', 'width', 'height', 'file_name'
        - annotations: List of dicts with keys: 'id', 'image_id', 'category_id', 
          and task-specific keys (e.g., 'bbox', 'segmentation', 'keypoints')
        - categories: List of dicts with keys: 'id', 'name'
    """
    def __init__(self, config: Config, is_training: bool = True):
        # Validate input size
        if not isinstance(config.data.input_size, tuple) or len(config.data.input_size) != 2:
            raise ValueError("input_size in config must be a tuple of (height, width)")
            
        self.image_dir = config.data.image_dir
        self.mask_dir = config.data.mask_dir
        self.coco = COCO(config.data.coco_json)
        self.input_size = config.data.input_size
        self.augment = config.data.augment if is_training else False
        
        # Get all image IDs
        self.image_ids = list(sorted(self.coco.imgs.keys()))
        print(f"Dataset initialized with {len(self.image_ids)} images.")
        
    def load_image(self, image_id):
        """Load image given image ID."""
        image_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert("L")  # Grayscale
        
        # Verify dimensions match (optional validation)
        if (image.height != image_info['height'] or 
            image.width != image_info['width']):
            print(f"Warning: Image {image_id} dimensions mismatch. "
                  f"Annotation: {image_info['height']}x{image_info['width']}, "
                  f"Actual: {image.height}x{image.width}")
            
        return image
    
    def get_annotations(self, image_id):
        """Get annotations for a specific image."""
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        return self.coco.loadAnns(ann_ids)
    
    def __len__(self):
        return len(self.image_ids)