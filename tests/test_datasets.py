import pytest
import os
import json
import numpy as np
print(np.__file__)
import torch
from PIL import Image
from pathlib import Path
import albumentations as A

from data.datasets import BaseDataset, UNetDataset
from data.transforms import SegmentationTransforms
from configs.config import Config, DataConfig


@pytest.fixture
def sample_coco_data():
    """Create a minimal COCO dataset structure for testing."""
    return {
        "images": [
            {
                "id": 1,
                "file_name": "test_image_1.png",
                "height": 512,
                "width": 512
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[100, 100, 200, 100, 200, 200, 100, 200]],  # Simple square
                "bbox": [100, 100, 100, 100],  # [x, y, width, height]
                "area": 10000
            }
        ],
        "categories": [
            {"id": 1, "name": "tooth", "supercategory": "anatomy"}
        ]
    }

@pytest.fixture
def temp_dataset(tmp_path, sample_coco_data):
    """Create a temporary dataset structure with sample data."""
    # Create directories
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    
    # Create a test image
    image = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    image.save(image_dir / "test_image_1.png")
    
    # Save COCO annotations
    coco_path = tmp_path / "annotations.json"
    with open(coco_path, 'w') as f:
        json.dump(sample_coco_data, f)
    
    return {
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "coco_json": str(coco_path)
    }

@pytest.fixture
def config(temp_dataset):
    """Create a config for testing."""
    config = Config()
    config.data.image_dir = temp_dataset["image_dir"]
    config.data.mask_dir = temp_dataset["mask_dir"]
    config.data.coco_json = temp_dataset["coco_json"]
    config.data.input_size = (256, 256)
    config.data.augment = False
    return config

class TestBaseDataset:
    def test_initialization(self, temp_dataset):
        dataset = BaseDataset(
            image_dir=temp_dataset["image_dir"],
            coco_json=temp_dataset["coco_json"],
            mask_dir=temp_dataset["mask_dir"],
            input_size=(256, 256),
            augment=False
        )
        assert dataset.image_ids == [1]
        assert dataset.input_size == (256, 256)
        
    def test_load_image(self, temp_dataset):
        dataset = BaseDataset(
            image_dir=temp_dataset["image_dir"],
            coco_json=temp_dataset["coco_json"],
            mask_dir=temp_dataset["mask_dir"],
            input_size=(256, 256),
            augment=False
        )
        image = dataset.load_image(1)
        assert isinstance(image, Image.Image)
        assert image.size == (512, 512)
        
    def test_get_annotations(self, temp_dataset):
        dataset = BaseDataset(
            image_dir=temp_dataset["image_dir"],
            coco_json=temp_dataset["coco_json"],
            mask_dir=temp_dataset["mask_dir"],
            input_size=(256, 256),
            augment=False
        )
        annotations = dataset.get_annotations(1)
        assert len(annotations) == 1
        assert annotations[0]["category_id"] == 1

class TestUNetDataset:
    def test_initialization(self, config):
        dataset = UNetDataset(config)
        assert dataset.image_ids == [1]
        assert dataset.input_size == (256, 256)
        
    def test_create_mask(self, config):
        dataset = UNetDataset(config)
        mask = dataset.create_mask(1)
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (512, 512)
        
    def test_getitem(self, config):
        dataset = UNetDataset(config)
        image, mask = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert image.shape == (1, 256, 256)  # Single channel
        assert mask.shape == (256, 256)
        
    def test_invalid_image_handling(self, config):
        # Remove the image file to test error handling
        os.remove(os.path.join(config.data.image_dir, "test_image_1.png"))
        
        dataset = UNetDataset(config)
        with pytest.raises(FileNotFoundError):
            _ = dataset[0]
            
    def test_empty_annotations(self, config, sample_coco_data):
        # Modify annotations to be empty
        sample_coco_data["annotations"] = []
        coco_path = config.data.coco_json
        
        with open(coco_path, 'w') as f:
            json.dump(sample_coco_data, f)
            
        dataset = UNetDataset(config)
        image, mask = dataset[0]
        assert torch.all(mask == 0)  # All background
        
    def test_invalid_input_size(self, config):
        config.data.input_size = 256  # Invalid input size (should be tuple)
        with pytest.raises(ValueError):
            _ = UNetDataset(config)
            
    def test_augmentations(self, config):
        config.data.augment = True
        dataset = UNetDataset(config)
        
        # Get multiple samples to ensure augmentations are being applied
        samples = [dataset[0] for _ in range(5)]
        images = torch.stack([s[0] for s in samples])
        
        # Check that we get some variation in the augmented images
        assert not torch.allclose(images[0], images[1])

def test_mask_cache(config):
    # Test with cache
    dataset_with_cache = UNetDataset(config, use_cache=True)
    mask1 = dataset_with_cache.get_mask(1)
    mask2 = dataset_with_cache.get_mask(1)
    assert mask1 is mask2  # Same object in memory
    
    # Test without cache
    dataset_no_cache = UNetDataset(config, use_cache=False)
    mask1 = dataset_no_cache.get_mask(1)
    mask2 = dataset_no_cache.get_mask(1)
    assert mask1 is not mask2  # Different objects in memory