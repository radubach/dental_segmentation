import pytest
import os
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from data.datasets.maskrcnn import MaskRCNNDataset
from configs.config import Config

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

def test_initialization(config):
    dataset = MaskRCNNDataset(config)
    assert dataset.image_ids == [1]
    assert dataset.input_size == (256, 256)

def test_get_image_info(config):
    dataset = MaskRCNNDataset(config)
    image_info = dataset.get_image_info(1)
    assert image_info["file_name"] == "test_image_1.png"
    assert image_info["height"] == 512
    assert image_info["width"] == 512

def test_load_mask(config):
    dataset = MaskRCNNDataset(config)
    mask, class_ids = dataset.load_mask(1)
    assert isinstance(mask, np.ndarray)
    assert isinstance(class_ids, np.ndarray)
    assert mask.shape[2] == 1  # One instance
    assert class_ids.shape == (1,)  # One instance
    assert class_ids[0] == 1  # Category ID

def test_getitem(config):
    dataset = MaskRCNNDataset(config)
    sample = dataset[0]
    
    assert isinstance(sample, dict)
    assert all(k in sample for k in ["image", "mask", "class_ids", "bbox"])
    
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["mask"], torch.Tensor)
    assert isinstance(sample["class_ids"], torch.Tensor)
    assert isinstance(sample["bbox"], torch.Tensor)
    
    assert sample["image"].shape == (3, 256, 256)  # RGB image
    assert len(sample["mask"].shape) == 3  # (H, W, N)
    assert sample["class_ids"].shape == (sample["mask"].shape[2],)
    assert sample["bbox"].shape[1] == 4  # [x1, y1, x2, y2]

def test_empty_annotations(config, sample_coco_data):
    # Modify annotations to be empty
    sample_coco_data["annotations"] = []
    with open(config.data.coco_json, 'w') as f:
        json.dump(sample_coco_data, f)
        
    dataset = MaskRCNNDataset(config)
    sample = dataset[0]
    
    assert sample["mask"].shape[2] == 0  # No instances
    assert sample["class_ids"].shape[0] == 0
    assert sample["bbox"].shape[0] == 0

def test_multiple_instances(config, sample_coco_data):
    # Add another instance
    sample_coco_data["annotations"].append({
        "id": 2,
        "image_id": 1,
        "category_id": 1,
        "segmentation": [[300, 300, 400, 300, 400, 400, 300, 400]],
        "bbox": [300, 300, 100, 100],
        "area": 10000
    })
    
    with open(config.data.coco_json, 'w') as f:
        json.dump(sample_coco_data, f)
        
    dataset = MaskRCNNDataset(config)
    sample = dataset[0]
    
    assert sample["mask"].shape[2] == 2  # Two instances
    assert sample["class_ids"].shape[0] == 2
    assert sample["bbox"].shape[0] == 2

def test_augmentations(config):
    config.data.augment = True
    dataset = MaskRCNNDataset(config)
    
    # Get multiple samples to check augmentations
    samples = [dataset[0] for _ in range(5)]
    images = torch.stack([s["image"] for s in samples])
    
    # Check that augmentations are being applied
    assert not torch.allclose(images[0], images[1])

def test_invalid_image(config):
    # Remove the image file
    os.remove(os.path.join(config.data.image_dir, "test_image_1.png"))
    
    dataset = MaskRCNNDataset(config)
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]

def test_invalid_input_size(config):
    config.data.input_size = 256  # Invalid input size (should be tuple)
    with pytest.raises(ValueError):
        _ = MaskRCNNDataset(config)