import pytest
import os
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from data.datasets import MaskRCNNDataset

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
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "segmentation": [[300, 300, 400, 300, 400, 400, 300, 400]],  # Another square
                "bbox": [300, 300, 100, 100],
                "area": 10000
            }
        ],
        "categories": [
            {"id": 1, "name": "tooth1", "supercategory": "anatomy"},
            {"id": 2, "name": "tooth2", "supercategory": "anatomy"}
        ]
    }

@pytest.fixture
def temp_dataset(tmp_path, sample_coco_data):
    """Create a temporary dataset structure with sample data."""
    # Create directories
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    
    # Create a test image (gray square)
    test_image = Image.new('L', (512, 512), color=128)
    image_path = image_dir / "test_image_1.png"
    test_image.save(image_path)
    
    # Save COCO JSON
    coco_path = tmp_path / "annotations.json"
    with open(coco_path, 'w') as f:
        json.dump(sample_coco_data, f)
        
    return {
        "image_dir": str(image_dir),
        "coco_json": str(coco_path)
    }

class TestMaskRCNNDataset:
    def test_initialization(self, temp_dataset):
        dataset = MaskRCNNDataset(
            image_dir=temp_dataset["image_dir"],
            coco_json=temp_dataset["coco_json"]
        )
        assert len(dataset) == 1
        assert dataset.image_ids == [1]

    def test_create_instance_masks(self, temp_dataset):
        dataset = MaskRCNNDataset(
            image_dir=temp_dataset["image_dir"],
            coco_json=temp_dataset["coco_json"]
        )
        boxes, labels, masks = dataset.create_instance_masks(1)
        
        # Should have two instances
        assert len(boxes) == 2
        assert len(labels) == 2
        assert len(masks) == 2
        
        # Check box format and values
        assert len(boxes[0]) == 4  # [x1, y1, x2, y2]
        assert boxes[0] == [100, 100, 200, 200]  # First box
        
        # Check labels
        assert labels[0] == 2  # category_id + 1
        assert labels[1] == 3
        
        # Check masks
        assert masks[0].shape == (512, 512)
        assert masks[0].dtype == np.uint8
        assert np.any(masks[0] == 1)  # Should have some positive pixels

    def test_getitem(self, temp_dataset):
        dataset = MaskRCNNDataset(
            image_dir=temp_dataset["image_dir"],
            coco_json=temp_dataset["coco_json"]
        )
        image, target = dataset[0]
        
        # Check image
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 512, 512)  # RGB
        
        # Check target structure
        assert isinstance(target, dict)
        assert all(k in target for k in ['boxes', 'labels', 'masks', 'image_id'])
        
        # Check target contents
        assert target['boxes'].shape == (2, 4)
        assert target['labels'].shape == (2,)
        assert target['masks'].shape == (2, 512, 512)
        assert target['image_id'].item() == 1

    def test_empty_annotations(self, temp_dataset, sample_coco_data):
        # Modify annotations to be empty
        sample_coco_data["annotations"] = []
        coco_path = temp_dataset["coco_json"]
        
        with open(coco_path, 'w') as f:
            json.dump(sample_coco_data, f)
            
        dataset = MaskRCNNDataset(
            image_dir=temp_dataset["image_dir"],
            coco_json=coco_path
        )
        
        image, target = dataset[0]
        assert target['boxes'].shape == (0, 4)
        assert target['labels'].shape == (0,)
        assert target['masks'].shape == (0, 512, 512)

    def test_invalid_image_handling(self, temp_dataset):
        # Remove the image file to test error handling
        os.remove(os.path.join(temp_dataset["image_dir"], "test_image_1.png"))
        
        dataset = MaskRCNNDataset(
            image_dir=temp_dataset["image_dir"],
            coco_json=temp_dataset["coco_json"]
        )
        
        with pytest.raises(FileNotFoundError):
            _ = dataset[0]

    def test_transform(self, temp_dataset):
        # Test with a simple transform
        def dummy_transform(image, target):
            # Simple transform that doesn't modify the data
            return image, target

        dataset = MaskRCNNDataset(
            image_dir=temp_dataset["image_dir"],
            coco_json=temp_dataset["coco_json"],
            transform=dummy_transform
        )
        
        image, target = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(target, dict)