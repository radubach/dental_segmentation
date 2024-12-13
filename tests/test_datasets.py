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
                "area": 10000,
                "bbox": [100, 100, 100, 100]
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
    
    # Create a test image (gray square)
    test_image = Image.new('L', (512, 512), color=128)
    image_path = image_dir / "test_image_1.png"
    test_image.save(image_path)
    
    # Make sure COCO annotations match image size
    sample_coco_data["images"][0]["height"] = 512
    sample_coco_data["images"][0]["width"] = 512

    # Save COCO JSON
    coco_path = tmp_path / "annotations.json"
    with open(coco_path, 'w') as f:
        json.dump(sample_coco_data, f)
        
    return {
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "coco_json": str(coco_path)
    }

class TestBaseDataset:
    def test_initialization(self, temp_dataset):
        dataset = BaseDataset(
            image_dir=temp_dataset["image_dir"],
            coco_json=temp_dataset["coco_json"]
        )
        assert len(dataset) == 1
        assert dataset.image_ids == [1]
        
    def test_load_image(self, temp_dataset):
        dataset = BaseDataset(
            image_dir=temp_dataset["image_dir"],
            coco_json=temp_dataset["coco_json"]
        )
        image = dataset.load_image(1)
        assert isinstance(image, Image.Image)
        assert image.mode == "L"  # Should be grayscale
        assert image.size == (512, 512)
        
    def test_get_annotations(self, temp_dataset):
        dataset = BaseDataset(
            image_dir=temp_dataset["image_dir"],
            coco_json=temp_dataset["coco_json"]
        )
        annotations = dataset.get_annotations(1)
        assert len(annotations) == 1
        assert annotations[0]["category_id"] == 1

class TestUNetDataset:
    def test_initialization(self, temp_dataset):
        dataset = UNetDataset(
            image_dir=temp_dataset["image_dir"],
            mask_dir=temp_dataset["mask_dir"],
            coco_json=temp_dataset["coco_json"]
        )
        assert len(dataset) == 1
        assert hasattr(dataset, 'transform')
        
    def test_create_mask(self, temp_dataset):
        dataset = UNetDataset(
            image_dir=temp_dataset["image_dir"],
            mask_dir=temp_dataset["mask_dir"],
            coco_json=temp_dataset["coco_json"]
        )
        mask = dataset.create_mask(1)
        assert isinstance(mask, Image.Image)
        assert mask.size == dataset.input_size
        
    def test_getitem(self, temp_dataset):
        dataset = UNetDataset(
            image_dir=temp_dataset["image_dir"],
            mask_dir=temp_dataset["mask_dir"],
            coco_json=temp_dataset["coco_json"]
        )
        image, mask = dataset[0]
        
        # Check types
        assert torch.is_tensor(image)
        assert torch.is_tensor(mask)
        
        # Check shapes
        assert image.shape[0] == 1  # Single channel
        assert image.shape[1:] == dataset.input_size
        assert mask.shape == dataset.input_size
        
        # Check value ranges
        assert image.dtype == torch.float32
        assert mask.dtype == torch.int64
        assert mask.min() >= 0
        assert mask.max() <= len(dataset.coco.cats) + 1

    def test_invalid_image_handling(self, temp_dataset):
        # Remove the image file to test error handling
        os.remove(os.path.join(temp_dataset["image_dir"], "test_image_1.png"))
        
        dataset = UNetDataset(
            image_dir=temp_dataset["image_dir"],
            mask_dir=temp_dataset["mask_dir"],
            coco_json=temp_dataset["coco_json"]
        )
        
        with pytest.raises(FileNotFoundError):
            _ = dataset[0]

    def test_empty_annotations(self, temp_dataset, sample_coco_data):
        # Modify annotations to be empty
        sample_coco_data["annotations"] = []
        coco_path = temp_dataset["coco_json"]
        
        with open(coco_path, 'w') as f:
            json.dump(sample_coco_data, f)
            
        dataset = UNetDataset(
            image_dir=temp_dataset["image_dir"],
            mask_dir=temp_dataset["mask_dir"],
            coco_json=coco_path
        )
        
        image, mask = dataset[0]
        assert torch.all(mask == 0)  # Mask should be all background

    def test_invalid_input_size(self, temp_dataset):
        with pytest.raises(ValueError):
            _ = UNetDataset(
                image_dir=temp_dataset["image_dir"],
                mask_dir=temp_dataset["mask_dir"],
                coco_json=temp_dataset["coco_json"],
                input_size=256  # Invalid input size (should be tuple)
            )

    def test_augmentations(self, temp_dataset):
        dataset = UNetDataset(
            image_dir=temp_dataset["image_dir"],
            mask_dir=temp_dataset["mask_dir"],
            coco_json=temp_dataset["coco_json"],
            augment=True
        )
        
        # Get the same item multiple times to check if augmentations are applied
        images = [dataset[0][0] for _ in range(5)]
        
        # Check if at least some images are different (augmented)
        # Note: This test could theoretically fail even with working augmentations,
        # but it's very unlikely
        assert not all(torch.all(images[0] == img) for img in images[1:])

def test_mask_cache(temp_dataset):
    dataset = UNetDataset(
        image_dir=temp_dataset["image_dir"],
        mask_dir=temp_dataset["mask_dir"],
        coco_json=temp_dataset["coco_json"]
    )
    
    # Access mask first time
    _ = dataset.get_mask(1)
    assert 1 in dataset._mask_cache
    
    # Check if cached version is used
    cached_mask = dataset._mask_cache[1]
    loaded_mask = dataset.create_mask(1)
    assert np.array_equal(np.array(cached_mask), np.array(loaded_mask))