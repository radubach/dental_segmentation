import pytest
import numpy as np
import torch
import albumentations as A
from PIL import Image
from data.transforms import SegmentationTransforms
from configs.config import DataConfig

@pytest.fixture
def config():
    """Create a test configuration."""
    config = DataConfig()
    config.input_size = (256, 256)
    config.normalize_mean = (0.5,)
    config.normalize_std = (0.5,)
    config.augment = True
    config.augment_prob = 0.5
    config.rotation_limit = 10
    config.brightness_limit = 0.2
    config.contrast_limit = 0.2
    config.noise_limit = (10.0, 50.0)
    return config

def test_training_transforms_semantic(config):
    """Test training transforms for semantic segmentation (UNet)."""
    transform = SegmentationTransforms.get_training_transforms(
        config=config,
        is_instance_segmentation=False
    )
    
    # Create test data
    image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)  # 2D array for grayscale
    mask = np.random.randint(0, 2, (128, 128), dtype=np.uint8)  # 2D semantic mask
    
    # Apply transform
    result = transform(image=image, mask=mask)
    
    # Check outputs
    assert isinstance(result['image'], torch.Tensor)
    assert isinstance(result['mask'], torch.Tensor)
    assert result['image'].shape == (1, 256, 256)  # Resized single channel
    assert result['mask'].shape == (256, 256)
    assert result['image'].dtype == torch.float32
    assert result['mask'].dtype == torch.float32

def test_training_transforms_instance(config):
    """Test training transforms for instance segmentation (Mask R-CNN)."""
    transform = SegmentationTransforms.get_training_transforms(
        config=config,
        is_instance_segmentation=True
    )
    
    # Create test data with bounding box
    image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)          # 2D grayscale image
    num_instances = 3  # example number of instances
    mask = np.random.randint(0, 2, (128, 128, num_instances), dtype=np.uint8)  # Binary masks for each instance
    
    # Apply transform
    result = transform(
        image=image,
        masks=mask,
        bboxes=[[30, 30, 60, 60]],
        labels=[1]
    )
    
    assert isinstance(result['image'], torch.Tensor)
    assert isinstance(result['masks'], torch.Tensor)
    assert len(result['bboxes']) == 1
    assert len(result['labels']) == 1

def test_empty_mask_handling(config):
    """Test handling of empty masks."""
    transform = SegmentationTransforms.get_training_transforms(
        config=config,
        is_instance_segmentation=False
    )
    
    # Create test data with empty mask
    image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.uint8)  # Use mask for semantic segmentation
    
    result = transform(image=image, mask=mask)  # Use mask instead of masks
    
    assert result['mask'].sum() == 0  # Use mask instead of masks
    assert result['mask'].shape == (256, 256)

def test_boundary_conditions(config):
    """Test handling of extreme image sizes."""
    transform = SegmentationTransforms.get_training_transforms(
        config=config,
        is_instance_segmentation=False
    )
    
    # Test very small image
    tiny_image = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    tiny_mask = np.zeros((32, 32), dtype=np.uint8)  # Use mask for semantic segmentation
    tiny_result = transform(image=tiny_image, mask=tiny_mask)  # Use mask instead of masks
    assert tiny_result['image'].shape == (1, 256, 256)
    
    # Test very large image
    large_image = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
    large_mask = np.zeros((1024, 1024), dtype=np.uint8)  # Use mask for semantic segmentation
    large_result = transform(image=large_image, mask=large_mask)  # Use mask instead of masks
    assert large_result['image'].shape == (1, 256, 256)

def test_transform_consistency(config):
    """Test that transforms maintain alignment between image and mask."""
    transform = SegmentationTransforms.get_training_transforms(
        config=config,
        is_instance_segmentation=False
    )
    
    # Create test data with specific pattern
    image = np.zeros((128, 128), dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.uint8)  # Use mask for semantic segmentation
    
    # Create a square in both image and mask
    image[32:96, 32:96] = 255
    mask[32:96, 32:96] = 1
    
    # Apply transform multiple times to test consistency
    for _ in range(10):
        result = transform(image=image, mask=mask)  # Use mask instead of masks
        # Check that non-zero pixels in mask correspond to high-value pixels in image
        image_binary = result['image'].squeeze() > 0
        mask_binary = result['mask'] > 0  # Use mask instead of masks
        # Calculate IoU to ensure alignment
        intersection = (image_binary & mask_binary).sum()
        union = (image_binary | mask_binary).sum()
        iou = intersection / (union + 1e-6)
        assert iou > 0.5  # Allow some variation due to interpolation

def test_invalid_config_handling():
    """Test handling of invalid configuration parameters."""
    invalid_config = DataConfig()
    invalid_config.input_size = (256,)  # Invalid size tuple
    
    with pytest.raises(ValueError):
        SegmentationTransforms.get_training_transforms(invalid_config)

def test_multi_class_mask_handling(config):
    """Test handling of masks with multiple classes."""
    transform = SegmentationTransforms.get_training_transforms(
        config=config,
        is_instance_segmentation=False
    )
    
    # Create test data with multiple classes
    image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.uint8)  # Use mask for semantic segmentation
    mask[32:64, 32:64] = 1  # Class 1
    mask[64:96, 64:96] = 2  # Class 2
    
    result = transform(image=image, mask=mask)  # Use mask instead of masks
    
    # Check that classes are preserved
    unique_classes = torch.unique(result['mask'])  # Use mask instead of masks
    assert len(unique_classes) == 3  # Background (0) + 2 classes
    assert 0 in unique_classes
    assert 1 in unique_classes
    assert 2 in unique_classes

def test_augmentation_probability(config):
    """Test that augmentation probability is respected."""
    # Set augmentation probability to 1.0 for testing
    config.augment_prob = 1.0
    config.rotation_limit = 90
    
    transform = SegmentationTransforms.get_training_transforms(
        config=config,
        is_instance_segmentation=False
    )
    
    # Create test data with horizontal line
    image = np.zeros((128, 128), dtype=np.uint8)
    image[64, :] = 255
    mask = np.zeros((128, 128), dtype=np.uint8)  # Use mask for semantic segmentation
    mask[64, :] = 1
    
    # Apply transform and check if rotation was applied
    result = transform(image=image, mask=mask)  # Use mask instead of masks
    
    # The horizontal line should no longer be perfectly horizontal
    image_line = result['image'].squeeze().numpy()
    assert not np.all(image_line[config.input_size[0]//2, :] > 0)

def test_augmentation_application(config):
    """Test that augmentations are correctly applied when enabled."""
    transform = SegmentationTransforms.get_training_transforms(
        config=config,
        is_instance_segmentation=False
    )
    
    # Create test data
    image = np.ones((128, 128), dtype=np.uint8) * 128  # Gray image
    mask = np.zeros((128, 128), dtype=np.uint8)  # Use mask for semantic segmentation
    mask[32:64, 32:64] = 1
    
    # Apply transform multiple times to ensure augmentations are applied
    results = [
        transform(image=image.copy(), mask=mask.copy())  # Use mask instead of masks
        for _ in range(10)
    ]
    
    # Check that we get some variation in the results
    images = torch.stack([r['image'] for r in results])
    assert not torch.allclose(images[0], images[1])  # Images should be different

def test_inference_transforms():
    """Test inference transforms for both segmentation types."""
    config = DataConfig()
    config.input_size = (64, 64)
    config.normalize_mean = (0.5,)
    config.normalize_std = (0.5,)
    
    # Test semantic segmentation transforms
    transform = SegmentationTransforms.get_inference_transforms(
        config=config,
        is_instance_segmentation=False
    )
    
    # Create test data
    image = np.random.randint(0, 255, (128, 128, 1), dtype=np.uint8)  # Add channel dimension
    
    # Apply transform
    result = transform(image=image)
    
    # Check outputs
    assert isinstance(result['image'], torch.Tensor)
    assert result['image'].shape == (1, 64, 64)
    assert result['image'].dtype == torch.float32
    
    # Test instance segmentation transforms
    transform = SegmentationTransforms.get_inference_transforms(
        config=config,
        is_instance_segmentation=True
    )
    
    # Apply transform with required label field
    result = transform(
        image=image,
        masks=np.zeros((128, 128, 1), dtype=np.uint8),  # Transpose to match image dimensions
        bboxes=[[30, 30, 60, 60]],
        labels=[1]
    )
    
    assert isinstance(result['image'], torch.Tensor)
    assert isinstance(result['masks'], torch.Tensor)
    assert result['image'].shape == (1, 64, 64)
    assert result['image'].dtype == torch.float32

def test_resize_transform():
    """Test the resize transform."""
    transform = SegmentationTransforms.get_resize_transform(32, 32)
    
    # Create test data
    image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    masks = np.zeros((1, 64, 64), dtype=np.uint8)  # Add instance dimension
    masks[0, 16:32, 16:32] = 1
    
    # Apply transform
    result = transform(image=image, masks=masks)
    
    # Check outputs
    assert result['image'].shape == (32, 32)
    assert result['masks'].shape == (1, 32, 32)  # Check instance dimension

def test_sliding_window_transforms():
    """Test transforms for sliding window inference."""
    transform = SegmentationTransforms.get_sliding_window_transforms()
    
    # Create test data
    image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    
    # Apply transform
    result = transform(image=image)
    
    # Check outputs
    assert isinstance(result['image'], torch.Tensor)
    assert result['image'].shape == (1, 128, 128)  # Should preserve size
    assert result['image'].dtype == torch.float32

def test_normalization_values(config):
    """Test that normalization is correctly applied."""
    transform = SegmentationTransforms.get_training_transforms(
        config=config,
        is_instance_segmentation=False
    )
    
    # Create test data
    image = np.ones((128, 128), dtype=np.uint8) * 128  # Gray image with value 128
    mask = np.zeros((128, 128), dtype=np.uint8)  # Use mask for semantic segmentation
    
    # Apply transform
    result = transform(image=image, mask=mask)  # Use mask instead of masks
    
    # Check that normalization was applied correctly
    # For mean=0.5, std=0.5:
    # (128/255 - 0.5) / 0.5 should be 0.0 approximately
    assert torch.abs(result['image'].mean()) < 0.1 