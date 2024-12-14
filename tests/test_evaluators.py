import pytest
import numpy as np
import torch
from PIL import Image
from unittest.mock import Mock, patch
from evaluation.base import BaseEvaluator
from evaluation.unet import UNetEvaluator

class MockDataset:
    def __init__(self):
        self.image_ids = [1, 2, 3]
        self.coco = Mock()
        # Mock image info
        self.coco.imgs = {
            1: {'height': 64, 'width': 64},
            2: {'height': 64, 'width': 64},
            3: {'height': 64, 'width': 64}
        }
    
    def get_annotations(self, image_id):
        # Return mock annotations for testing
        return [
            {
                'category_id': 0,  # First tooth (will be 1 in predictions)
                'segmentation': [[0, 0, 10, 0, 10, 10, 0, 10]],  # Simple polygon
                'bbox': [0, 0, 10, 10]
            },
            {
                'category_id': 1,  # Second tooth (will be 2 in predictions)
                'segmentation': [[20, 20, 30, 20, 30, 30, 20, 30]],
                'bbox': [20, 20, 10, 10]
            }
        ]
    
    def load_image(self, image_id):
        # Return a dummy grayscale image
        return np.zeros((64, 64), dtype=np.uint8)

class MockUNet(torch.nn.Module):
    def forward(self, x):
        # Return mock predictions
        batch_size, _, height, width = x.shape
        output = torch.zeros((batch_size, 33, height, width))  # 33 classes (including background)
        # Add some mock predictions
        output[:, 1, :10, :10] = 1.0  # First tooth
        output[:, 2, 20:30, 20:30] = 1.0  # Second tooth
        return output

# Tests for BaseEvaluator
def test_base_evaluator_initialization():
    dataset = MockDataset()
    evaluator = BaseEvaluator(dataset)
    assert evaluator.val_dataset == dataset
    assert evaluator.results == {}

def test_get_ground_truth():
    dataset = MockDataset()
    evaluator = BaseEvaluator(dataset)
    masks, boxes = evaluator.get_ground_truth(1)
    
    assert masks.shape == (32, 64, 64)
    assert boxes.shape == (32, 4)
    # Check first tooth mask has content
    assert masks[0].any()
    # Check first tooth box is not zeros
    assert boxes[0].any()

def test_compute_dice_scores():
    dataset = MockDataset()
    evaluator = BaseEvaluator(dataset)
    
    # Mock get_predictions to return known values
    def mock_predictions(image_id):
        masks = np.zeros((32, 64, 64), dtype=bool)
        boxes = np.zeros((32, 4), dtype=float)
        # Add some predictions matching ground truth
        masks[0, :10, :10] = True
        boxes[0] = [0, 0, 10, 10]
        return masks, boxes
    
    evaluator.get_predictions = mock_predictions
    
    scores = evaluator.compute_dice_scores()
    assert 'overall' in scores
    assert 'by_class' in scores
    assert 'by_type' in scores
    assert len(scores['by_class']) == 32

# Tests for UNetEvaluator
def test_unet_evaluator_initialization():
    dataset = MockDataset()
    model = MockUNet()
    device = 'cpu'
    evaluator = UNetEvaluator(dataset, model, device)
    
    assert evaluator.model == model
    assert evaluator.device == device

def test_unet_get_predictions():
    dataset = MockDataset()
    model = MockUNet()
    device = 'cpu'
    evaluator = UNetEvaluator(dataset, model, device)
    
    masks, boxes = evaluator.get_predictions(1)
    
    assert masks.shape == (32, 64, 64)
    assert boxes.shape == (32, 4)
    # Check that predictions contain expected content
    assert masks[0, :10, :10].any()  # First tooth
    assert masks[1, 20:30, 20:30].any()  # Second tooth

def test_unet_infer_boxes():
    dataset = MockDataset()
    model = MockUNet()
    device = 'cpu'
    evaluator = UNetEvaluator(dataset, model, device)
    
    # Create test masks
    masks = np.zeros((32, 64, 64), dtype=bool)
    masks[0, :10, :10] = True  # Simple square mask
    
    boxes = evaluator.infer_boxes(masks)
    assert boxes.shape == (32, 4)
    # Check first box coordinates
    np.testing.assert_array_equal(boxes[0], [0, 0, 9, 9])

def test_unet_visualization():
    dataset = MockDataset()
    model = MockUNet()
    device = 'cpu'
    evaluator = UNetEvaluator(dataset, model, device)
    
    try:
        # Get predictions and print shapes
        masks, boxes = evaluator.get_predictions(1)
        image = dataset.load_image(1)
        print(f"Image shape: {image.shape}")
        print(f"Masks shape: {masks.shape}")
        print(f"Boxes shape: {boxes.shape}")
        
        evaluator.visualize_prediction(1)
        success = True
    except Exception as e:
        success = False
        print(f"Visualization failed with error: {str(e)}")
    assert success