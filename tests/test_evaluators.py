import pytest
import numpy as np
import torch
from PIL import Image
from unittest.mock import Mock, patch
from evaluation.base import BaseEvaluator
from evaluation.unet import UNetEvaluator
from configs.config import Config

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

class MockConfig:
    def __init__(self):
        self.model = self
        self.device = "cpu"
        self.data = DataConfig()
        self.data.input_size = (64, 64)
        self.data.normalize_mean = (0.5,)
        self.data.normalize_std = (0.5,)
        self.training = Mock()
        self.training.device = "cpu"
        
    def to(self, device):
        self.device = device
        return self

class MockUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.model = self  # Add self reference for compatibility
        
    def forward(self, x):
        # Return mock predictions
        batch_size, _, height, width = x.shape
        return torch.randn(batch_size, 2, height, width)  # 2 classes
        
    def to(self, device):
        self.device = device
        return self

class MockMaskRCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        
    def forward(self, x):
        # Return mock predictions
        return [{
            'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64),
            'scores': torch.tensor([0.9], dtype=torch.float32),
            'masks': torch.ones((1, 1, x.shape[2], x.shape[3]), dtype=torch.float32)
        }]
        
    def to(self, device):
        self.device = device
        return self

class MockModelWithOverlap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.model = self  # Add self reference for compatibility
        
    def forward(self, images):
        return [{
            'boxes': torch.tensor([[100, 100, 200, 200],
                                 [110, 110, 210, 210]], dtype=torch.float32),
            'labels': torch.tensor([1, 1], dtype=torch.int64),
            'scores': torch.tensor([0.9, 0.8], dtype=torch.float32),
            'masks': torch.ones((2, 1, 256, 256), dtype=torch.float32)
        }]
        
    def to(self, device):
        self.device = device
        return self

class MockModelEmpty(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.model = self  # Add self reference for compatibility
        
    def forward(self, x):
        return [{
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros(0, dtype=torch.int64),
            'scores': torch.zeros(0, dtype=torch.float32),
            'masks': torch.zeros((0, 1, x.shape[2], x.shape[3]), dtype=torch.float32)
        }]
        
    def to(self, device):
        self.device = device
        return self

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
    config = Config()
    config.model.device = 'cpu'
    model = MockUNet()
    evaluator = UNetEvaluator(config, model)
    
    assert evaluator.model == model
    assert evaluator.device == torch.device('cpu')

def test_unet_get_predictions():
    config = Config()
    config.model.device = 'cpu'
    model = MockUNet()
    evaluator = UNetEvaluator(config, model)
    
    # Create test image
    image = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
    pred = evaluator.predict(image)
    
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (64, 64)

def test_unet_infer_boxes():
    config = Config()
    config.model.device = 'cpu'
    model = MockUNet()
    evaluator = UNetEvaluator(config, model)
    
    # Create test masks
    masks = np.zeros((32, 64, 64), dtype=bool)
    masks[0, :10, :10] = True  # Simple square mask
    
    boxes = evaluator.infer_boxes(masks)
    assert boxes.shape == (32, 4)
    # Check first box coordinates
    np.testing.assert_array_equal(boxes[0], [0, 0, 9, 9])

def test_unet_visualization():
    config = Config()
    config.model.device = 'cpu'
    model = MockUNet()
    evaluator = UNetEvaluator(config, model)
    
    # Create test image
    image = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
    
    try:
        # Get predictions
        pred = evaluator.predict(image)
        print(f"Image shape: {image.size}")
        print(f"Prediction shape: {pred.shape}")
        success = True
    except Exception as e:
        success = False
        print(f"Visualization failed with error: {str(e)}")
    assert success

@pytest.fixture
def config():
    """Create a config for testing."""
    config = Config()
    config.model.num_classes = 3
    config.model.input_size = (256, 256)
    config.model.device = "cpu"
    return config

@pytest.fixture
def mock_model():
    """Create a mock UNet model."""
    class MockUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x):
            batch_size = x.shape[0]
            return torch.randn(batch_size, 3, 256, 256)  # 3 classes
            
    return MockUNet()

def test_initialization(config, mock_model):
    evaluator = UNetEvaluator(config, mock_model)
    assert evaluator.device == torch.device("cpu")
    assert evaluator.num_classes == 3
    assert evaluator.input_size == (256, 256)

def test_preprocess_image(config, mock_model):
    evaluator = UNetEvaluator(config, mock_model)
    
    # Create a test image
    image = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    
    # Test preprocessing
    processed = evaluator.preprocess_image(image)
    assert isinstance(processed, torch.Tensor)
    assert processed.shape == (1, 1, 256, 256)  # Batch size 1, 1 channel
    assert processed.device == torch.device("cpu")

def test_postprocess_output(config, mock_model):
    evaluator = UNetEvaluator(config, mock_model)
    
    # Create mock output
    output = torch.randn(1, 3, 256, 256)  # Batch size 1, 3 classes
    
    # Test postprocessing
    processed = evaluator.postprocess_output(output)
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (256, 256)
    assert processed.dtype == np.uint8
    assert np.all(processed >= 0) and np.all(processed < 3)

def test_predict(config, mock_model):
    evaluator = UNetEvaluator(config, mock_model)
    
    # Create a test image
    image = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    
    # Test prediction
    prediction = evaluator.predict(image)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (512, 512)  # Original size
    assert prediction.dtype == np.uint8

def test_predict_batch(config, mock_model):
    evaluator = UNetEvaluator(config, mock_model)
    
    # Create test images
    images = [Image.fromarray(np.zeros((512, 512), dtype=np.uint8)) for _ in range(3)]
    
    # Test batch prediction
    predictions = evaluator.predict_batch(images)
    assert isinstance(predictions, list)
    assert len(predictions) == 3
    assert all(isinstance(p, np.ndarray) for p in predictions)
    assert all(p.shape == (512, 512) for p in predictions)
    assert all(p.dtype == np.uint8 for p in predictions)

def test_sliding_window_inference(config, mock_model):
    evaluator = UNetEvaluator(config, mock_model)
    
    # Create a large test image
    image = Image.fromarray(np.zeros((1024, 1024), dtype=np.uint8))
    
    # Test sliding window inference
    prediction = evaluator.predict(image, use_sliding_window=True)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1024, 1024)  # Original size
    assert prediction.dtype == np.uint8

def test_invalid_input(config, mock_model):
    evaluator = UNetEvaluator(config, mock_model)
    
    # Test with invalid input type
    with pytest.raises(TypeError):
        evaluator.predict(np.zeros((512, 512)))
    
    # Test with invalid input size (too small)
    image = Image.fromarray(np.zeros((32, 32), dtype=np.uint8))
    with pytest.raises(ValueError):
        evaluator.predict(image)

def test_device_handling(config, mock_model):
    # Test CPU device
    config.model.device = "cpu"
    evaluator = UNetEvaluator(config, mock_model)
    assert evaluator.device == torch.device("cpu")
    
    # Test invalid device
    config.model.device = "invalid_device"
    with pytest.raises(ValueError):
        _ = UNetEvaluator(config, mock_model)