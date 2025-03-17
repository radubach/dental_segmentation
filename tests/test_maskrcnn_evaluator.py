import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch
from evaluation.maskrcnn import MaskRCNNEvaluator
from configs.config import Config, DataConfig, ModelConfig

class MockConfig:
    def __init__(self):
        self.data = DataConfig()
        self.data.input_size = (64, 64)
        self.data.normalize_mean = (0.5,)
        self.data.normalize_std = (0.5,)
        self.training = Mock()
        self.training.device = "cpu"

class MockMaskRCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.model = self  # Add self reference for compatibility
        
    def forward(self, images):
        batch_size = len(images)
        return [
            {
                'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),
                'scores': torch.tensor([0.9], dtype=torch.float32),
                'masks': torch.ones((1, 1, 256, 256), dtype=torch.float32)
            }
            for _ in range(batch_size)
        ]
        
    def to(self, device):
        self.device = device
        return self

@pytest.fixture
def mock_dataset():
    dataset = Mock()
    dataset.image_ids = [1, 2, 3]
    dataset.load_image = lambda image_id: Image.fromarray(
        np.zeros((64, 64), dtype=np.uint8)
    )
    return dataset

@pytest.fixture
def evaluator(mock_dataset):
    config = MockConfig()
    model = MockMaskRCNN()
    return MaskRCNNEvaluator(
        val_dataset=mock_dataset,
        model=model,
        config=config,
        confidence_threshold=0.5,
        nms_threshold=0.5
    )

@pytest.fixture
def config():
    """Create a config for testing."""
    config = Config()
    config.model.num_classes = 2  # Background + 1 class
    config.model.input_size = (256, 256)
    config.model.device = "cpu"
    config.model.confidence_threshold = 0.5
    config.model.nms_threshold = 0.3
    return config

@pytest.fixture
def mock_model():
    """Create a mock Mask R-CNN model."""
    class MockMaskRCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, images):
            batch_size = len(images)
            return [
                {
                    'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
                    'labels': torch.tensor([1], dtype=torch.int64),
                    'scores': torch.tensor([0.9], dtype=torch.float32),
                    'masks': torch.ones((1, 1, 256, 256), dtype=torch.float32)
                }
                for _ in range(batch_size)
            ]
            
    return MockMaskRCNN()

def test_initialization(mock_dataset):
    """Test evaluator initialization."""
    config = MockConfig()
    model = MockMaskRCNN()
    evaluator = MaskRCNNEvaluator(mock_dataset, model, config)
    
    assert evaluator.model is model
    assert evaluator.config is config
    assert evaluator.device == "cpu"
    assert evaluator.confidence_threshold == 0.5
    assert evaluator.nms_threshold == 0.5

def test_get_predictions(evaluator):
    """Test prediction generation and post-processing."""
    predictions = evaluator.get_predictions(1)
    
    # Check prediction structure
    assert isinstance(predictions, dict)
    assert all(k in predictions for k in ['boxes', 'masks', 'labels', 'scores'])
    
    # Check shapes and types
    assert predictions['boxes'].shape[1] == 4  # [N, 4] boxes
    assert len(predictions['masks'].shape) == 3  # [N, H, W] masks
    assert predictions['labels'].dim() == 1  # [N] labels
    assert predictions['scores'].dim() == 1  # [N] scores
    
    # Check confidence filtering
    assert torch.all(predictions['scores'] >= evaluator.confidence_threshold)

def test_low_confidence_filtering(evaluator):
    """Test filtering of low confidence predictions."""
    evaluator.confidence_threshold = 0.85
    predictions = evaluator.get_predictions(1)
    
    # Only the first prediction should remain (score 0.9)
    assert predictions['boxes'].shape[0] == 1
    assert predictions['scores'][0] >= 0.85

def test_nms_filtering(evaluator):
    """Test NMS filtering of overlapping predictions."""
    # Mock overlapping boxes
    with patch.object(evaluator.model, 'forward') as mock_forward:
        mock_forward.return_value = [{
            'boxes': torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float32),
            'labels': torch.tensor([1, 1], dtype=torch.int64),
            'scores': torch.tensor([0.9, 0.8], dtype=torch.float32),
            'masks': torch.ones((2, 1, 64, 64), dtype=torch.float32)
        }]
        
        predictions = evaluator.get_predictions(1)
        # Should keep only the highest scoring box due to NMS
        assert predictions['boxes'].shape[0] == 1
        assert predictions['scores'][0] == 0.9

def test_visualization(evaluator):
    """Test prediction visualization."""
    # Create a test image
    image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    predictions = evaluator.get_predictions(1)
    
    # Test visualization
    result = evaluator.visualize_predictions(image, predictions)
    assert isinstance(result, Image.Image)
    assert result.size == (64, 64)

def test_empty_predictions(evaluator):
    """Test handling of images with no predictions above threshold."""
    # Mock no predictions above threshold
    with patch.object(evaluator.model, 'forward') as mock_forward:
        mock_forward.return_value = [{
            'boxes': torch.tensor([], dtype=torch.float32).reshape(0, 4),
            'labels': torch.tensor([], dtype=torch.int64),
            'scores': torch.tensor([], dtype=torch.float32),
            'masks': torch.zeros((0, 1, 64, 64), dtype=torch.float32)
        }]
        
        predictions = evaluator.get_predictions(1)
        assert all(predictions[k].shape[0] == 0 for k in predictions)

def test_resize_to_original(evaluator):
    """Test resizing predictions to original image size."""
    # Mock different size original image
    with patch.object(evaluator.val_dataset, 'load_image') as mock_load:
        mock_load.return_value = Image.fromarray(
            np.zeros((128, 128), dtype=np.uint8)
        )
        
        predictions = evaluator.get_predictions(1)
        # Masks should be resized to 128x128
        assert predictions['masks'].shape[1:] == (128, 128)
        # Boxes should be scaled accordingly
        assert torch.all(predictions['boxes'] >= 0)
        assert torch.all(predictions['boxes'] <= 128)

def test_initialization(config, mock_model):
    evaluator = MaskRCNNEvaluator(config, mock_model)
    assert evaluator.device == torch.device("cpu")
    assert evaluator.num_classes == 2
    assert evaluator.input_size == (256, 256)
    assert evaluator.confidence_threshold == 0.5
    assert evaluator.nms_threshold == 0.3

def test_preprocess_image(config, mock_model):
    evaluator = MaskRCNNEvaluator(config, mock_model)
    
    # Create a test image
    image = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    
    # Test preprocessing
    processed = evaluator.preprocess_image(image)
    assert isinstance(processed, torch.Tensor)
    assert processed.shape == (3, 256, 256)  # RGB image
    assert processed.device == torch.device("cpu")

def test_postprocess_output(config, mock_model):
    evaluator = MaskRCNNEvaluator(config, mock_model)
    
    # Create mock output
    output = {
        'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
        'labels': torch.tensor([1], dtype=torch.int64),
        'scores': torch.tensor([0.9], dtype=torch.float32),
        'masks': torch.ones((1, 1, 256, 256), dtype=torch.float32)
    }
    
    # Test postprocessing
    boxes, labels, scores, masks = evaluator.postprocess_output(output)
    assert isinstance(boxes, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert isinstance(masks, np.ndarray)
    
    assert boxes.shape == (1, 4)
    assert labels.shape == (1,)
    assert scores.shape == (1,)
    assert masks.shape == (1, 256, 256)

def test_predict(config, mock_model):
    evaluator = MaskRCNNEvaluator(config, mock_model)
    
    # Create a test image
    image = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    
    # Test prediction
    boxes, labels, scores, masks = evaluator.predict(image)
    assert isinstance(boxes, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert isinstance(masks, np.ndarray)
    
    assert boxes.shape[1] == 4
    assert len(labels.shape) == 1
    assert len(scores.shape) == 1
    assert len(masks.shape) == 3
    assert masks.shape[1:] == (512, 512)  # Original size

def test_predict_batch(config, mock_model):
    evaluator = MaskRCNNEvaluator(config, mock_model)
    
    # Create test images
    images = [Image.fromarray(np.zeros((512, 512), dtype=np.uint8)) for _ in range(3)]
    
    # Test batch prediction
    predictions = evaluator.predict_batch(images)
    assert isinstance(predictions, list)
    assert len(predictions) == 3
    
    for pred in predictions:
        boxes, labels, scores, masks = pred
        assert isinstance(boxes, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert isinstance(masks, np.ndarray)
        
        assert boxes.shape[1] == 4
        assert len(labels.shape) == 1
        assert len(scores.shape) == 1
        assert masks.shape[1:] == (512, 512)

def test_confidence_filtering(config, mock_model):
    evaluator = MaskRCNNEvaluator(config, mock_model)
    
    # Create a test image
    image = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    
    # Set low confidence threshold
    evaluator.confidence_threshold = 0.95
    boxes, labels, scores, masks = evaluator.predict(image)
    assert len(boxes) == 0  # No predictions above threshold
    
    # Set high confidence threshold
    evaluator.confidence_threshold = 0.1
    boxes, labels, scores, masks = evaluator.predict(image)
    assert len(boxes) > 0  # Should have predictions

def test_nms_filtering(config, mock_model):
    class MockModelWithOverlap(torch.nn.Module):
        def forward(self, images):
            return [{
                'boxes': torch.tensor([[100, 100, 200, 200], 
                                     [110, 110, 210, 210]], dtype=torch.float32),
                'labels': torch.tensor([1, 1], dtype=torch.int64),
                'scores': torch.tensor([0.9, 0.8], dtype=torch.float32),
                'masks': torch.ones((2, 1, 256, 256), dtype=torch.float32)
            }]
    
    evaluator = MaskRCNNEvaluator(config, MockModelWithOverlap())
    
    # Create a test image
    image = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    
    # Test with high NMS threshold (keep both boxes)
    evaluator.nms_threshold = 0.9
    boxes, labels, scores, masks = evaluator.predict(image)
    assert len(boxes) == 2
    
    # Test with low NMS threshold (keep one box)
    evaluator.nms_threshold = 0.1
    boxes, labels, scores, masks = evaluator.predict(image)
    assert len(boxes) == 1

def test_invalid_input(config, mock_model):
    evaluator = MaskRCNNEvaluator(config, mock_model)
    
    # Test with invalid input type
    with pytest.raises(TypeError):
        evaluator.predict(np.zeros((512, 512)))
    
    # Test with invalid input size (too small)
    image = Image.fromarray(np.zeros((32, 32), dtype=np.uint8))
    with pytest.raises(ValueError):
        evaluator.predict(image)

def test_visualization(config, mock_model):
    evaluator = MaskRCNNEvaluator(config, mock_model)
    
    # Create a test image
    image = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    
    # Test visualization
    try:
        vis_image = evaluator.visualize_predictions(image)
        assert isinstance(vis_image, Image.Image)
        assert vis_image.size == (512, 512)
    except Exception as e:
        pytest.fail(f"Visualization failed with error: {str(e)}")

def test_empty_predictions(config, mock_model):
    class MockModelEmpty(torch.nn.Module):
        def forward(self, images):
            return [{
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'scores': torch.zeros(0, dtype=torch.float32),
                'masks': torch.zeros((0, 1, 256, 256), dtype=torch.float32)
            }]
    
    evaluator = MaskRCNNEvaluator(config, MockModelEmpty())
    
    # Create a test image
    image = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    
    # Test prediction with no detections
    boxes, labels, scores, masks = evaluator.predict(image)
    assert len(boxes) == 0
    assert len(labels) == 0
    assert len(scores) == 0
    assert masks.shape[0] == 0

def test_device_handling(config, mock_model):
    # Test CPU device
    config.model.device = "cpu"
    evaluator = MaskRCNNEvaluator(config, mock_model)
    assert evaluator.device == torch.device("cpu")
    
    # Test invalid device
    config.model.device = "invalid_device"
    with pytest.raises(ValueError):
        _ = MaskRCNNEvaluator(config, mock_model) 