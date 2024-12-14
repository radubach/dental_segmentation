import pytest
import torch
import torchvision
from torch.utils.data import DataLoader
import tempfile
import os

from training.maskrcnn_trainer import MaskRCNNTrainer

@pytest.fixture
def mock_model():
    """Create a small Mask R-CNN model for testing."""
    return torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=3)  # 2 classes + background

@pytest.fixture
def mock_data():
    """Create mock data for testing."""
    # Create a simple batch of data
    images = [torch.randn(3, 64, 64)]
    targets = [{
        'boxes': torch.tensor([[10, 10, 20, 20]], dtype=torch.float32),
        'labels': torch.tensor([1], dtype=torch.int64),
        'masks': torch.ones((1, 64, 64), dtype=torch.uint8),
        'image_id': torch.tensor([1])
    }]
    return images, targets

@pytest.fixture
def mock_dataloader(mock_data):
    """Create a mock dataloader that returns the same batch."""
    class MockDataLoader:
        def __init__(self, mock_data):
            self.mock_data = mock_data
            
        def __iter__(self):
            for _ in range(1):  # Return just one batch
                yield self.mock_data
            
        def __len__(self):
            return 1
    
    return MockDataLoader(mock_data)

class TestMaskRCNNTrainer:
    def test_initialization(self, mock_model, mock_dataloader):
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=1e-4)
        
        trainer = MaskRCNNTrainer(
            model=mock_model,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            device=device,
            optimizer=optimizer
        )
        
        assert trainer.model is not None
        assert trainer.current_epoch == 0
        assert trainer.best_metric == float('-inf')

    def test_train_epoch(self, mock_model, mock_dataloader):
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=1e-4)
        
        trainer = MaskRCNNTrainer(
            model=mock_model,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            device=device,
            optimizer=optimizer
        )
        
        loss = trainer.train_epoch()
        assert isinstance(loss, float)
        assert loss > 0  # Loss should be positive
        
    def test_validate(self, mock_model, mock_dataloader):
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=1e-4)
        
        trainer = MaskRCNNTrainer(
            model=mock_model,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            device=device,
            optimizer=optimizer
        )
        
        metrics = trainer.validate()
        assert isinstance(metrics, dict)
        assert 'val_loss' in metrics
        
    def test_checkpoint_saving_loading(self, mock_model, mock_dataloader):
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=1e-4)
        
        trainer = MaskRCNNTrainer(
            model=mock_model,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            device=device,
            optimizer=optimizer
        )
        
        # Create temporary directory for checkpoints
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save checkpoint
            trainer.save_checkpoint(tmp_dir)
            saved_epoch = trainer.current_epoch  # Store the epoch at save time
            
            # Verify checkpoint exists
            assert os.path.exists(os.path.join(tmp_dir, "checkpoint_latest.pth"))
            
            # Create new trainer and load checkpoint
            new_trainer = MaskRCNNTrainer(
                model=mock_model,
                train_loader=mock_dataloader,
                val_loader=mock_dataloader,
                device=device,
                optimizer=optimizer
            )
            
            new_trainer.load_checkpoint(os.path.join(tmp_dir, "checkpoint_latest.pth"))
            
            # Verify state was loaded (new_trainer epoch should be saved_epoch + 1)
            assert new_trainer.current_epoch == saved_epoch + 1
            assert new_trainer.best_metric == trainer.best_metric

    def test_full_training_loop(self, mock_model, mock_dataloader):
        device = torch.device("cpu")
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=1e-4)
        
        trainer = MaskRCNNTrainer(
            model=mock_model,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            device=device,
            optimizer=optimizer
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Run training for 2 epochs
            try:
                trainer.train(
                    epochs=2,
                    save_dir=tmp_dir,
                    save_frequency=1
                )
            except Exception as e:
                pytest.fail(f"Training failed with error: {str(e)}")
            
            assert trainer.current_epoch == 1  # 0-based indexing
            assert len(trainer.metrics_history) == 2