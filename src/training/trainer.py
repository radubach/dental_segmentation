
# src/training/trainer.py
class SegmentationTrainer:
    """Base trainer class for segmentation models."""
    def __init__(self, model, train_loader, val_loader, 
                 device, loss_fn, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_epoch(self):
        pass

    def validate(self):
        pass

class UNetTrainer(SegmentationTrainer):
    """UNet-specific training logic."""
    pass