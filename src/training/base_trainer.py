# src/training/base_trainer.py
import torch
import os
from datetime import datetime
import json
from typing import Dict, Optional

class SegmentationTrainer:
    """Base trainer class for segmentation models."""
    def __init__(
        self, 
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.best_metric = float('-inf')
        self.metrics_history = []
        
    def save_checkpoint(self, save_dir: str, is_best: bool = False) -> None:
        """Save model checkpoint with training state."""
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics_history': self.metrics_history,
            'best_metric': self.best_metric
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(save_dir, f"checkpoint_epoch_{self.current_epoch}.pth")
        torch.save(checkpoint, latest_path)
        
        # Save best model if this is the best so far
        if is_best:
            best_path = os.path.join(save_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            
        # Log the save
        self._log_event(f"Saved checkpoint for epoch {self.current_epoch}")
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint and training state."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.metrics_history = checkpoint['metrics_history']
        self.best_metric = checkpoint['best_metric']
        
        self._log_event(f"Loaded checkpoint from epoch {self.current_epoch}")
        
    def _log_event(self, message: str) -> None:
        """Log a training event."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'epoch': self.current_epoch,
            'event': message
        }
        
        # Ensure log directory exists
        os.makedirs('logs', exist_ok=True)
        log_file = 'logs/training.log'
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def train(
        self, 
        epochs: int, 
        save_dir: str,
        resume_from: Optional[str] = None,
        save_frequency: int = 1
    ) -> None:
        """Full training loop with checkpoints."""
        # Resume if checkpoint provided
        if resume_from:
            self.load_checkpoint(resume_from)
            
        start_epoch = self.current_epoch
        end_epoch = start_epoch + epochs
        
        for epoch in range(start_epoch, end_epoch):
            self.current_epoch = epoch
            
            # Training
            train_loss = self.train_epoch()
            self._log_event(f"Epoch {epoch + 1} training complete. Loss: {train_loss:.4f}")
            
            # Validation
            val_metrics = self.validate()
            self.metrics_history.append(val_metrics)
            
            # Check if this is the best model
            current_metric = val_metrics['dice_overall']
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric
                
            # Save based on frequency and best model
            if (epoch + 1) % save_frequency == 0:
                self.save_checkpoint(save_dir, is_best)
                
            # Log metrics
            self._log_event(f"Validation metrics: {val_metrics}")