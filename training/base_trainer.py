import torch
import os
from datetime import datetime
import json
from typing import Dict, Optional
from abc import ABC, abstractmethod

class SegmentationTrainer(ABC):
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
        self.current_loss = float('inf')
        self.best_metric = float('-inf')
        self.metrics_history = []

    @abstractmethod
    def train_epoch(self) -> float:
        """Train for one epoch and return average loss.
        
        To be implemented by child classes.
        """
        pass

    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """Run validation and return metrics.
        
        To be implemented by child classes.
        """
        pass
        
    def save_checkpoint(self, save_dir: str, is_best: bool = False) -> None:
        """Save model checkpoint with training state."""
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics_history': self.metrics_history,
            'best_metric': self.best_metric,
            'current_loss': self.current_loss
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(save_dir, f"checkpoint_latest.pth")
        torch.save(checkpoint, latest_path)
        
        # Save best model if this is the best so far
        if is_best:
            best_path = os.path.join(save_dir, "best_model.pth")
            torch.save(checkpoint, best_path)

        # Save milestone model every 10 epochs
        if (self.current_epoch + 1) % 10 == 0:  # Every 10 epochs
            milestone_path = os.path.join(save_dir, f"checkpoint_epoch_{self.current_epoch + 1}.pth")
            torch.save(checkpoint, milestone_path)
            
        # Log the save
        self._log_event(f"Saved checkpoint for epoch {self.current_epoch}", save_dir)
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint and training state."""
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.metrics_history = checkpoint.get('metrics_history', [])
        self.best_metric = checkpoint.get('best_metric', float('-inf'))
        self.current_loss = checkpoint.get('current_loss', float('inf'))
        
        self._log_event(f"Loaded checkpoint from epoch {self.current_epoch}")
        self._log_event(f"Restored to epoch {self.current_epoch}")
        self._log_event(f"Best metric so far: {self.best_metric}")
        
    def _log_event(self, message: str, base_dir: Optional[str] = None) -> None:
        """Log a training event.
        
        Args:
            message: Message to log
            base_dir: Optional base directory for logs. If None, uses current directory
        """
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'epoch': self.current_epoch,
            'event': message
        }
        
        # Determine log directory
        if base_dir:
            log_dir = os.path.join(base_dir, 'logs')
        else:
            log_dir = 'logs'
            
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, 'training.log')
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Warning: Failed to write to log file: {e}")

    def train(
        self, 
        epochs: int, 
        save_dir: str,
        resume_from: Optional[str] = None,
        save_frequency: int = 1,
        val_frequency: int = 1
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
            self.current_loss = train_loss
            self._log_event(f"Epoch {epoch} training complete. Loss: {train_loss:.4f}", save_dir)
            print(f"Epoch {epoch} training complete. Loss: {train_loss:.4f}")
            
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
            self._log_event(f"Validation metrics: {val_metrics}", save_dir)