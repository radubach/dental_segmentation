import torch
from typing import Dict
from torch.nn import CrossEntropyLoss

from .base_trainer import SegmentationTrainer
from utils.metrics import dice_metric

class UNetTrainer(SegmentationTrainer):
    """UNet-specific training implementation."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        num_classes: int = 33
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer
        )
        self.num_classes = num_classes

    def train_epoch(self) -> float:
        """Implement UNet-specific training step."""
        self.model.train()
        total_loss = 0.0
        
        for images, masks in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
        
    def validate(self) -> Dict[str, float]:
        """Implement UNet-specific validation."""
        self.model.eval()
        all_dice_scores = {c: [] for c in range(self.num_classes)}
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)
                per_class_dice = dice_metric(outputs, masks)
                
                for c, score in enumerate(per_class_dice):
                    all_dice_scores[c].append(score)
                    
        # Calculate mean dice scores
        mean_dice_scores = {
            f"dice_class_{c}": sum(scores) / len(scores) 
            for c, scores in all_dice_scores.items()
        }
        
        # Add overall dice score
        mean_dice_scores["dice_overall"] = sum(mean_dice_scores.values()) / len(mean_dice_scores)
        
        return mean_dice_scores