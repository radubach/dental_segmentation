from typing import Dict
import torch
from .base_trainer import SegmentationTrainer


class MaskRCNNTrainer(SegmentationTrainer):
    """Trainer for Mask R-CNN model."""
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
    ):
        # Note: we don't pass loss_fn because Mask R-CNN computes loss internally
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            loss_fn=None,  # Mask R-CNN handles loss internally
            optimizer=optimizer
        )
        self.loss_components = {}  # Track individual loss components

    def train_epoch(self) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        self.loss_components = {}  # Reset loss components for this epoch
        
        for images, targets in self.train_loader:
            # Move data to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass (returns dict of losses)
            loss_dict = self.model(images, targets)
            
            # Track individual losses
            for k, v in loss_dict.items():
                self.loss_components[k] = self.loss_components.get(k, 0) + v.item()
            
            # Compute total loss
            losses = sum(loss for loss in loss_dict.values())
            
            # Check for invalid loss
            if not torch.isfinite(losses):
                self._log_event(f"Invalid loss detected: {losses}")
                self._log_event(f"Loss components: {loss_dict}")
                raise ValueError(f"Loss is {losses}, stopping training")
            
            # Backward pass
            losses.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += losses.item()
        
        # Compute averages
        avg_loss = total_loss / len(self.train_loader)
        avg_components = {
            k: v / len(self.train_loader) 
            for k, v in self.loss_components.items()
        }
        
        # Log detailed loss components
        self._log_event(f"Average loss components for epoch {self.current_epoch}:")
        for k, v in avg_components.items():
            self._log_event(f"{k}: {v:.4f}")
        
        return avg_loss

    def validate(self) -> Dict[str, float]:
        """Run validation and return metrics."""
        self.model.eval()
        metrics = {}
        val_loss = 0.0
        all_dice_scores = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                # Move data to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Get loss
                self.model.train()  # Temporarily set to train to get losses
                loss_dict = self.model(images, targets)
                self.model.eval()   # Set back to eval
                
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                
                # Get predictions
                predictions = self.model(images)
                
                # Compute dice score for each image in batch
                for pred, target in zip(predictions, targets):
                    dice_score = self._compute_instance_dice(pred, target)
                    all_dice_scores.append(dice_score)
        
        # Compute metrics
        metrics['val_loss'] = val_loss / len(self.val_loader)
        metrics['dice_overall'] = sum(all_dice_scores) / len(all_dice_scores) if all_dice_scores else 0.0
        
        return metrics

    def _compute_instance_dice(self, prediction: Dict, target: Dict) -> float:
        """Compute Dice score between predicted and target masks."""
        # Get target masks and create a combined semantic mask
        target_masks = target['masks']  # Shape: (N, H, W)
        target_labels = target['labels']
        H, W = target_masks.shape[1:]
        semantic_target = torch.zeros((H, W), device=self.device)
        
        # Fill target semantic mask
        for mask, label in zip(target_masks, target_labels):
            semantic_target[mask > 0.5] = label
            
        # Get predicted masks and create a combined semantic mask
        pred_masks = prediction['masks']  # Shape: (N, 1, H, W)
        pred_labels = prediction['labels']
        pred_scores = prediction['scores']
        semantic_pred = torch.zeros((H, W), device=self.device)
        
        # Use score threshold
        score_threshold = 0.5
        for mask, label, score in zip(pred_masks, pred_labels, pred_scores):
            if score > score_threshold:
                semantic_pred[mask[0] > 0.5] = label
        
        # Compute dice score
        intersection = torch.sum((semantic_pred > 0) & (semantic_target > 0))
        pred_area = torch.sum(semantic_pred > 0)
        target_area = torch.sum(semantic_target > 0)
        
        dice = 2 * intersection / (pred_area + target_area + 1e-6)  # Add small epsilon to avoid division by zero
        
        return dice.item()

    def _log_loss_components(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        """Helper method to log individual loss components."""
        components_str = ", ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
        self._log_event(f"Loss components: {components_str}")