from .base import BaseEvaluator
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

class UNetEvaluator(BaseEvaluator):
    def __init__(self, val_dataset, model, device):
        super().__init__(val_dataset)
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def get_predictions(self, image_id):
        # Load and preprocess image
        image = self.val_dataset.load_image(image_id)
        # Resize to model input size
        image = image.resize(self.val_dataset.input_size, resample=Image.BILINEAR)
        # Convert to tensor properly
        image_tensor = TF.to_tensor(image).unsqueeze(0)  # Only need one unsqueeze for batch dim
        image_tensor = image_tensor.to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            
        # Convert output to numpy array of binary masks
        # Assuming output is (1, C, H, W) where C is number of classes
        pred = output.argmax(dim=1).squeeze().cpu().numpy()  # (H, W) with values 0-32
        print(f"Output shape: {output.shape}")
        print(f"Unique values in prediction: {np.unique(pred)}")
        
        # Convert to 32 binary masks (excluding background class 0)
        masks = np.zeros((32, pred.shape[0], pred.shape[1]), dtype=bool)
        for i in range(32):
            masks[i] = (pred == (i + 1))  # i+1 because 0 is background
            
        # Generate boxes from masks
        boxes = self.infer_boxes(masks)
        
        return masks, boxes

    def infer_boxes(self, masks):
        # Initialize boxes array
        boxes = np.zeros((32, 4), dtype=float)
        
        for i in range(32):
            mask = masks[i]
            if mask.any():  # If mask contains any True values
                # Find the bounding box of the mask
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                y1, y2 = np.where(rows)[0][[0, -1]]
                x1, x2 = np.where(cols)[0][[0, -1]]
                boxes[i] = [x1, y1, x2, y2]
                
        return boxes 