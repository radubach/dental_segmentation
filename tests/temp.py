import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# Create a simple numpy array
img = np.zeros((256, 256, 1), dtype=np.float32)

# Try the transform
transform = A.Compose([ToTensorV2()])
result = transform(image=img)