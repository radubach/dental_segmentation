import albumentations as A

def get_unet_transforms():
    return A.Compose([
        A.Resize(height=2048, width=1024),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        # ... other transforms
    ])