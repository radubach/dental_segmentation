from torch.utils.data import Dataset
import os
import json
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
import numpy as np
import torch


class UNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, coco_json, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        with open(coco_json, "r") as f:
            coco_data = json.load(f)

        # Group annotations by image_id
        self.image_info = {img["id"]: img for img in coco_data["images"]}
        self.image_annotations = {img_id: [] for img_id in self.image_info.keys()}
        for annotation in coco_data["annotations"]:
            self.image_annotations[annotation["image_id"]].append(annotation)

        # Use only image IDs for indexing
        self.image_ids = list(self.image_info.keys())
        print(f"Dataset initialized with {len(self.image_ids)} images.")

    def __getitem__(self, index):
        # print(f"getting item {index}")
        image_id = self.image_ids[index]
        image_name = self.image_info[image_id]["file_name"]
        image_path = os.path.join(self.image_dir, image_name)

        # Load image
        image = Image.open(image_path).convert("L")  # Grayscale

        # Create composite mask
        mask = Image.new("L", image.size, 0)  # Start with a blank mask
        draw = ImageDraw.Draw(mask)
        for annotation in self.image_annotations[image_id]:
            points = np.array(annotation["segmentation"]).reshape(-1, 2)
            draw.polygon([tuple(p) for p in points], fill=annotation["category_id"]+1) # source data is 0-indexed

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

    def __len__(self):
        return len(self.image_ids)


def transform(image, mask):
    image = TF.resize(image, (256, 256))
    mask = TF.resize(mask, (256, 256), interpolation=Image.NEAREST)
    image = TF.to_tensor(image)
    mask = torch.from_numpy(np.array(mask, dtype=np.int64))  # Convert to tensor
    return image, mask