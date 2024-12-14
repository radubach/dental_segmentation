import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image, ImageDraw
import json
from pycocotools.coco import COCO
import torchvision.transforms.functional as TF




class BaseEvaluator:
    # mapping after adding 1 to category_id
    TOOTH_TYPE_MAPPING = {
        # Quadrant 1 (Upper Right)
        1: 'incisor', 
        2: 'incisor', 
        3: 'canine',  
        4: 'premolar',
        5: 'premolar',
        6: 'molar',   
        7: 'molar',   
        8: 'molar',   

        # Quadrant 2 (Upper Left)
        9: 'incisor', 
        10: 'incisor', 
        11: 'canine', 
        12: 'premolar',
        13: 'premolar',
        14: 'molar',   
        15: 'molar',   
        16: 'molar',   

        # Quadrant 3 (Lower Left)
        17: 'incisor', 
        18: 'incisor', 
        19: 'canine',  
        20: 'premolar',
        21: 'premolar',
        22: 'molar',   
        23: 'molar',   
        24: 'molar',   

        # Quadrant 4 (Lower Right)
        25: 'incisor', 
        26: 'incisor', 
        27: 'canine',  
        28: 'premolar',
        29: 'premolar',
        30: 'molar',   
        31: 'molar',   
        32: 'molar',   
    }

    def __init__(self, val_dataset):
        """
        Args:
            val_dataset: ValidationDataset instance containing image_ids to evaluate
                * has attribute `coco_data` containing COCO json data
                * has methods `get_annotations(image_id)` and `load_image(image_id)`
        """
        self.val_dataset = val_dataset
        
        # Data structures to store results
        self.results = {}  # Dictionary keyed by image_id containing all predictions and ground truth

    def get_ground_truth(self, image_id):
        """Extract ground truth masks and boxes from COCO json for given image_id.
        Returns exactly 32 masks and boxes, one for each tooth class.
        Masks and boxes are numpy arrays.
        Missing classes get zero masks and [0,0,0,0] boxes.
        If multiple instances exist for a class, uses the first one encountered."""
        img_info = self.val_dataset.coco.imgs[image_id]
        height, width = img_info['height'], img_info['width']
        
        # Initialize fixed-size arrays for 32 classes
        masks = np.zeros((32, height, width), dtype=bool)  # 32 binary masks
        boxes = np.zeros((32, 4), dtype=float)  # 32 boxes with [x1,y1,x2,y2] coordinates
        
        annotations = self.val_dataset.get_annotations(image_id)
        
        for ann in annotations:
            class_idx = ann['category_id']  # 0-index matches raw annotations
            # remember that predicted masks will include class labels that are 1-indexed
            # expect that predictions will also be numpy arrays of length 32
            
            if ann['segmentation'] and not masks[class_idx].any():
                if isinstance(ann['segmentation'], dict):
                    mask = self.val_dataset.coco.annToMask(ann)
                else:
                    mask = self.val_dataset.coco.annToMask(ann)
                masks[class_idx] = mask
            
            if ann['bbox'] and not boxes[class_idx].any():
                x, y, w, h = ann['bbox']
                boxes[class_idx] = [x, y, x + w, y + h]
        
        return masks, boxes
    

    def get_predictions(self, image_id):
        """
        MUST BE DEFINED IN SUBCLASS
        Get predictions from model in standardized format.
        
        Args:
            model: The model to generate predictions
            image_id: ID of image to predict
            
        Returns:
            tuple: (masks, boxes, labels) where:
                masks: numpy array of shape (N, H, W) containing N instance masks
                boxes: numpy array of shape (N, 4) containing boxes in [x1,y1,x2,y2] format
                labels: numpy array of shape (N,) containing class labels (1-32)
        """
        raise NotImplementedError("Each model type must implement get_predictions()")    
    

    def compute_dice_scores(self):
        """
        Compute dice scores for the validation dataset.
        Returns dictionary containing:
            - overall: dice score across all teeth
            - by_class: dice scores for each tooth class (1-32)
            - by_type: dice scores for each tooth type
        """
        # Initialize dice score components
        dice_scores = {
            'overall': {'numerator': 0, 'denominator': 0},
            'by_class': {i: {'numerator': 0, 'denominator': 0} for i in range(1, 33)},
            'by_type': {
                'incisor': {'numerator': 0, 'denominator': 0},
                'canine': {'numerator': 0, 'denominator': 0},
                'premolar': {'numerator': 0, 'denominator': 0},
                'molar': {'numerator': 0, 'denominator': 0}
            }
        }
        
        # Iterate through validation dataset
        for image_id in self.val_dataset.image_ids:
            y_true, _ = self.get_ground_truth(image_id)
            y_pred, _ = self.get_predictions(image_id)
            
            # Loop through each class
            for loop_id in range(32):
                class_id = loop_id + 1  # Convert to 1-indexed class ID
                
                # Compute intersection and union for current class
                intersection = np.logical_and(y_true[loop_id], y_pred[loop_id]).sum()
                total = y_true[loop_id].sum() + y_pred[loop_id].sum()
                
                # Add to overall dice score
                dice_scores['overall']['numerator'] += 2 * intersection
                dice_scores['overall']['denominator'] += total
                
                # Add to class-specific dice score
                dice_scores['by_class'][class_id]['numerator'] += 2 * intersection
                dice_scores['by_class'][class_id]['denominator'] += total
                
                # Add to tooth-type dice score
                tooth_type = self.TOOTH_TYPE_MAPPING[class_id]
                dice_scores['by_type'][tooth_type]['numerator'] += 2 * intersection
                dice_scores['by_type'][tooth_type]['denominator'] += total
        
        # Compute final dice scores
        final_scores = {
            'overall': self._compute_final_dice(dice_scores['overall']),
            'by_class': {
                class_id: self._compute_final_dice(scores) 
                for class_id, scores in dice_scores['by_class'].items()
            },
            'by_type': {
                tooth_type: self._compute_final_dice(scores)
                for tooth_type, scores in dice_scores['by_type'].items()
            }
        }
        
        return final_scores

    def _compute_final_dice(self, scores):
        """Helper method to compute final dice score from numerator and denominator"""
        return scores['numerator'] / scores['denominator'] if scores['denominator'] > 0 else 0.0


    # def visualize_prediction(self, image_id, is_bbox=True):
    #     """Visualize model predictions for given image."""
    #     try:

    #         # Get predictions and load image
    #         print("Getting predictions...")
    #         masks, boxes = self.get_predictions(image_id)
            
    #         print("Loading image...")
    #         pil_image = self.val_dataset.load_image(image_id)
    #         print(f"PIL Image type: {type(pil_image)}")
    #         print(f"PIL Image attributes: {dir(pil_image)}")
            
    #         # First resize the image like in __getitem__
    #         print("Resizing image...")
    #         pil_image = pil_image.resize(self.val_dataset.input_size, resample=Image.BILINEAR)
            
    #         print("Converting to array...")
    #         image_array = np.array(pil_image)
            
    #         # Convert grayscale to RGB if necessary
    #         if len(image_array.shape) == 2:
    #             image_array = np.stack([image_array] * 3, axis=-1)
            
    #         # Create color map (one distinct color per class)
    #         colors = plt.cm.rainbow(np.linspace(0, 1, 32))
            
    #         # Create figure and axes
    #         fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    #         ax.imshow(image_array, cmap='gray')
            
    #         # Plot masks and boxes for each class
    #         legend_elements = []
    #         for idx in range(32):
    #             class_id = idx + 1
    #             color = colors[idx]
                
    #             # Add mask if exists
    #         if masks[idx].any():
    #             mask = masks[idx][..., np.newaxis]  # Add channel dimension: (64, 64, 1)
    #             mask = np.repeat(mask, 3, axis=2)    # Repeat to match RGB: (64, 64, 3)
    #             masked = np.ones_like(image_array) * color[:3]
    #             ax.imshow(np.ma.masked_array(masked, ~mask), alpha=0.3)
                
    #             # Add box if requested and exists
    #             if is_bbox and boxes[idx].any():
    #                 x1, y1, x2, y2 = boxes[idx]
    #                 rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
    #                                 fill=False, color=color, linewidth=2)
    #                 ax.add_patch(rect)
                
    #             # Add to legend if class has either mask or box
    #             if masks[idx].any() or (is_bbox and boxes[idx].any()):
    #                 legend_elements.append(plt.Line2D([0], [0], color=color, 
    #                                             label=f'Tooth {class_id}'))
            
    #         # Add legend
    #         ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
    #                 loc='upper left', borderaxespad=0.)
            
    #         ax.set_title(f'Predictions for Image {image_id}')
    #         ax.axis('off')
    #         plt.tight_layout()
    #         plt.show()

    #     except Exception as e:
    #         print(f"Error occurred: {type(e).__name__}")
    #         print(f"Error message: {str(e)}")
    #         print(f"Current line number: {e.__traceback__.tb_lineno}")
    #         raise  # Re-raise the exception after printing debug info



    def visualize_prediction(self, image_id, is_bbox=True):
        """Visualize model predictions for given image."""
        try:
            print("Starting visualization...")
            
            # Get predictions and load image
            print("Before get_predictions...")
            masks, boxes = self.get_predictions(image_id)
            print("After get_predictions...")
            
            print("Before load_image...")
            temp_image = self.val_dataset.load_image(image_id)
            print("After load_image...")
            print(f"Type: {type(temp_image)}")
            
        except Exception as e:
            print(f"Error occurred at line {e.__traceback__.tb_lineno}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise