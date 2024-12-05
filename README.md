# Dental Segmentation and Disease Detection

This project uses conputer vision models on panoramic dental xrays to:

- identify teeth with bounding boxes
- identify teeth with segmentation masks
- predict dental diseases

## Source Data

We primarily use the dataset from the 2023 DENTEX Challenge organized by MICCAI.

- [DENTEX dataset is available on huggingface](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)

Data includes 3 separate folders each with 600-700 different images

#### quadrant dataset

- includes bounding boxes and segmentations for each of 4 teeth quadrants
- Note that segmentations are very approximate in this dataset.
- quadrants are 0-indexed

#### quadrant enumeration dataset

- includes bounding boxes and segmentations for each individual tooth
- category_id_1 is the 0-indexed quadrant
- category_id_2 is the 0-indexed tooth number

#### quadrant enumeration disease dataset

- includes bounding box and segmentation only for teeth that have disease
- disease is represented by category_id_3 label
- disease categories have the following definitions:

  categories = [
  {"id": 0, "name": "Impacted", "supercategory": "Impacted"},
  {"id": 1, "name": "Caries", "supercategory": "Caries"},
  {"id": 2, "name": "Periapical Lesion", "supercategory": "Periapical Lesion"},
  {"id": 3, "name": "Deep Caries", "supercategory": "Deep Caries"},
  ]

#### FDI tooth numbering

The standard practice for labelling teeth is a 2-digit number.  
The first digit represents the quadrant of the moth.  
The second digit represents the tooth within the qudrant, starting from the front/center of the mouth and counting up to the sides/back.

## References

We build on work done in two projects related to the DENTEX Challenge:

- [Intergrated Segmentation and Detection Models for Dentex Challenge 2023](https://arxiv.org/abs/2308.14161)
- - [github](https://github.com/xyzlancehe/DentexSegAndDet)
- [Diffusion-Based Hierarchical Multi-Label Object Detection to Analyze Panoramic Dental X-rays](https://arxiv.org/abs/2303.06500)
- - [github](https://github.com/ibrahimethemhamamci/HierarchicalDet)

## Other

We are also interested in a similar project using a different dataset.

- [Instance Segmentation and Teeth Classification in Panoramic X-rays](https://arxiv.org/abs/2406.03747)
- - data and code are linked in the paper
    This paper focusing only on segmentation and the data does not include disease labels. We are interested in learning from the segmentation techniques used in the paper, and in using the data as an alternative set for validation of segmentation omodels trained on the DENTEX dataset.

# Process

## Data Processing

1. We resize the raw image data to a standard 2048x1024 size to make model training more efficient.

- Raw images vary in size, with mosy around 2900x1300. We choose 2048x1024 as binary representations that approximately maintian the aspect ratio.

2. We also create a set of square 2048x2048 images by adding blank padding to the 2048x1024 images, because square images are necessary in some models.

3. We apply the necessary transformations to the annotations metadata, producing COCO and YOLO formatted metadata for both rectangual and square image datasets.

# COCO vs YOLO Annotation Formats

## COCO Format

COCO (Common Objects in Context) is a popular dataset format for object detection, segmentation, and keypoint detection. It is designed to handle a wide range of annotation types.

### Structure

COCO annotations are stored in a JSON file with the following key components:

1. **Images**: Metadata for each image, including:
   - `id`: A unique identifier for the image.
   - `file_name`: The image file name.
   - `width` and `height`: Image dimensions.
2. **Annotations**: Object-level data for each image, including:
   - `image_id`: The `id` of the corresponding image.
   - `bbox`: The bounding box `[x_min, y_min, width, height]`.
   - `segmentation`: A list of polygons describing the object shape.
   - `category_id`: The class of the object (0-indexed).
3. **Categories**: A list of classes with their `id` and `name`.

### Example

```json
{
  "images": [
    { "id": 1, "file_name": "image1.jpg", "width": 1024, "height": 768 }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [100, 200, 300, 400],
      "segmentation": [[100, 200, 400, 200, 400, 600, 100, 600]]
    }
  ],
  "categories": [{ "id": 0, "name": "category1" }]
}
```

## YOLO Format

YOLO (You Only Look Once) uses a simple and lightweight annotation format optimized for real-time object detection. Each image has a corresponding text file containing its bounding box annotations.

### File Structure

1. **One `.txt` File Per Image**:

   - Each image has a `.txt` file with the same base name as the image (e.g., `image1.jpg` → `image1.txt`).
   - The file contains one line per object in the image.

2. **Annotation Format**:
   Each line in the `.txt` file represents one object and follows this structure:

- `class_id`: The class of the object (0-indexed).
- `x_center`: The normalized x-coordinate of the bounding box center (`[0, 1]`).
- `y_center`: The normalized y-coordinate of the bounding box center (`[0, 1]`).
- `width`: The normalized width of the bounding box (`[0, 1]`).
- `height`: The normalized height of the bounding box (`[0, 1]`).

3. **Classes File**:

- A separate file named `classes.txt` lists all class names in 0-indexed order.
- Each line corresponds to a class ID.

### Example

#### `image1.txt`

```
0 0.631684 0.421661 0.270728 0.148414
1 0.367301 0.427598 0.262267 0.141290
2 0.646489 0.533268 0.283418 0.191157
3 0.355668 0.530300 0.304569 0.197093
```

#### `classes.txt`

```
Quadrant 1
Quadrant 2
Quadrant 3
Quadrant 4
```

### Key Notes

- **Normalized Coordinates**: Bounding box values are normalized to the range `[0, 1]` relative to the image dimensions.
- **Class Indexing**: Class IDs must be 0-indexed and correspond to the order in `classes.txt`.
- **Multiple Objects**: Multiple objects in the same image are represented as separate lines in the `.txt` file.

### Advantages

- Compact and simple format for fast processing.
- Directly compatible with YOLO training pipelines.

### Limitations

- Does not support advanced annotations like segmentation or keypoints.
- Each object is limited to a single class label.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/radubach/dental_segmentation/blob/main/data_processing.ipynb)

