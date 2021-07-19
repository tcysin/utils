# utils
Utility package for working with **object detection** problems.

## Content

- `utils` module with functions for drawing *bounding boxes* and *segmentation masks*
- `dataset` module with generic *ObjectDetection* and *InstanceSegmentation* datasets, which extend [torchvision.datasets.CocoDetection](
    https://pytorch.org/vision/stable/_modules/torchvision/datasets/coco.html#CocoDetection) for 
    object detection and instance segmentation tasks on Coco data format.
