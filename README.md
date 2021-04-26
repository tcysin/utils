# utils
Utility package for working with [instance segmentation in PyTorch](
    https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).

Contains two modules:

- `utils` with functions for drawing *bounding boxes* and *segmentation masks*
- `dataset` with generic *FloorplanDataset*, which extends [torchvision.datasets.CocoDetection](
    https://pytorch.org/vision/stable/_modules/torchvision/datasets/coco.html#CocoDetection)
