"""
Custom PyTorch dataset for instance segmentation.
"""

from collections import defaultdict

import numpy as np
import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as TF

from .utils import area_pascal


class BaseDataset(CocoDetection):
    """
    Base class for Coco-style object detection, semantic and instance
    segmentation datasets.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that takes in RGB
            image array and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in a list of Coco annotations and transforms it.
        transforms (callable, optional): A function/transform that takes input
            sample and annotations as entry and returns a transformed version.
        albumentations (Compose): augmentation pipeline from `albumentations`.
            Must be configured to work with bounding boxes in Pascal VOC
            format; `label_fields` parameter must contain `labels` value.
        as_tensors (bool): whether to convert image and target arrays to tensors.
            Image tensor will be normalized to [0,1].
    """

    def __init__(
            self, root, annFile,
            transform=None, target_transform=None, transforms=None,
            albumentations=None, as_tensors=False):
        CocoDetection.__init__(
            self, root, annFile, transform, target_transform, transforms)
        self.albumentations = albumentations
        self.as_tensors = as_tensors

    def __getitem__(self, index):
        """Return (image, target) tuple at given index.

        Loads image and corresponding annotations, optionally applies
        transformations, converts the results to ndarrays and optionally 
        transfers them to tensors.

        Returns:
            image (ndarray): (albumented) RGB image array.
                Has shape (H, W, C). Image is loaded as-is, without additional 
                normalization / processing.
            target (dict): (albumented) target dictionary.
                Fields depend on the type of problem at hand. See child classes
                for more info. Possible fields include:
                    `boxes`: (N, 4) float array of bounding boxes.
                    `labels`: (N,) int64 array of labels.
                    `masks`: (N, H, W) uint8 array of binary {0, 1} masks.
                    `image_id`: (1,) int64 array with image id (from annotaions).
                    `area`: (N,) float array with bbox areas.              
        """

        # load image and annotations at corresponding index
        id_ = self.ids[index]
        image = np.array(self._load_image(id_))  # (H,W,C) RGB image array
        anns = self._load_target(id_)

        # apply transforms on RGB image array and Coco annotations list
        if self.transforms is not None:
            image, anns = self.transforms(image, anns)

        # initialize target dict
        target = defaultdict(list)

        # process Coco annotations and populate target dict
        for ann in anns:
            self._add_annotation(ann, target)

        # if required, apply albumentations on arrays in target
        if self.albumentations is not None:
            image, target = self._apply_albumentations(image, target)

        # add misc information for PyTorch evaluation scripts
        N = len(target['boxes']) if 'boxes' in target else len(target['masks'])
        target['image_id'] = np.array([id_], dtype=np.int64)
        target['iscrowd'] = np.zeros(N, dtype=np.uint8)

        # convert entires in target to numpy arrays with correct data types
        self._convert_dtypes(target)

        # convert numpy arrays to tensors
        if self.as_tensors:
            image = TF.to_tensor(image)

            for key in target:
                target[key] = torch.from_numpy(target[key])

        return image, target

    def _add_annotation(self, annotation, target):
        """
        Convert Coco annotation to correct format and append to corresponding
        lists in `target` dictionary.

        Extracts `label` and `area` parts of Coco annotation, then calls 
        `self.add_annotation` method of a child class to convert and add 
        the rest. 
        """

        label = annotation['category_id']
        target['labels'].append(label)

        area = abs(int(annotation['area']))
        target['area'].append(area)

        self.add_box(annotation, target)
        self.add_mask(annotation, target)

    def add_box(self, annotation, target):
        """
        Override this method in a child class to convert bounding box and add
        it to target dictionary.
        """
        pass
    
    def add_mask(self, annotation, target):
        """
        Override this method in a child class to generate binary mask and add
        it to target dictionary.
        """
        pass

    def _apply_albumentations(self, image, target):
        # either masks, or boxes or both will be present
        masks = target.get('masks')
        boxes = target.get('boxes')
        labels = target['labels']

        albumented = self.albumentations(
            image=image, masks=masks, bboxes=boxes, labels=labels)

        image = albumented['image']
        masks = albumented.get('masks')
        boxes = albumented.get('bboxes')

        # re-compute areas for new sets of masks / boxes
        if masks is not None:
            target['area'] = [np.count_nonzero(mask) for mask in masks]
        elif boxes is not None:
            target['area'] = [area_pascal(box) for box in boxes]

        # update target with new masks, boxes and labels
        if masks is not None:
            target['masks'] = masks
        if boxes is not None:
            target['boxes'] = boxes
        target['labels'] = albumented['labels']

        return image, target

    @staticmethod
    def _convert_dtypes(target):
        target['labels'] = np.array(target['labels'], dtype=np.int64)

        if 'boxes' in target:
            target['boxes'] = np.array(target['boxes'], dtype=np.float)

        if 'masks' in target:
            target['masks'] = np.array(target['masks'], dtype=np.uint8)

        target['area'] = np.array(target['area'], dtype=np.float)

    def category_name(self, category_id):
        """Return category name given its id."""

        cat = self.coco.loadCats([category_id])[0]
        return cat['name']

    def image_name(self, image_id):
        """Return image filename given its id."""

        image = self.coco.loadImgs([image_id])[0]
        return image['file_name']


class InstanceSegmentation(BaseDataset):

    def add_box(self, annotation, target):
        # convert to bbox in Pascal VOC format -- [x_min, y_min, x_max, y_max]
        x, y, width, height = annotation['bbox']
        box = [int(x), int(y), int(x + width), int(y + height)]
        target['boxes'].append(box)

    def add_mask(self, annotation, target):
        # construct binary masks using segmentation polygons
        mask = self.coco.annToMask(annotation)
        target['masks'].append(mask)


class ObjectDetection(BaseDataset):

    def add_box(self, annotation, target):
        # convert to bbox in Pascal VOC format -- [x_min, y_min, x_max, y_max]
        x, y, width, height = annotation['bbox']
        box = [int(x), int(y), int(x + width), int(y + height)]
        target['boxes'].append(box)
