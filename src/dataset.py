"""
Custom PyTorch dataset for instance segmentation.
"""

import numpy as np
import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as TF


class InstanceSegmentation(CocoDetection):
    """Custom dataset for working with instance segmentation tasks.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to COCO annotation file.
        albumentations (Compose): augmentation pipeline from `albumentations`.
            Must be configured to work with bounding boxes in `pascal_voc`
            format. `label_fields` parameter must contain `labels` value.
        as_tensors (bool): whether to convert image and target arrays to tensors.
            Image tensor will be normalized to [0,1].
    """

    def __init__(self, root, annFile, albumentations=None, as_tensors=False):
        CocoDetection.__init__(self, root, annFile)
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
            target (dict): (albumented) target dictionary. Fields are:
                'boxes': (N, 4) float array of bounding boxes.
                'labels': (N,) int64 array of labels.
                'masks': (N, H, W) uin8 array of binary {0, 1} masks.
                'image_id': (1,) int64 array with image id (from annotaions).
                'area': (N,) float array with bbox areas.              
        """

        # load image and annotations at corresponding index
        id_ = self.ids[index]
        image = CocoDetection._load_image(self, id_)
        anns = CocoDetection._load_target(self, id_)

        image = np.array(image)  # (H,W,C) RGB image array

        # construct stuff from annotations list
        boxes = []
        labels = []
        areas = []
        masks = []

        for ann in anns:
            # bbox in Pascal VOC format -- [x_min, y_min, x_max, y_max]
            x, y, width, height = ann['bbox']
            bbox = [int(x), int(y), int(x + width), int(y + height)]
            boxes.append(bbox)

            label = ann['category_id']
            labels.append(label)

            # bbox area
            area = abs(int(ann['area']))
            areas.append(area)

            # construct binary masks using segmentation polygons
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        # if required, apply albumentations on arrays
        if self.albumentations is not None:
            albumented = self.albumentations(
                image=image, masks=masks, bboxes=boxes, labels=labels)

            image = albumented['image']
            boxes = albumented['bboxes']
            labels = albumented['labels']
            masks = albumented['masks']
            # re-compute areas for new set of masks
            areas = [np.count_nonzero(mask) for mask in masks]

        # inputs for Mask R-CNN
        boxes = np.array(boxes, dtype=np.float)
        labels = np.array(labels, dtype=np.int64)
        masks = np.array(masks, dtype=np.uint8)
        # for evaluation script
        image_id = np.array([id_], dtype=np.int64)
        area = np.array(areas, dtype=np.float)
        iscrowd = np.zeros(len(boxes), dtype=np.uint8)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        # convert numpy arrays to tensors
        if self.as_tensors:
            image = TF.to_tensor(image)

            for key in target:
                target[key] = torch.from_numpy(target[key])

        return image, target

    def cat_name(self, cat_id):
        """Return category name corresponding to its id."""
        cat = self.coco.loadCats([cat_id])[0]
        return cat['name']

    def img_name(self, img_id):
        """Return image filename given its id."""
        image = self.coco.loadImgs([img_id])[0]
        return image['file_name']
