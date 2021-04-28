"""
Custom PyTorch dataset for instance segmentation.
"""

import numpy as np
from torchvision.datasets import CocoDetection


class FloorplanDataset(CocoDetection):
    def __init__(
            self, root, annFile,
            transform=None, target_transform=None, transforms=None):
        # call parent's init
        CocoDetection.__init__(
            self, root, annFile, transform, target_transform, transforms)

    def __getitem__(self, index):
        # load image and annotations at corresponding index
        ID = self.ids[index]
        img = CocoDetection._load_image(self, ID)
        anns = CocoDetection._load_target(self, ID)

        img = np.array(img)  # (H,W,C) RGB image

        # construct stuff from annotations list
        boxes = []
        labels = []
        areas = []
        masks = []

        for ann in anns:
            # bbox in Pascal VOC format - [x_min, y_min, x_max, y_max]
            # TODO: convert coords to ints?
            x, y, width, height = ann['bbox']
            bbox = [x, y, x+width, y+height]
            boxes.append(bbox)

            # get the room label from category_id
            label = ann['category_id']
            labels.append(label)

            # bbox area
            # TODO: abs()?
            # TODO: int instead of float for pixels?
            area = ann['area']
            areas.append(area)

            # construct binary masks using segmentation polygons
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        # essentials
        # TODO: int for coords instead of float?
        boxes = np.array(boxes, dtype=np.float)
        labels = np.array(labels, dtype=np.int64)
        masks = np.array(masks, dtype=np.uint8)
        # for evaluation script
        image_id = np.array(index, dtype=np.int64)
        area = np.array(areas, dtype=np.float)
        iscrowd = np.zeros(len(anns), dtype=np.uint8)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def cat_name(self, cat_id):
        """Return category name corresponding to its id."""
        cat = self.coco.loadCats([cat_id])[0]
        return cat['name']
