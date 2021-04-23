"""
Custom PyTorch dataset for instance segmentation.

The expected layout of 'root' directory is as follows:
  root/
    images/
      cute-dog.png
      puffy-cat.png
      ...
    coco-annotations.json
"""

from pathlib import Path
from PIL import Image

from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset


# TODO: maybe transfer tensors to good device explicitly?

class FloorplanDataset(Dataset):
    def __init__(self, root, annotations_file, transforms=None):
        self.root = Path(root)
        self.img_dir = self.root / 'images'
        self.coco = COCO(self.root / annotations_file)
        self.img_ids = self.coco.getImgIds()
        self.transforms = transforms

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # load image at corresponding index
        img_id = self.img_ids[idx]
        img = self.load_image(img_id)

        # construct stuff from annotations list
        anns = self.annotations(img_id)
        boxes = []
        labels = []
        areas = []
        masks = []

        for ann in anns:
            # bbox
            x, y, width, height = ann['bbox']
            bbox = [x, y, x+width, y+height]
            boxes.append(bbox)

            # get the room label from category_id
            label = ann['category_id']
            labels.append(label)

            # bbox area
            area = ann['area']
            areas.append(area)

            # construct binary masks using segmentation polygons
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        # essentials
        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        masks = torch.tensor(masks, dtype=torch.uint8)
        # for evaluation script
        img_id_tensor = torch.tensor(img_id, dtype=torch.int64)
        area = torch.tensor(areas, dtype=torch.float)
        iscrowd = torch.zeros(len(anns), dtype=torch.uint8)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = img_id_tensor
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def get_height_and_width(self):
        # TODO
        raise NotImplementedError

    def load_image(self, image_id):
        """Load and return PIL image corresponding to given image_id.

        Note that image_id is the one in annotations file.
        """

        img_info = self.coco.loadImgs([image_id])[0]
        img_path = self.img_dir / img_info['file_name']
        img = Image.open(img_path).convert('RGB')

        return img

    def annotations(self, image_id):
        """Return a list of annotations for given image."""

        annIds = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(annIds)

        return anns

    def id_to_cat(self, cat_id):
        """Return category name corresponding to its id."""
        cat = self.coco.loadCats([cat_id])[0]
        return cat['name']
