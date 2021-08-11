import json
from math import inf


def clamp(x, smallest=-inf, largest=inf):
    """Return x clamped to the [smallest, largest] interval."""
    return max(smallest, min(x, largest))

def pad_box(box_pascal, pad, height, width):
    """Safely pad bounding box (Pascal VOC) coordinates."""

    x1, y1, x2, y2 = box_pascal

    x1 = clamp(x1 - pad, smallest=0)
    y1 = clamp(y1 - pad, smallest=0)
    x2 = clamp(x2 + pad, largest=width)
    y2 = clamp(y2 + pad, largest=height)

    return x1, y1, x2, y2


def load_coco(path):
    """
    Load JSON file with Coco annotations and return
    `(images, annotations, categories, licenses, info)` tuple.
    """

    # load json with COCO annotations
    with open(path, 'r') as fin:
        coco = json.load(fin)

    images = coco.get('images')
    annotations = coco.get('annotations')
    categories = coco.get('categories')
    licenses = coco.get('licenses')
    info = coco.get('info')

    return images, annotations, categories, licenses, info


def save_coco(path, images, annotations, categories, licenses, info):
    """Save new Coco annotation file to given destination path."""

    data = {
        'info': info,
        'images': images,
        'annotations': annotations,
        'categories': categories,
        'licenses': licenses,
    }

    with open(path, 'w', encoding='UTF-8') as fout:
        json.dump(data, fout, indent=1)
