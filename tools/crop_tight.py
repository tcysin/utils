import argparse
import sys

from operator import itemgetter
from pathlib import Path

from _base import clamp, load_coco, pad_box, save_coco


def subtract_from_coords(x, y, poly):
    """
    For each pair of (X, Y) coordinates in a polygon, subtract x from each X 
    coordinate and y from each Y coordinate.

    Polygon is a list of coordinates [x0, y0, x1, y1, ...].
    """

    new_poly = []

    for idx, val in enumerate(poly):
        # if even position, then value corresponds to X coordinate
        if idx % 2 == 0:
            new_poly.append(clamp(val - x, smallest=0))
        # otherwise, value corresponds to Y coordinate
        else:
            new_poly.append(clamp(val - y, smallest=0))

    return new_poly


def crop_tight(pil_image, annotations, pad=0):
    """Crop tight to boxes and return altered image and annotations."""

    # add script's grandparent dir to PYTHONPATH in order to find src module
    home = sys.path[0]  # **/utils/tools/
    parent = Path(home).parent  # **/utils/
    sys.path.append(str(parent))

    from src.utils import poly2int, coco2pascal

    # find coordinates for ROI rectangle
    boxes = map(itemgetter('bbox'), annotations)
    boxes = map(coco2pascal, boxes)
    boxes = list(map(poly2int, boxes))
    # get coords of upper left corner of ROI
    x_min = min(map(itemgetter(0), boxes))
    y_min = min(map(itemgetter(1), boxes))
    # lower right corner
    x_max = max(map(itemgetter(2), boxes))
    y_max = max(map(itemgetter(3), boxes))

    # pad coordinates a little
    x_min, y_min, x_max, y_max = pad_box(
        [x_min, y_min, x_max, y_max], pad, pil_image.height, pil_image.width
    )

    # crop ROI from image using ROI coordinates
    roi = pil_image.crop((x_min, y_min, x_max, y_max))

    # adjust coordinates for boxes and polygons in annotations
    new_annotations = []

    for ann in annotations:
        new_ann = ann.copy()

        # adjust bbox coordinates
        x, y, w, h = poly2int(new_ann['bbox'])
        new_ann['bbox'] = [x-x_min, y-y_min, w, h]

        # adjust segmentation polygon coords
        segmentation = map(poly2int, new_ann['segmentation'])
        segmentation = [
            subtract_from_coords(x_min, y_min, poly)
            for poly in segmentation
        ]
        new_ann['segmentation'] = segmentation

        # save new annotation
        new_annotations.append(new_ann)

    return roi, new_annotations


def crop_tight_coco(src, file, out, pad=0, suffix='cropped'):
    from PIL import Image
    from pycocotools.coco import COCO
    from tqdm import tqdm

    # prepare directories
    img_dir = out / 'images'
    img_dir.mkdir(exist_ok=True)

    # load COCO dataset
    coco = COCO(file)

    # get the list of image records
    image_records = coco.loadImgs(coco.getImgIds())

    new_image_records = []
    new_annotations = []

    # for each image record
    for record in tqdm(image_records):
        # load an image, convert to array
        path = src / record['file_name']
        pil_image = Image.open(path)

        # load a list of annotations for this image
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[record['id']]))

        # get cropped image and adjusted annotations
        roi, anns_cropped = crop_tight(pil_image, anns, pad)

        # save cropped image
        name_image = path.stem + '_' + suffix + path.suffix
        roi.save(img_dir / name_image)

        # save image record information
        new_record = record.copy()
        new_record['file_name'] = name_image
        new_record['width'] = roi.width
        new_record['height'] = roi.height
        new_image_records.append(new_record)

        # save annotations
        new_annotations.extend(anns_cropped)

    # load json with COCO annotations
    _, _, categories, licenses, info = load_coco(file)
    name_coco = file.stem + '_' + suffix + file.suffix
    save_coco(
        out / name_coco, new_image_records, new_annotations,
        categories, licenses, info)


if __name__ == '__main__':
    # PARSE AND VERIFY ARGUMENTS
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Crop tight to boxes and return altered images and Coco annotations.')

    parser.add_argument(
        'src', type=Path, help='source directory with image files')
    parser.add_argument(
        'file', type=Path, help='json file with COCO annotations')
    parser.add_argument(
        'out', type=Path,
        help='output directory for adjusted images and annotation file')
    parser.add_argument(
        '--pad', type=int, default=0,
        help='by how much to pad (px) the final contour (default: 0)'
    )
    parser.add_argument(
        '--suffix', default='cropped',
        help='suffix for plot filenames (default: "cropped")')

    args = parser.parse_args()

    crop_tight_coco(args.src, args.file, args.out, args.pad, args.suffix)
