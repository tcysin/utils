import argparse
import json
import os
import yaml
from collections import defaultdict
from math import inf
from pathlib import Path
from typing import List

from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def get_files(path, exts):
    """Return sorted list of absolute paths to files in the tree rooted at path.

    Only return the files with given extensions.
    """

    result = []

    for dirpath, _, filenames in os.walk(path):
        selected = [fn for fn in filenames if fn.lower().endswith(exts)]
        abspath = Path(dirpath).resolve()
        paths = (abspath / fn for fn in selected)
        result.extend(paths)

    return sorted(result)


def get_image_files(path) -> List[Path]:
    """Return sorted list of absolute paths to image files in the tree rooted at path."""
    return get_files(path, exts=IMAGE_EXTENSIONS)


def clamp(x, smallest=-inf, largest=inf):
    """Return x clamped to the given interval."""
    return max(smallest, min(x, largest))


def to_xyxy_abs(box_xywh_rel, width, height):
    """
    Convert box coordinates from relative XYWH to absolute XYXY.

    Relative coordinates are [x_center, y_center, width, height],
    absolute are [x_min, y_min, x_max, y_max].
    """

    x_c, y_c, w, h = box_xywh_rel

    x_min = (x_c - w / 2) * width
    y_min = (y_c - h / 2) * height
    x_max = (x_c + w / 2) * width
    y_max = (y_c + h / 2) * height

    # clamp coordinates to appropriate ranges
    x_min = clamp(int(x_min), 0, width)
    y_min = clamp(int(y_min), 0, height)
    x_max = clamp(int(x_max), 0, width)
    y_max = clamp(int(y_max), 0, height)

    return [x_min, y_min, x_max, y_max]


def load_yolo_predictions(txt_path, classes=None):
    """Return a list of records from YOLO prediction file.

    Each record is a list `[label, box, score]`, where the box contains
    coordinates in YOLOv5 format. `score` can be None if text file does
    not contain scores, and `label` is either integer, or string if the
    mapping `classes` is provided.
    """

    result = []

    for line in open(txt_path, "r"):
        label, x_c, y_c, w, h, *rest = line.split()
        label = int(label)
        if classes:
            label = classes[label]
        box = [float(x_c), float(y_c), float(w), float(h)]
        score = float(rest[0]) if rest else None

        result.append([label, box, score])

    return result


def get_parser():
    """Init and return ArgumentParser for this script."""

    parser = argparse.ArgumentParser(
        description="Extract object patches from source images using YOLOv5 predictions."
    )

    parser.add_argument("data_dir", type=Path, help="source directory with image files")
    parser.add_argument(
        "labels_dir",
        type=Path,
        help="directory with corresponding txt predictions in YOLOv5 format",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="output directory for folder with cropped ROIs and predictions json file",
    )
    parser.add_argument(
        "--yaml_path",
        type=Path,
        default=None,
        help="dataset.yaml file with class names",
    )
    parser.add_argument(
        "--folder",
        default="rois",
        help='name for output folder with cropped ROIs (default: "rois")',
    )
    parser.add_argument(
        "--file",
        default="filename2objects.json",
        help='name for predictions output file in json format (default: "filename2objects.json")',
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=0,
        help="by how much pixels to pad the ROI before cropping (default: 0)",
    )
    parser.add_argument(
        "--prefix", default="", help="prefix for ROI filenames (empty by default)"
    )

    return parser


def crop_objects_yolo(
    data_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    yaml_path: Path = None,
    folder="rois",
    file="filename2objects.json",
    pad=0,
    prefix="",
    verbose=False,
):

    # prepare output directory for roi crops
    rois_dir = out_dir / folder
    rois_dir.mkdir()

    # load classes from dataset.yaml, if any
    if yaml_path:
        with open(yaml_path) as f:
            yaml_dict = yaml.safe_load(f)
            classes = yaml_dict.get("names")
    else:
        classes = None

    # set up the procedure
    filename2objects = defaultdict(list)
    obj_id = 1

    # get paths to images in given directory
    fnames = get_image_files(data_dir)
    assert fnames, f"no images in {data_dir}"

    # extract ROIs
    wrapper = tqdm if verbose else iter
    for image_path in wrapper(fnames):
        # get path to txt predictions file for this image
        txt_name = image_path.stem + ".txt"
        txt_path = labels_dir / txt_name

        # load YOLOv5 predictions
        output = load_yolo_predictions(txt_path, classes)

        # load image
        im = Image.open(image_path).convert("RGB")

        for label, box_xywh, score in output:
            # convert yolo box to [top-left-x, top-left-y, bottom-right-x, bottom-right-y]
            # in absolute coords [0, width] x [0, height]
            box_xyxy = to_xyxy_abs(box_xywh, im.width, im.height)

            # crop roi and save it
            left, upper, right, lower = box_xyxy
            im_crop = im.crop((left - pad, upper - pad, right + pad, lower + pad))
            name_obj = f"{prefix}{obj_id}_{label}.jpg"
            im_crop.save(rois_dir / name_obj)

            # create an entry for this object: [id, box_xyxy, score, label_name]
            entry = [obj_id, box_xyxy, score, label]
            filename2objects[image_path.name].append(entry)
            obj_id += 1

    # save predictions file
    with open(out_dir / file, "w") as fout:
        json.dump(filename2objects, fout, indent=1)

    if verbose:
        N = len(get_image_files(rois_dir))
        print(f"Extracted {N} ROIs.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    crop_objects_yolo(
        args.data_dir,
        args.labels_dir,
        args.out_dir,
        args.yaml_path,
        args.folder,
        args.file,
        args.pad,
        args.prefix,
        verbose=True,
    )
