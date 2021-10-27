import argparse
from pathlib import Path

from _base import load_coco, save_coco
from reset_index import reset_index


def split_coco(src, val, out, verbose=False):
    """
    Split Coco dataset into train and val subsets.

    Args:
        src (Path): JSON file with input Coco dataset.
        val (Path): text file with image filenames in validation set.
        out (Path): output directory for train and val subsets.
        verbose (bool): output verbosity.
    """

    # load json with COCO annotations
    images, annotations, categories, licenses, info = load_coco(src)

    if verbose:
        print("Original set:", len(images), "images,", len(annotations), "annotations.")

    # load fnames from val file
    with open(val, "r") as fin:
        VAL = {line.strip() for line in fin}

    # image ids in validation set
    ids_val = {im["id"] for im in images if im["file_name"] in VAL}

    # extract images and anns in train set, save new train dataset
    images_train = [im for im in images if im["id"] not in ids_val]
    anns_train = [ann for ann in annotations if ann["image_id"] not in ids_val]

    if verbose:
        print(
            "Train set:", len(images_train), "images,", len(anns_train), "annotations."
        )

    dst_train = out / "train.json"
    save_coco(dst_train, images_train, anns_train, categories, licenses, info)
    reset_index(dst_train, dst_train)

    # extract images and anns in validation set, save new val dataset
    images_val = [im for im in images if im["id"] in ids_val]
    anns_val = [ann for ann in annotations if ann["image_id"] in ids_val]

    if verbose:
        print(
            "Validation set:", len(images_val), "images,", len(anns_val), "annotations."
        )

    dst_val = out / "val.json"
    save_coco(dst_val, images_val, anns_val, categories, licenses, info)
    reset_index(dst_val, dst_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split COCO dataset into train and val sets."
    )
    parser.add_argument("file", type=Path, help="json file with COCO annotations")
    parser.add_argument(
        "val", type=Path, help="text file with filenames in validation set"
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="save resulting train and val files in this directory; defaults to parent dir of file",
    )
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    args = parser.parse_args()

    # CHECKS
    if args.out:
        assert args.out.is_dir(), "OUT must be a directory"

    out = args.out if args.out is not None else args.file.parent
    split_coco(args.file, args.val, out, args.verbose)
