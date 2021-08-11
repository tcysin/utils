import argparse
from pathlib import Path

from _base import pad_box


def apartments2camvid(src, file, out, pad=0):
    """
    Extract rectangular ROI and segmentation mask for each apartment.

    We take each apartment and get ROI and segmentation mask crops using
    its box coordinates.
    """

    # HEAVY IMPORTS
    # ------------------------------------------------------------------
    import numpy as np
    from PIL import Image
    from pycocotools.coco import COCO
    from tqdm import tqdm

    # FUNCTION BODY
    # ------------------------------------------------------------------
    # set up output dirs
    IMAGES_DIR = out / 'images'
    IMAGES_DIR.mkdir()
    LABELS_DIR = out / 'labels'
    LABELS_DIR.mkdir()

    # load COCO dataset
    coco = COCO(file)

    # TODO: set up category naming for masks

    # get the list of image records
    image_records = coco.loadImgs(coco.getImgIds())

    for record in tqdm(image_records):
        # load corresponding image, convert to array
        fn = src / record['file_name']
        pil_image = Image.open(fn)
        image = np.asarray(pil_image)

        # load a list of anotations for this image record
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[record['id']]))

        # for each annotation
        for ann in anns:
            # get box coordinates
            x1, y1, w, h = map(int, ann['bbox'])
            x2 = x1 + w
            y2 = y1 + h

            # (optional) adjust coordinates
            x1, y1, x2, y2 = pad_box(
                [x1, y1, x2, y2], pad, pil_image.height, pil_image.width)

            # crop out ROI using coords and save it
            roi = image[y1:y2, x1:x2]
            name_roi = fn.stem + '_' + str(ann['id']) + fn.suffix
            Image.fromarray(roi).save(IMAGES_DIR / name_roi)

            # instantiate binary mask, crop it using coords
            mask = coco.annToMask(ann)
            mask = mask[y1:y2, x1:x2]
            # TODO set values to correspond with category_id
            mask[mask > 0] = 1
            # save the mask
            name_mask = Path(name_roi).stem + '_P.png'
            Image.fromarray(mask).save(LABELS_DIR / name_mask)

    # save a codes.txt file (index - category name), with background as 0
    with open(out / 'codes.txt', 'w') as f:
        f.write('background\n')
        f.write('apartment\n')


if __name__ == '__main__':

    # PARSE AND VERIFY ARGUMENTS
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Extract rectangular ROI and segmentation mask for each apartment.')

    parser.add_argument(
        'src', type=Path, help='source directory with image files')
    parser.add_argument(
        'file', type=Path, help='json file with COCO annotations')
    parser.add_argument(
        'out', type=Path,
        help='output directory for resulting images and labels (masks)')
    parser.add_argument(
        '--pad', type=int, default=0,
        help='by how much to pad (px) the final contour (default: 0)'
    )

    args = parser.parse_args()

    apartments2camvid(args.src, args.file, args.out, args.pad)
