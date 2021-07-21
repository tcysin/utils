import argparse
from operator import itemgetter
from pathlib import Path


from _base import load_coco, save_coco


def reset_index(src, dst, verbose=False):
    """
    Reset indices of images and annotations in given Coco dataset.
    
    Args:
        src (Path): JSON file with input Coco dataset.
        dst (Path): destination for new Coco dataset.
        verbose (bool): output verbosity.
    """

    # load json with COCO annotations
    images, annotations, categories, licenses, info = load_coco(src)

    if verbose:
        print(
            'Original:',
            len(images), 'images,',
            len(annotations), 'annotations.')


    # FILTER IMAGES & ANNOTATIONS
    # ------------------------------------------------------------------------
    # exclude images without anns and anns without images
    img_ids_anns = {ann['image_id'] for ann in annotations}
    img_ids_imgs = {img['id'] for img in images}
    ids = img_ids_anns & img_ids_imgs
    images = [img for img in images if img['id'] in ids]
    annotations = [ann for ann in annotations if ann['image_id'] in ids]

    if verbose:
        print(
            'After cleaning:',
            len(images), 'images,',
            len(annotations), 'annotations.')


    # RE-SET IMAGE IDS
    # ------------------------------------------------------------------------
    # re-set image IDs in 'images' and 'annotations' fields
    # create a mapping from old image IDs to new ones
    images = sorted(images, key=itemgetter('file_name'))

    old2new_images = {
        image['id']: new_id
        for (new_id, image) in enumerate(images, start=1)
    }

    # re-set image IDs in images
    for img in images:
        img['id'] = old2new_images[img['id']]

    # re-set image IDs in annotations
    for ann in annotations:
        ann['image_id'] = old2new_images[ann['image_id']]

    if verbose: print('Image IDs reset.')


    # RE-SET ANNOTATION IDS
    # ------------------------------------------------------------------------
    annotations = sorted(annotations, key=itemgetter('image_id', 'id'))
    
    old2new_anns = {
        ann['id']: new_id
        for (new_id, ann) in enumerate(annotations, start=1)
    }

    for ann in annotations:
        ann['id'] = old2new_anns[ann['id']]

    if verbose: print('Annotation IDs reset.')


    # SAVE NEW DATASET
    # ------------------------------------------------------------------------
    save_coco(dst, images, annotations, categories, licenses, info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reset image and annotation indices in given Coco dataset.'
    )
    parser.add_argument(
        'file', type=Path, help='json file with COCO annotations')
    parser.add_argument(
        '-o', '--out', type=Path, default=None,
        help="destination of the resulting file (default: file)")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="increase output verbosity")
    args = parser.parse_args()

    out = args.out if args.out is not None else args.file
    reset_index(args.file, out, args.verbose)
