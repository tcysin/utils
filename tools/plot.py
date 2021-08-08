import argparse
import sys
from operator import itemgetter
from pathlib import Path

# TODO profile the timing of this script

def plot_coco(src, file, out, suffix='plot', **kwargs):
    # HEAVY IMPORTS
    # -------------------------------------------------------------------------
    import numpy as np
    from PIL import Image
    from pycocotools.coco import COCO
    from tqdm import tqdm

    # add script's grandparent dir to PYTHONPATH in order to find src module
    home = sys.path[0]  # **/utils/tools/
    parent = Path(home).parent  # **/utils/
    sys.path.append(str(parent))

    from src.utils import poly2int, coco2pascal, draw_boxes


    # FUNCTION BODY
    # -------------------------------------------------------------------------
    # load COCO dataset
    coco = COCO(file)

    # create a mapping between category id and name
    cat_records = coco.loadCats(coco.getCatIds())
    id2name = {
        cat['id']: cat['name']
        for cat in cat_records
    }

    # get the list of image records
    image_records = coco.loadImgs(coco.getImgIds())

    # TODO switch tqdm from images to annotations - is this better?
    # for each image record
    for record in tqdm(image_records):
        # load an image, convert to array
        path = src / record['file_name']
        pil_image = Image.open(path)
        image = np.asarray(pil_image)

        # load a list of annotations for this image
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[record['id']]))

        # get a list of categories
        cat_ids = map(itemgetter('category_id'), anns)
        cat_names = [id2name[cat] for cat in cat_ids]

        # get a list of boxes in pascal VOC format
        boxes = map(itemgetter('bbox'), anns)
        # convert from Coco to Pascal VOC format
        boxes = map(coco2pascal, boxes)
        boxes = map(poly2int, boxes)  # convert to integer coordinates
        boxes = list(boxes)
        # draw boxes on top of an image
        # TODO: add unique colors for each category
        plot = draw_boxes(image, boxes, texts=cat_names, **kwargs)

        # TODO if needed, iteratively construct and plot segmentation masks

        # save the plot to output directory
        name = path.stem + '_' + suffix + path.suffix
        Image.fromarray(plot).save(out / name)


if __name__ == '__main__':

    # HELPER FUNCTIONS FOR ARGPARSER
    # -------------------------------------------------------------------------
    def color_tuple(x):
        try:
            x = x[1:-1]  # strip braces
            b, g, r = map(int, x.split(','))  # extract BGR components
        except:
            raise argparse.ArgumentTypeError(
                'Color tuple must be in (B,G,R) format.')

        b_ok = 0 <= b <= 255
        g_ok = 0 <= g <= 255
        r_ok = 0 <= r <= 255

        if not (b_ok and g_ok and r_ok):
            raise argparse.ArgumentTypeError(
                'Each argument in color tuple must be in [0,255].')

        return (b, g, r)

    # PARSE AND VERIFY ARGUMENTS
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Plot images with bounding boxes and/or masks from Coco dataset.')

    parser.add_argument(
        'src', type=Path, help='source directory with image files')
    parser.add_argument(
        'file', type=Path, help='json file with COCO annotations')
    parser.add_argument(
        'out', type=Path,
        help='output directory for plots')
    parser.add_argument(
        '--suffix', default='plot',
        help='suffix for plot filenames (default: "plot")')
    # keyword arguments for drawing routine
    parser.add_argument(
        '--thickness', type=int, default=5,
        help='thickness (px) of lines that make up the bounding box (default: 5)')
    parser.add_argument(
        '--color', type=color_tuple, default=(75, 25, 230),
        help='BGR color tuple for bounding boxes (default: (75,25,230))')
    parser.add_argument(
        '--font-scale', type=float, default=1.,
        help='factor that multiplies font-specific base size (default: 1.0)')
    parser.add_argument(
        '--font-thickness', type=int, default=1,
        help='thickness (px) of lines to draw text (default: 1)')

    args = parser.parse_args()

    plot_coco(
        args.src, args.file, args.out, args.suffix,
        thickness = args.thickness,
        color = args.color,
        font_scale = args.font_scale,
        font_thickness = args.font_thickness,
    )
