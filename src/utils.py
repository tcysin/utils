"""
Module with various helper routines for instance segmentation tasks.
"""
from itertools import cycle

import numpy as np
import cv2 as cv

# TODO refactor comments according to python PEP
# TODO make a wrapper to convert images to BGR before funcs and to RGB after


# default padding (px)
OFFSET = 2

# distinct colors in BGR format
COLORS = (
    (75, 25, 230),  # red
    (75, 180, 60),  # green
    (200, 130, 0),  # blue
    (48, 130, 245),  # orange
    (230, 50, 240),  # magenta
    (0, 0, 128),  # maroon
    (128, 0, 0)  # navy
)

# TODO def extract_foreground(image, mask)
# TODO def crop_foreground(image, mask)


def draw_mask(
        image, mask, alpha, color,
        text=None, font_scale=1, font_thickness=1, offset=OFFSET):
    """Return image array with segmentation mask drawn on top.

    Params:
        image (ndarray): uint8 BGR image w. shape (H,W,C).
        mask (ndarray): uint8 binary mask w. shape (H,W).
            Values above zero are treated as 1.
        alpha (float): weight of the image array element, must be in [0,1].
            Used for blending original image and colored segmentation mask.
        color (tuple): BGR color of the mask.
        text (str): text to put on top of the final image.
        font_scale (float): factor that multiplies font-specific base size.
        font_thickness (int): thickness (px) of lines to draw text.

    Returns:
        uint8 BGR image array with segmentation mask.
    """

    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # create opencv-compatible binary mask of a segment and its inverse
    mask = mask.copy()
    mask[mask > 0] = 255
    mask_inv = cv.bitwise_not(mask)

    # black out segment area in original image
    background = cv.bitwise_and(image, image, mask=mask_inv)

    # only blend area under segmentation mask
    foreground = cv.bitwise_and(image, image, mask=mask)

    # create a color array
    # TODO: optimize
    c_arr = np.zeros_like(foreground)
    c_arr += np.array(color, dtype=np.uint8)  # (H,W,C) + (1,1,C)
    c_arr[mask < 255] = 0  # black out background part

    # blend segmentation ROI with its color array
    blended_fg = cv.addWeighted(foreground, alpha, c_arr, 1-alpha, 0)

    # put blended patch(es) on original background
    result = cv.add(blended_fg, background)

    if text is not None:
        # place the text on top of the mask - upper left corner
        rows, cols = np.nonzero(mask)
        x0, y0 = np.min(cols), np.min(rows)
        origin = (x0, y0 - offset)
        result = cv.putText(
            result, text, org=origin,
            fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
            color=color, thickness=font_thickness)

    return cv.cvtColor(result, cv.COLOR_BGR2RGB)

# TODO: let user pass a list of colors - one for each box


def draw_boxes(
        image, boxes, thickness=2, color=(0, 255, 0), colors=None,
        texts=None, font_scale=1, font_thickness=1,
        offset=OFFSET, with_ids=True):
    """Return image with bounding boxes drawn on top of it.

    Params:
        image (ndarray): uint8 RGB image w. shape (H,W,C).
        boxes (list): bounding box coordinates.
            Contains N coordinates, which are [x0, y0, x1, y1] - Pascal VOC 
            format.
        thickness (int): thickness (px) of lines that make up the rectangle.
        color (tuple): color of bounding box borders in BGR format.
            Applies this color to all bounding boxes.
        colors (iterable): BGR color tuples for bounding box borders.
            If provided, cycles through this iterable to select colors for 
            corresponding boxes. Overrides `color` option.
        texts (list): N texts for each bbox prediction.
        font_scale (float): factor that multiplies font-specific base size.
        font_thickness (int): thickness (px) of lines to draw text.
        offset (int): by how much px to move texts up from upper left corner of
            the box.
        with_ids (bool): whether to plot IDs for each box.
            Box ID corresponds to its index in boxes list.

    Returns:
        uint8 RGB image array with boxes drawn on top.
    """

    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # generate IDs and texts for boxes if necessary
    n = len(boxes)
    ids = range(n)

    if texts is None:
        texts = ('',) * n
    if with_ids:
        texts = (f'{id_} {txt}' for id_, txt in zip(ids, texts))

    # re-calculate vertical text offset to take into account box thickness
    offset = offset + thickness

    # set up color cycle
    if colors is None:
        colors = (color,)
    color_cycle = cycle(colors)

    # add boxes and texts to the image
    for box, text, c in zip(boxes, texts, color_cycle):

        # draw bounding box
        x0, y0, x1, y1 = box2int(box)
        # TODO: does rectangle re-use an image or create a new one?
        cv.rectangle(image, (x0, y0), (x1, y1), c, thickness)

        # draw text and place it on top of the upper left corner of the box
        origin = (x0, y0 - offset)
        cv.putText(
            image, text=text, org=origin,
            fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
            color=c, thickness=font_thickness)

    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def box2int(box):
    """Return bounding box with int coordinates."""
    return [int(coord) for coord in box]


def area_pascal(box):
    """
    Compute and return the area of the bounding box.

    The box is in Pascal VOC format: [x1, y1, x2, y2].
    """

    x1, y1, x2, y2 = box2int(box)
    height = y2 - y1
    width = x2 - x1

    return height * width


def coco2pascal(box):
    """Convert bounding box coordinates from Coco format to Pascal VOC.

    Go from [x, y, width, height] to [x1, y1, x2, y2].
    """

    x, y, width, height = box
    return [x, y, x + width, y + height]
