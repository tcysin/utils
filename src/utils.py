"""
Module with drawing routines for instance segmentation tasks.
"""
import string
from itertools import count, cycle, islice, product

import numpy as np
import cv2 as cv


# default padding (px)
OFFSET = 2

# distinct colors in BGR format
COLORS = (
    (75, 25, 230),  # red
    (75, 180, 60),  # green
    (25, 225, 255),  # yellow
    (200, 130, 0),  # blue
    (48, 130, 245),  # orange
)

# TODO def box2int(box)
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

    return result


def ascii_gen():
    """Generate cartesian products of ascii letters (A B ... AA AB ... )"""

    for r in count(start=1):
        for tup in product(string.ascii_uppercase, repeat=r):
            yield ''.join(tup)


def ascii_ids(n):
    """Return a list of n cartesian products of ascii letters."""
    return list(islice(ascii_gen(), n))


def draw_boxes(
        image, boxes, thickness=2, color=(0, 255, 0), colors=None,
        texts=None, font_scale=1, font_thickness=1, offset=OFFSET):
    """Return image with bounding boxes drawn on top of it.

    Params:
        image (ndarray): uint8 BGR image w. shape (H,W,C).
        boxes (list): bounding box coordinates.
            Contains N coordinates, which are [x0, y0, x1, y1].
        thickness (int): thickness (px) of lines that make up the rectangle.
        color (tuple): color of bounding box borders in BGR format.
            Applies this color to all bounding boxes.
        colors (iterable): BGR color tuples for bounding box borders.
            If provided, selects colors for consecutive bboxes in a cycle.
        texts (list): N texts for each bbox prediction.
        font_scale (float): factor that multiplies font-specific base size.
        font_thickness (int): thickness (px) of lines to draw text.
        offset (int): by how much px to move texts up from bbox. 

    Returns:
        uint8 BGR image array with bboxes.
    """

    image = image.copy()

    # generate IDs and texts for boxes
    ids = ascii_ids(len(boxes))

    if texts is not None:
        texts = (f'{_id} {txt}' for _id, txt in zip(ids, texts))
    else:
        texts = (str(i) for i in ids)

    # re-calculate vertical text offset to take into account bbox thickness
    offset = offset + thickness

    # set up color cycle
    if colors is None:
        colors = (color,)
    color_cycle = cycle(colors)

    # add boxes and texts to the image
    for bbox, text, c in zip(boxes, texts, color_cycle):

        # draw bounding box
        x0, y0, x1, y1 = [int(v) for v in bbox]
        image = cv.rectangle(image, (x0, y0), (x1, y1), c, thickness)

        # draw text
        # place text on top of the upper left corner of the box
        origin = (x0, y0 - offset)
        image = cv.putText(
            image, text=text, org=origin,
            fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
            color=c, thickness=font_thickness)

    return image
