"""
Module with drawing routines for instance segmentation tasks.
"""

import numpy as np
import cv2 as cv


# default padding (px)
OFFSET = 2


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


def draw_boxes(
        image, boxes, thickness=2, color=(0, 255, 0),
        texts=None, font_scale=1, font_thickness=1,
        offset=OFFSET):
    """Return image with bounding boxes drawn on top of it.

    Params:
        image (ndarray): uint8 BGR image w. shape (H,W,C).
        boxes (list): bounding box coordinates.
            Contains N coordinates, which are [x0, y0, x1, y1].
        thickness (int): thickness (px) of lines that make up the rectangle.
        color (tuple): BGR.
        texts (list): N texts for each bbox prediction.
        font_scale (float): factor that multiplies font-specific base size.
        font_thickness (int): thickness (px) of lines to draw text.
        offset (int): by how much px to move texts up from bbox. 

    Returns:
        uint8 BGR image array with bboxes.
    """

    offset = offset + thickness

    image = image.copy()

    # add original boxes to the image
    for bbox in boxes:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        image = cv.rectangle(image, (x0, y0), (x1, y1), color, thickness)

    # add texts to the image if they are present
    if texts is not None:

        for bbox, text in zip(boxes, texts):
            x0, y0, *_ = [int(v) for v in bbox]

            # place text on top of the upper left corner of the box
            origin = (x0, y0 - offset)

            image = cv.putText(
                image, text=text, org=origin,
                fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
                color=color, thickness=font_thickness)

    return image
