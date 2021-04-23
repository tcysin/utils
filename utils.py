"""
Module with helper functions for instance segmentation tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


# horizontal padding (px)
PADDING_X = 15
# vertical padding (px)
PADDING_Y = 20


def draw_mask(img, mask, alpha, color):
    """Return image array with segmentation mask drawn on top.

    Params:
        img (ndarray): uint8 BGR image w. shape (H,W,C).
        mask (ndarray): uint8 binary mask w. shape (H,W).
            Values are in {0,1}.
        alpha (float): weight of the image array element, must be in [0,1].
        color (tuple): BGR color of the mask.

    Returns:
        uint8 BGR image array with segmentation mask.
    """

    # create opencv-compatible binary mask of a segment and its inverse
    mask[mask > 0] = 255
    mask_inv = cv.bitwise_not(mask)

    # black out segment area in original image
    background = cv.bitwise_and(img, img, mask=mask_inv)

    # only blend area under segmentation mask
    foreground = cv.bitwise_and(img, img, mask=mask)

    # create a color array
    c_arr = np.zeros_like(foreground)
    c_arr += np.array(color, dtype=np.uint8)  # (H,W,C) + (1,1,C)
    c_arr[mask < 255] = 0  # black out background part

    # blend segmentation ROI with its color array
    blended_fg = cv.addWeighted(foreground, alpha, c_arr, 1-alpha, 0)

    # put blended patch(es) on original background
    result = cv.add(blended_fg, background)

    return result


def draw_bboxes(img, boxes, scores=None, color=(0, 255, 0), thickness=2):
    """Return image with bounding boxes drawn on top of it.

    Params:
        img (ndarray): uint8 BGR image w. shape (H,W,C).
        boxes (list): bounding box coordinates.
            Contains N coordinates, which are [x0, y0, x1, y1].
        scores (list): N scores for each bbox prediction.
            Defaults to None.
        color (tuple): BGR.
        thickness (int): thickness of lines (px that make up the rectangle.

    Returns:
        uint8 BGR image array with bboxes.
    """

    # add original boxes to the image
    for bbox in boxes:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        img = cv.rectangle(img, (x0, y0), (x1, y1), color, thickness)

    # add scores to the image if they are present
    if scores is not None:

        for bbox, score in zip(boxes, scores):
            x0, y0, *_ = [int(v) for v in bbox]

            # pad the starting coordinates of a text a little
            x0 += PADDING_X
            y0 += PADDING_Y

            img = cv.putText(
                img, text=str(round(score, 2)), org=(x0+15, y0+20),
                fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=color, thickness=2)

    return img
