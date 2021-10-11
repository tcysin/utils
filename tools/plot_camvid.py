import argparse
import sys
from pathlib import Path


def plot_camvid(
    images_dir, masks_dir, out, alpha=0.4, color=(48, 130, 245), suffix="plot"
):
    # HEAVY IMPORTS
    # -------------------------------------------------------------------------
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    # add script's grandparent dir to PYTHONPATH in order to find src module
    home = sys.path[0]  # **/utils/tools/
    parent = Path(home).parent  # **/utils/
    sys.path.append(str(parent))

    from src.utils import draw_mask

    # FUNCTION BODY
    # -------------------------------------------------------------------------
    fnames = list(sorted(images_dir.iterdir()))

    for fn in tqdm(fnames):
        # load image array
        pil_image = Image.open(fn)
        image = np.asarray(pil_image)

        # load corresponding mask
        name_mask = fn.stem + "_P.png"
        pil_mask = Image.open(masks_dir / name_mask)
        mask = np.asarray(pil_mask)

        # plot mask on top of an array and save the plot
        plot = draw_mask(image, mask, alpha, color)
        name_plot = fn.stem + f"_{suffix}.jpg"
        Image.fromarray(plot).save(out / name_plot)


if __name__ == "__main__":

    # HELPER FUNCTIONS FOR ARGPARSER
    # -------------------------------------------------------------------------
    def color_tuple(x):
        try:
            x = x[1:-1]  # strip braces
            b, g, r = map(int, x.split(","))  # extract BGR components
        except Exception:
            raise argparse.ArgumentTypeError("Color tuple must be in (B,G,R) format.")

        b_ok = 0 <= b <= 255
        g_ok = 0 <= g <= 255
        r_ok = 0 <= r <= 255

        if not (b_ok and g_ok and r_ok):
            raise argparse.ArgumentTypeError(
                "Each argument in color tuple must be in [0,255]."
            )

        return (b, g, r)

    # PARSE AND VERIFY ARGUMENTS
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Plot segmentation masks (CAMVID) on top of original images."
    )

    parser.add_argument("images_dir", type=Path, help="directory with image files")
    parser.add_argument(
        "masks_dir",
        type=Path,
        help="directory with corresponding masks in CAMVID format",
    )
    parser.add_argument("out", type=Path, help="output directory for plots")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="weight of original image array when blending (default: 0.4)",
    )
    parser.add_argument(
        "--color",
        type=color_tuple,
        default=(48, 130, 245),
        help="BGR color tuple for mask (default: (48, 130, 245))",
    )
    parser.add_argument(
        "--suffix", default="plot", help='suffix for plot filenames (default: "plot")'
    )

    args = parser.parse_args()

    plot_camvid(
        args.images_dir, args.masks_dir, args.out, args.alpha, args.color, args.suffix
    )
