from random import sample


def sample_annotations(annotations, k):
    """Randomly sample k elements from annotations list."""
    return sample(annotations, k)


def mask_dropout(image, annotations, k):

    # randomly select annotations to keep
    selected_anns = sample_annotations(annotations, k)

    # create union using binary masks for each remaining annotation
    # use resulting binary mask to extract foreground
    # save foreground image to output image directory
    # save sampled anns
    pass