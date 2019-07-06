

"""

Region evaluation

"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_opening, binary_closing, disk, remove_small_objects
from scipy.ndimage.morphology import binary_fill_holes
from skimage import filters, color, morphology


def background_masking(X, threshold):

    """

    Check if a region is background

    :param reg:
    :return: True if foreground or False if background

    """

    bw_image = color.rgb2gray(X)
    # threshold = 0.75
    mask = bw_image < threshold

    # thresh = 220
    #
    # mask = np.logical_and(X[:, :, 0] > thresh, np.logical_and(X[:, :, 1] > thresh, X[:, :, 2] > thresh))
    # mask = np.logical_not(mask)

    # plt.figure()
    # # plt.imshow(X)
    # plt.figure()
    # plt.imshow(mask)
    # plt.show()

    mask = remove_small_objects(np.logical_not(mask), min_size=10000)

    mask = remove_small_objects(np.logical_not(mask), min_size=10000)

    # plt.figure()
    # plt.imshow(mask)
    # plt.show()

    SE = disk(10)
    mask = binary_opening(mask, SE)
    SE = disk(10)
    mask = binary_closing(mask, SE)

    mask = binary_fill_holes(mask)
    mask = np.logical_not(mask)

    # plt.figure()
    # plt.imshow(X)
    # plt.figure()
    # plt.imshow(mask)
    # plt.show()

    return mask