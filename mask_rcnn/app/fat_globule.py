
"""

Detect fat globules in the images

"""
from skimage import filters, color, morphology
import scipy.ndimage as ndi
from skimage.measure import regionprops, label
import numpy as np
import matplotlib.pyplot as plt


def detect_fat(image, imb, threshold=0.85):

    """

    Very simple method to detect fat

    :param im1: rgb histology image
    :param imb: binary image representing the background (1 is background)
    :return:

    """

    fat_area = 0
    bw_image = color.rgb2gray(image)
    # threshold = 0.75
    binary = bw_image > threshold
    # binary = np.logical_and(binary, np.logical_not(imb))
    #
    # plt.figure()
    # plt.imshow(imb)
    # #
    # plt.figure()
    # plt.imshow(binary)
    # plt.show()

    binary = np.logical_and(binary, np.logical_not(imb))

    cleaned = morphology.remove_small_objects(binary, min_size=350)

    holes_filled = ndi.binary_fill_holes(cleaned)

    # plt.figure()
    # plt.imshow(holes_filled)
    # plt.show()

    # SE = morphology.disk(3)
    # holes_filled = morphology.binary_closing(holes_filled, SE)

    blobs = label(holes_filled, connectivity=1)
    for blob in regionprops(blobs):
        if blob.eccentricity > 0.9 or blob.area > 16000:
            blobs[blobs == blob.label] = 0
        else:
            fat_area = fat_area + blob.area

    core_area = np.sum(np.logical_not(imb))

    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(binary)
    # plt.subplot(222)
    # plt.imshow(image)
    # plt.subplot(223)
    # plt.imshow(holes_filled)
    # plt.subplot(224)
    # plt.imshow(blobs)
    # plt.show()

    return blobs, fat_area, core_area