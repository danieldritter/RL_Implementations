"""

Benjamin Irving

"""

import xml.etree.ElementTree as ET
from skimage.draw import polygon
import numpy as np

import matplotlib.pyplot as plt


def con_xml(img1, file_xml):
    """

    @img1 : Image volume related to annotation
    @file_xml : xml file of the annotation

    """

    tree = ET.parse(file_xml)
    root = tree.getroot()
    data1 = root[0][1]

    imshape = img1.size
    roi_mask = np.zeros(imshape)
    # load image

    # Each data point
    x = []
    y = []

    # for each annotation point
    # pp[1] refers to the second array (the first is image values)

    print("Number of ROIs")
    print(len(data1)-1)

    for ii, roi1 in enumerate(data1):

        print("Roi: ", ii)
        if ii == 0:
            continue

        vertex = roi1[1].getchildren()

        x, y = [], []

        for vv in vertex:
            Xi = float(vv.get('X'))
            Yi = float(vv.get('Y'))

            x.append(Xi)
            y.append(imshape[1] - Yi)
            # y.append(imshape[1] - float(coord1[1]))

        y1 = np.array(y).astype(np.int16)
        x1 = np.array(x).astype(np.int16)
        rr, cc = polygon(x1, y1)

        # assign value of 1 to roi
        if len(rr) > 0:
            if np.max(rr) >= imshape[0] or np.max(cc) >= imshape[1]:
                print("Warning!!!!!!!!!!!!!!!!!!!!!!!")
                continue

        roi_mask[rr, cc] = 1

        # plt.figure()
        # plt.imshow(roi_mask)
        # plt.show()

    return roi_mask
