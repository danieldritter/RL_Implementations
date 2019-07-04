"""
Convertor from Osirix xml to roi


Advantages over previous matlab script:

- Does not depend on deprecated xml conversion pcode
- Search more explicit
- Does not rely on slices being in order
- Allows multiple annotations on a single slice
- Does not require a start slice

Benjamin Irving
2014/03/31

"""
from __future__ import print_function, division

import xml.etree.ElementTree as ET
from skimage.draw import polygon
import nibabel as nib
import numpy as np


def con_xml(img1, file_xml):
    """

    @img1 : Image volume related to annotation
    @file_xml : xml file of the annotation

    """

    tree = ET.parse(file_xml)
    root = tree.getroot()
    data1 = root[0][1]

    imshape = img1.shape[:3]
    roi1 = np.zeros(imshape)
    # load image

    # for each slice
    for ii in data1:

        #Slice
        slice1 = int(ii.find('integer').text) - 1
        print("Slice ", slice1)

        pp = ii.findall('array')

        # Each data point
        x = []
        y = []

        # for each annotation point
        # pp[1] refers to the second array (the first is image values)
        for jj in pp[1]:
            coord_text = jj.text
            coord1 = coord_text.strip('{},').split(',')
            x.append(float(coord1[0]))
            y.append(imshape[1] - float(coord1[1]))

        y1 = np.array(y)
        x1 = np.array(x)
        rr, cc = polygon(x1, y1)

        # assign value of 1 to roi
        roi1[rr, cc, slice1-1] = 1

    return roi1
