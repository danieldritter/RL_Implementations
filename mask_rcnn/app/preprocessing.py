import numpy as np
import warnings


def sliding_window(X1, ws=(512, 512)):

    """
    Sliding window over the array

    If the end of the array is reach then shift back until not overlapping

    :param X1: Defines the input array for processing
    :param ws: Defines the output size
    :return:

    """

    shp = X1.shape
    ndim = X1.ndim

    cx = int(2 * shp[0] / ws[0])
    cy = int(2 * shp[1] / ws[1])

    sx = int(ws[0] / 2)
    sy = int(ws[1] / 2)

    X2 = []
    for ii in range(cx):
        for jj in range(cy):

            ix1 = ii*sx
            ix2 = ii*sx + ws[0]

            iy1 = jj*sy
            iy2 = jj*sy + ws[1]

            # check if moving past boundary in x
            if ix2 > shp[0]:
                ix2 = shp[0]
                ix1 = ix2 - ws[0]

            # check if moving past the boundary in y
            if iy2 > shp[1]:
                iy2 = shp[1]
                iy1 = iy2 - ws[1]

            if ndim == 3:
                X_tmp = X1[ix1:ix2, iy1:iy2, :]
            else:
                X_tmp = X1[ix1:ix2, iy1:iy2]

            X_tmp = np.expand_dims(X_tmp, axis=0)


            X2.append(X_tmp)

    return X2


def reconstruct_sliding_window(X2, shp=(5094, 5094)):

    """
    Sliding window over the array

    If the end of the array is reach then shift back until not overlapping

    :param array1:
    :param steps:
    :return:

    """

    # TODO: currently just overwriting the overlapping parts of the array. Combine instead.

    X1 = np.zeros(shp)

    ws = X2[0].shape

    cx = int(2 * shp[0] / ws[0])
    cy = int(2 * shp[1] / ws[1])

    sx = int(ws[0] / 2)
    sy = int(ws[1] / 2)

    cc = 0
    for ii in range(cx):
        for jj in range(cy):

            ix1 = ii*sx
            ix2 = ii*sx + ws[0]

            iy1 = jj*sy
            iy2 = jj*sy + ws[1]

            # check if moving past boundary in x
            if ix2 > shp[0]:
                ix2 = shp[0]
                ix1 = ix2 - ws[0]

            # check if moving past the boundary in y
            if iy2 > shp[1]:
                iy2 = shp[1]
                iy1 = iy2 - ws[1]
            # print("fit")
            # print(ix1)
            # print(ix2)
            # print(iy1)
            # print(iy2)
            # # print(X2[cc].shape)
            # print(X1.shape)

            try:
                X1[ix1:ix2, iy1:iy2] = X2[cc]

            except:
                warnings.warn('Issue recontructing image')

            # X_tmp = np.expand_dims(X_tmp, axis=0)

            # X2.append(X_tmp)

            cc += 1

    return X1
