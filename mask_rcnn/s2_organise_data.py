"""

----------------------------------------------------------
-- Only needs to be performed once for the training set --
----------------------------------------------------------

NUCLEI DETECTION - Data organisation

Organising and saving the data

Pre-processing:
- Just normalise between 0 and 1
- Or provide more normalisation

"""


import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from matplotlib.colors import LinearSegmentedColormap
from skimage.exposure import rescale_intensity
from numpy.random import permutation
from app.preprocessing import sliding_window
from scipy.ndimage import zoom


def standardize_brightness(I, percentile=95):

    """
    Standardize brightness
    :param I:
    :return:
    """
    # assert is_uint8_image(I)
    I_LAB = color.rgb2lab(I)
    L = I_LAB[:, :, 0]
    p = np.percentile(L, percentile)
    I_LAB[:, :, 0] = np.clip(100 * L / p, 0, 100)
    I = color.lab2rgb(I_LAB)

    return I


def rescale_intensity_all(ihc_rgb, plot=True):

    ihc_hed = color.rgb2hed(ihc_rgb)

    if plot:

        cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white', 'navy'])
        cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white',
                                                                'saddlebrown'])
        cmap_eosin = LinearSegmentedColormap.from_list('mycmap', ['darkviolet',
                                                                  'white'])
        fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(ihc_rgb)
        ax[0].set_title("Original image")

        ax[1].imshow(ihc_hed[:, :, 0], cmap=cmap_hema)
        ax[1].set_title("Hematoxylin")

        ax[2].imshow(ihc_hed[:, :, 1], cmap=cmap_eosin)
        ax[2].set_title("Eosin")

        ax[3].imshow(ihc_hed[:, :, 2], cmap=cmap_dab)
        ax[3].set_title("DAB")

        for a in ax.ravel():
            a.axis('off')

        fig.tight_layout()

        plt.show()

    h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))
    e = rescale_intensity(ihc_hed[:, :, 1], out_range=(0, 1))
    d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1))
    ihc_hed = np.dstack((e, h, d))

    return ihc_hed



folder1 = './data/MoNuSeg_Training_Data/'

output_folder = folder1 + 'Numpy'

options1 = {}
options1['rescale'] = True
options1['plot'] = False


dir1 = os.listdir(output_folder)
dir1 = [a for a in dir1 if a.endswith('.npz')]

X = []
Xhe = []
y = []


ci = []   # Keep track of the index of each case
ci.append(0)
for count1, c1 in enumerate(dir1):
    print(count1)

    d1 = np.load(output_folder + '/' + c1)

    # Preprocessing:
    im = d1['im']
    im = im / 255
    roi = d1['roi']

    if options1['rescale']:
        # Currently rescaling to be just above 512x512
        im = zoom(im, [0.52, 0.52, 1])
        roi = zoom(roi, [0.52, 0.52, 1]) > 0.5

    if im.shape[-1] == 4:
        im = im[:, :, :3]

    im2 = rescale_intensity_all(im, options1['plot'])
    im2 = im2.astype(np.float32)

    if options1['plot']:

        plt.figure()
        plt.imshow(im)
        plt.contour(roi[:, :, 0])

        plt.figure()
        plt.imshow(im2)
        plt.contour(roi[:, :, 0])
        plt.show()

    xa = sliding_window(im)
    xa2 = sliding_window(im2)
    ya = sliding_window(roi)

    X.extend(xa)
    Xhe.extend(xa2)
    y.extend(ya)

    ci.append(ci[-1] + len(xa))

    # for pp in range(len(xa)):
    #
    #     plt.figure()
    #     plt.imshow(xa[pp])
    #     plt.contour(ya[pp][:, :, 0])
    #     plt.show()

X = np.concatenate(X, axis=0)
Xhe = np.concatenate(Xhe, axis=0)
y = np.concatenate(y, axis=0)

# Split into testing and training
X_train, Xhe_train, y_train = X[:ci[24]], Xhe[:ci[24]], y[:ci[24]]
X_test, Xhe_test, y_test = X[ci[24]:], Xhe[ci[24]:], y[ci[24]:]

# Randomise training
perm1 = permutation(np.arange(len(X_train)))
X_train, Xhe_train, y_train = X_train[perm1], Xhe_train[perm1], y_train[perm1]

# for pp in range(len(X_train)):
#
#     plt.figure()
#     plt.imshow(Xhe_train[pp])
#     plt.contour(y_train[pp][:, :, 0])
#     plt.show()

if not options1['rescale']:
    np.savez(folder1 + 'data_train.npz', X=X_train, y=y_train, Xhe=Xhe_train)
    np.savez(folder1 + 'data_test.npz', X=X_test, y=y_test, Xhe=Xhe_test)
else:
    np.savez(folder1 + 'data_train_rescale.npz', X=X_train, y=y_train, Xhe=Xhe_train)
    np.savez(folder1 + 'data_test_rescale.npz', X=X_test, y=y_test, Xhe=Xhe_test)
