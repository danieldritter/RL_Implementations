

"""

Run on a single case

"""


from app.model.model import get_simple_unet
import time
import numpy as np
# from keras.preprocessing.image import ImageDataGenerator, ImageEnhance
import matplotlib.pyplot as plt

from skimage.measure import label, regionprops
from skimage.io import imread, imsave
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from skimage.segmentation import mark_boundaries

from scipy import ndimage

from app.preprocessing import sliding_window, reconstruct_sliding_window
from app.region_evaluation import background_masking
from scipy.ndimage import zoom
from app.fat_globule import detect_fat
from skimage import filters, color, morphology
from skimage.color import label2rgb
from skimage.exposure import is_low_contrast



def process_region(options, modelweights='nuclei_100.h5'):
    """
    options defines a dictionary of the naming structure
    :param options:
    :return:
    """

    metrics = {}

    model = get_simple_unet()

    model.load_weights('saved_models/' + modelweights)

    im1 = imread(options['input_folder'] + options['level'] + '/' + options['name'])

    if is_low_contrast(im1):
        return -1, -1

    # Zoom image
    # im1 = zoom(im1, [2, 2, 1])

    # Identify background
    imb = background_masking(im1)

    # Detect fat regions
    imf, fperc, coreperc = detect_fat(im1, imb)
    imfrgb = label2rgb(imf)

    # im1 = im1[:, :, :]

    if options['plot']:
        plt.figure()
        plt.imshow(im1)

    X_test = im1 / 255
    X_test = X_test.astype(np.float32)

    # X_test_patch = extract_patches_2d(X_test, (512, 512))

    X_test = sliding_window(X_test)
    X_test = np.concatenate(X_test, axis=0)
    imb2 = sliding_window(imb)
    imb2 = np.concatenate(imb2, axis=0)

    # X_test = np.expand_dims(X_test, axis=0)
    # X_test = X_test[:, 3000:3512, 3000:3512, :]

    pred_vec = []

    for ii in range(len(X_test)):

        Xi = X_test[ii]

        if np.sum(imb2[ii] == 0) < 200:
            y_predi = np.zeros(Xi.shape[:2])

        else:
            Xi = np.expand_dims(Xi, axis=0)

            # Model prediction
            y_pred = model.predict(Xi)

            y_predi = y_pred[0][:, :, 0] > 0.5

            ylab = label(y_predi)
            # for region in regionprops(ylab):
            #     # take regions with large enough areas
            #     if region.area < 110:
            #         y_predi[ylab == region.label] = 0

        pred_vec.append(y_predi)

    shp = im1.shape[:2]
    Y = reconstruct_sliding_window(pred_vec, shp=shp)

    Ylab = label(Y)
    rp = regionprops(Ylab)

    xx, yy = [], []
    for r1 in rp:
        xx.append(r1.centroid[0])
        yy.append(r1.centroid[1])

    xx = np.asarray(xx, dtype=int)
    yy = np.asarray(yy, dtype=int)

    Ycentroid = np.zeros_like(Y)
    Ycentroid[xx, yy] = 1
    heat_map = ndimage.filters.gaussian_filter(Ycentroid, sigma=40)

    print('Max heat: ', np.max(heat_map))

    th = 0.001
    thm = 0.004
    heat_map[heat_map < th] = 0
    heat_map = heat_map - th
    heat_map = heat_map / (thm - th)

    im2 = np.copy(im1)

    # NUCLEI
    Y = Y > 0.5
    Yp = np.zeros_like(Y)
    Ys = np.stack((Yp, Y, Yp), axis=-1)
    im2[Ys] = 255
    Ys = np.stack((Y, Yp, Y), axis=-1)
    im2[Ys] = 0

    im2b = np.copy(im2)
    # FAT
    Y = imf > 0.5
    Yp = np.zeros_like(Y)
    Ys = np.stack((Y, Yp, Yp), axis=-1)
    im2b[Ys] = 255
    Ys = np.stack((Yp, Y, Y), axis=-1)
    im2b[Ys] = 0

    # Save an image with detected nuclei
    # imsave(options['out'] + 'test.jpeg', im1)
    imsave(options['data_folder'] + options['level'] + '/' + options['name'], im2b)

    im3 = np.copy(im2)
    im3 = im3 * (1/3)
    im3[:, :, 0] = im3[:, :, 0] + heat_map*(255*(2/3))
    im3 = im3.astype('int')

    # Save an image with detected nuclei
    imsave(options['data_folder2'] + options['level'] + '/' + options['name'], im3)
    # imsave(options['out'] + 'test3.jpeg', np.stack([imb, imb, imb], axis=-1)*255)

    # imsave(options['out'] + 'imfat.jpeg', imfrgb)

    if np.sum(np.logical_not(imb) > 0) / np.sum(np.logical_not(imb) > -1) > 0.7:
        fat_perc = np.sum(imf > 0) / np.sum(np.logical_not(imb))
    else:
        fat_perc = -1

    nuclei_count = Ylab.max()
    print("Nuclei count: ", nuclei_count)
    print("Fat perc: ", fat_perc)

    return fat_perc, nuclei_count


def fat_simple(name1, folder1, fthreshold=0.85):
    """
    options defines a dictionary of the naming structure
    :param options:
    :return:
    """

    metrics = {}
    #
    # model = get_simple_unet()
    #
    # model.load_weights('saved_models/' + modelweights)

    im1 = imread(folder1 + name1)

    if is_low_contrast(im1):
        print('Low contrast')
        return -1, -1

    # Zoom image
    # im1 = zoom(im1, [2, 2, 1])

    # Identify background
    imb = background_masking(im1, fthreshold)

    # Detect fat regions
    imf, fperc, coreperc = detect_fat(im1, imb, threshold=fthreshold)
    imfrgb = label2rgb(imf)

    # im1 = im1[:, :, :]

    if np.sum(np.logical_not(imb) > 0) / np.sum(np.logical_not(imb) > -1) > 0.3:
        fat_perc = np.sum(imf > 0) / np.sum(np.logical_not(imb))
    else:
        fat_perc = -1

    return im1, imf, fat_perc


def inflammation_simple(name1, folder1, modelweights='nuclei_rescale_200.h5', th=0.0015, thm=0.004):

    """
    options defines a dictionary of the naming structure
    :param options:
    :return:
    """

    # th = 0.001
    # thm = 0.004

    metrics = {}

    model = get_simple_unet()

    model.load_weights('saved_models/' + modelweights)

    im1 = imread(folder1 + name1)

    if is_low_contrast(im1):
        return -1, -1

    # Identify background
    imb = background_masking(im1)

    X_test = im1 / 255
    X_test = X_test.astype(np.float32)

    # X_test_patch = extract_patches_2d(X_test, (512, 512))

    X_test = sliding_window(X_test)
    X_test = np.concatenate(X_test, axis=0)
    imb2 = sliding_window(imb)
    imb2 = np.concatenate(imb2, axis=0)

    # X_test = np.expand_dims(X_test, axis=0)
    # X_test = X_test[:, 3000:3512, 3000:3512, :]

    pred_vec = []

    for ii in range(len(X_test)):

        Xi = X_test[ii]

        if np.sum(imb2[ii] == 0) < 200:
            y_predi = np.zeros(Xi.shape[:2])

        else:
            Xi = np.expand_dims(Xi, axis=0)

            # Model prediction
            y_pred = model.predict(Xi)

            y_predi = y_pred[0][:, :, 0] > 0.5

            ylab = label(y_predi)
            # for region in regionprops(ylab):
            #     # take regions with large enough areas
            #     if region.area < 110:
            #         y_predi[ylab == region.label] = 0

        pred_vec.append(y_predi)

    shp = im1.shape[:2]
    Y = reconstruct_sliding_window(pred_vec, shp=shp)

    Ylab = label(Y)
    rp = regionprops(Ylab)

    xx, yy = [], []
    for r1 in rp:
        xx.append(r1.centroid[0])
        yy.append(r1.centroid[1])

    xx = np.asarray(xx, dtype=int)
    yy = np.asarray(yy, dtype=int)

    Ycentroid = np.zeros_like(Y)
    Ycentroid[xx, yy] = 1
    heat_map = ndimage.filters.gaussian_filter(Ycentroid, sigma=30)

    print('Max heat: ', np.max(heat_map))

    heat_map[heat_map < th] = 0
    heat_map = heat_map - th
    heat_map = heat_map / (thm - th)

    im2 = np.copy(im1)

    # NUCLEI
    Y = Y > 0.5
    Yp = np.zeros_like(Y)
    Ys = np.stack((Yp, Y, Yp), axis=-1)
    im2[Ys] = 255
    Ys = np.stack((Y, Yp, Y), axis=-1)
    im2[Ys] = 0

    return im1, heat_map, Y