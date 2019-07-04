
"""

----------------------------------------------------------
-- Only needs to be performed once for the training set --
----------------------------------------------------------

NUCLEI DETECTION - Data Preprocessing

Convert slides and masks to numpy arrays

"""
import os
import glob
from app.xmlroi.xml_to_vol_2d import con_xml
import nibabel as nib
import numpy as np

from PIL import Image, TiffImagePlugin
TiffImagePlugin.READ_LIBTIFF = True

import matplotlib.pyplot as plt
from openslide import OpenSlide


def save_roi(volume1, refimg, folder1, name1):

    h1 = refimg.get_header()
    af1 = refimg.get_affine()
    roi_out = nib.Nifti1Image(volume1, af1, header=h1)
    roi_out.update_header()
    f_out = folder1 + name1 + '.nii'
    roi_out.to_filename(f_out)


folder1 = './data/MoNuSeg_Training_Data/'
# folder1 = '/opt/data/'

print('files')

annotation_folder = folder1 + 'Annotations'
images_folder = folder1 + 'Tissue-images'
output_folder = folder1 + 'Numpy'

dir1 = os.listdir(annotation_folder)
dir1 = [a for a in dir1 if a.endswith('.xml')]


for pat1 in dir1:

    print(pat1)

    # find tumour file and name with wildcard
    file1_xml = annotation_folder + '/' + pat1
    # find lesion file and name with wildcard
    file2_im = images_folder + '/' + pat1[:-4] + '.tif'
    if not os.path.exists(file2_im):
        print("Warning the file doesn't exist")
        continue

    print(file2_im)
    im = Image.open(file2_im)
    # Try using either pillow
    if file2_im.endswith("TS1.tif"):
        img2 = np.array(im)
    else:
        slide1 = OpenSlide(file2_im)
        slide1d = slide1.read_region(location=(0, 0), level=0, size=slide1.dimensions)
        img2 = np.array(slide1d)

    # Convert
    roi_im = con_xml(im, file1_xml)
    roi_im = roi_im.astype(np.int)
    roi_im = np.fliplr(roi_im)
    roi_im = roi_im.transpose()
    roi_im = roi_im[:, :, np.newaxis]

    name1 = pat1[:-4] + '.npz'
    np.savez(output_folder + '/' + name1, im=img2, roi=roi_im)

    # plt.figure()
    # plt.imshow(img2)
    # plt.contour(roi_im[:, :, 0])
    # plt.show()


#
