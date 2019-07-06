
"""

Script for converting ndpi files to pyvips files

"""

import pyvips
import os
import shutil


def convert_to_dzi(folder1, folder2, name1, reg_size=1024):
    """

    Convert a file to dzi
    :param folder1: Input folder
    :param folder2: Output folder
    :param name1: File name

    :return:

    """
    save_path = folder2 + name1[:-4] + 'dzi'
    image = pyvips.Image.new_from_file(folder1 + name1)

    if os.path.exists(save_path):
        print('Removing existing file')
        shutil.rmtree(save_path[:-4] + '_files')
        os.remove(save_path)

    if save_path.endswith('dzi'):
        image.dzsave(save_path, tile_size=reg_size)
    else:
        print('Savepath is not dzi... skipping')


def remove_a_dzi_file(folder1, name1):

    """

    Delete a file
    :param folder1:
    :param name1:
    :return:

    """
    if not name1.endswith('.dzi'):
        print('Not a dzi file')
        return 0

    if os.path.exists(folder1 + name1):
        print('Removing existing file')
        shutil.rmtree(folder1 + name1[:-4] + '_files')
        os.remove(folder1 + name1)

    return 1


if __name__ == "__main__":

    folder1 = '/home/benirving/Data_DP/CALM/'
    folder2 = '/home/benirving/Data_DP/CALM_deepzoom/'
    name1 = 'HandE CALM-E-010 - 2018-06-18 11.20.53.ndpi'

    convert_to_dzi(folder1, folder2, name1)

