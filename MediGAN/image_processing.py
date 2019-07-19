"""
Functions for processing images to feed in Medical Image Generative Adversarial
Networks
"""
import torch.utils as utils
import os
import skimage.io as io


class NucleiDataset(utils.data.Dataset):
    """
    Dataset for MoNuSeg Nuclear Segmentation Dataset
    """
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.index_map = dict([(i, os.path.splitext(file)[0]) for i, file in enumerate(os.listdir(image_dir))])

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, index):
        file_name = self.index_map[index]
        image = io.imread(self.image_dir + "/" + file_name + ".tif")
        label = io.imread(self.label_dir + "/" + file_name + ".png")

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
