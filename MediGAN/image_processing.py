"""
Functions for processing images to feed in Medical Image Generative Adversarial
Networks
"""
import torch.utils.data as data
import os
import skimage.io as io
from PIL import Image
import numpy as np


class NucleiDataset(data.Dataset):
    """
    Dataset for MoNuSeg Nuclear Segmentation Dataset
    """
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.index_map = dict([(i, os.path.splitext(file)[0]) for i, file in enumerate(os.listdir(image_dir))])

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, index):
        file_name = self.index_map[index]
        image = Image.open(self.image_dir + "/" + file_name + ".tif")
        label = Image.open(self.label_dir + "/" + file_name + ".png")
        sample = {'image': image, 'label': label}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['label'] = self.transform(sample['label'])
        return sample

class HistoryBuffer():
    """
    This is a class to store previously generated fake images and then
    sample from the pool once it is full.
    """

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def insert_image(self, image):
        # Remove last element if buffer is full, otherwise just add the image
        if len(self.buffer) > self.buffer_size:
            self.buffer.insert(0,image)
            self.buffer.pop()
        else:
            self.buffer.insert(0, image)

    def get_image_samples(self, sample_size):
        if sample_size > self.buffer_size:
            print("Error: Cannot sample a batch greater than size of buffer")
            return None
        else:
            return np.random.choice(self.buffer, sample_size)
    
