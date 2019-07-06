import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import xml.etree.ElementTree as ET

class MoNuSegDataset(Dataset):

    def __init__(self, image_dir, annotations_dir, transform=None):
        self.image_dir = image_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.name_map = {}
        name_list = os.listdir(image_dir)
        for i, name in enumerate(name_list):
            stripped_name = os.path.splitext(name)[0]
            self.name_map[i] = image_dir+"/"+stripped_name

    def __len__(self):
        return len(os.listdir(image_dir))

    def __getitem__(self,index):
        img_name = self.name_map[index]
        example = Image.open(img_name+".tif")
        annotation = self.process_xml(img_name+".xml")

        if self.transform:
            example = self.transform(example)
        return example

    def process_xml(self,file_name):
        """
        Takes in xml annotation files and converts them into a binary mask
        """
        tree = ET.parse(file_name)
        root = tree.getroot()
        regions = root.findall('Region')
        print(regions)


def __main__():
    dataset = MoNuSegDataset(os.getcwd()+"/MoNuSeg Training Data/Tissue images",os.getcwd()+"/MoNuSeg Training Data/Annotations")
    dataset.__getitem__(5)

if __name__ == "__main__":
    __main__()
