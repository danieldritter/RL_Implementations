import numpy as np
import keras
from PIL import Image
import os
import xml.etree.ElementTree as ET

class MoNuSegDataGenerator(keras.utils.Sequence):


    def __init__(self,image_dir,labels_dir,batch_size=32,dim=(32,32,32),n_channels=1,n_classes=10,shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.label_dir = labels_dir
        self.image_dir = image_dir
        self.list_IDs = [os.path.splitext(file)[0] for file in os.listdir(image_dir)]
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self,list_IDs_temp):
        X = np.empty((self.batch_size,*self.dim,self.n_channels))
        y = np.empty((self.batch_size),dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = Image.open(self.image_dir+ID+".tif")
            y[i] = self.__process_xml(self.label_dir+ID+".xml")
        return X, y



    def __len__(self):
        return int(np.floor(len(self.list_IDs)/self.batch_size))

    def __getitem__(self,index):
        indexes = self.indexes[index:min(index+self.batch_size,self.__len__()-1)]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X,y = self.__data_generation(list_IDs_temp)
        return X,y


def __main__():
    generator = MoNuSegDataGenerator("/Users/danielritter/MaskRCNN/data/MoNuSeg Training Data/Tissue images/"
    ,"/Users/danielritter/MaskRCNN/data/MoNuSeg Training Data/Annotations/",dim=(1000,1000),n_channels=3)
    generator.__getitem__(5)

if __name__ == "__main__":
    __main__()
