"""
File to handle dataset class and data processing. Resize images to match 224x336
before feeding in to network.
"""
import keras

class CardGenerator():

    def __init__(self, image_path, batch_size, shuffle=True):
        self.image_path = image_path
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
