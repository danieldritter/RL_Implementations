"""
File to handle dataset class and data processing. Resize images to match 224x336
before feeding in to network.
"""
import keras
import json
import requests

class CardGenerator():

    def __init__(self, image_path, batch_size, shuffle=True):
        self.image_path = image_path
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

def __main__():
    data = None
    with open('./data/AllCards.json', encoding="utf8") as w:
        data = json.load(w)
    for key in data.keys():
        for dict1 in data[key]['foreignData']:
            print(dict1)
            if 'multiverseId' not in dict1.keys():
                continue
            else:
                print(dict1['multiverseId'])


if __name__ == "__main__":
    __main__()
