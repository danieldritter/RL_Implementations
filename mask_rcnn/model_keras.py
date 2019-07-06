"""
A short description.

:Params:
    variable (type): description
:Output:
    type: description
Raises:
    Exception: description
"""


import copy
from keras.layers import Conv2D, Dense, Input, Conv2DTranspose
from keras.models import Model
import keras.backend as KB
import numpy as np

class MaskRCNN():

    """
    A short description.
    :Params:
        variable (type): description
    :Output:
        type: description
    Raises:
        Exception: description
    """


    def __init__(self):
        super(MaskRCNN, self).__init__()
        # Bottom up convolution layers
        self.conv1 = Conv2D(16, (3, 3), activation="relu")
        self.conv2 = Conv2D(32, (3, 3), activation="relu")
        self.conv3 = Conv2D(64, (3, 3), activation="relu")
        self.conv4 = Conv2D(128, (3, 3), activation="relu")
        # Top down upsampling layerss
        self.up1 = Conv2DTranspose(64, (3, 3), activation="relu")
        self.up2 = Conv2DTranspose(32, (3, 3), activation="relu")
        self.up3 = Conv2DTranspose(16, (3, 3), activation="relu")
        self.up4 = Conv2DTranspose(1, (3, 3), activation="relu")
        # TODO: Check filter dimensions to see if they should stay the same
        self.skipconv1 = Conv2D(16, (1, 1), activation="relu")
        self.skipconv2 = Conv2D(32, (1, 1), activation="relu")
        self.skipconv3 = Conv2D(64, (1, 1), activation="relu")


        self.conv5 = Conv2D(16, (2, 2), activation="relu")
        self.conv6 = Conv2D(32, (2, 2), activation="relu")
        self.conv7 = Conv2D(64, (2, 2), activation="relu")
        self.reg_layer = Dense(8, activation="relu")
        self.cls_layer = Dense(4, activation="relu")
        self.masking_layer = Conv2D(256, (2, 2), activation="relu")

    def forward(self, input):
        # Bottom Down FPN
        out = self.conv1(input)
        skip1 = copy.copy(out)
        out = self.conv2(out)
        skip2 = copy.copy(out)
        out = self.conv3(out)
        skip3 = copy.copy(out)
        out = self.conv4(out)
        print(out.shape)
        # Top Up Pathway
        out = self.up1(out)
        pred1 = copy.copy(out)
        out = out + self.skipconv3(skip3)
        out = self.up2(out)
        pred2 = copy.copy(out)
        out = out + self.skipconv2(skip2)
        out = self.up3(out)
        pred3 = copy.copy(out)
        out = out + self.skipconv1(skip1)
        # MaskRCNN Network
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        regression_out = self.reg_layer(out)
        classification_out = self.cls_layer(out)
        mask = self.masking_layer(out)
        return [regression_out, classification_out, mask]
