import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

class MaskRCNN(nn.Module):

    def __init__(self):
        """
        A short description.
        :Params:
            variable (type): description
        :Output:
            type: description
        Raises:
            Exception: description
        """

        super(MaskRCNN, self).__init__()
        self.FPN = FPN()
        self.conv1 = nn.Conv2d(4, 16, 2)
        self.conv2 = nn.Conv2d(16, 32, 2)
        self.conv3 = nn.Conv2d(32, 64, 2)
        self.reg_layer = nn.Linear(77,8)
        self.cls_layer = nn.Linear(77,4)
        self.masking_layer = nn.Conv2d(64, 256, 2)

    def forward(self,input):
        feature_maps = self.FPN(input)
        out = F.relu(self.conv1(feature_maps))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        regression_out = F.relu(self.reg_layer(out))
        classification_out = F.relu(self.clas_layer(out))
        mask = F.relu(self.masking_layer(out))
        return [regression_out,classification_out,mask]


class FPN(nn.Module):
    """
    Class to implement a feature pyramid network for generating region
    proposals based on an image. Region proposals are then passed on to
    MaskRCNN.
    """

    def __init__(self):
        super(FPN, self).__init__()
        # Bottom up convolution layers
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        # Top down upsampling layerss
        self.up1 = nn.ConvTranspose2d(128, 64, 3)
        self.up2 = nn.ConvTranspose2d(64, 32, 3)
        self.up3 = nn.ConvTranspose2d(32, 16, 3)
        self.up4 = nn.ConvTranspose2d(16, 1 , 3)
        # TODO: Check filter dimensions to see if they should stay the same
        self.skipconv1 = nn.Conv2d(16, 16, 1)
        self.skipconv2 = nn.Conv2d(32, 32, 1)
        self.skipconv3 = nn.Conv2d(64, 64, 1)

    def bottom_up_pathway(self, input):
        out1 = F.relu(self.conv1(input))
        skip1 = copy.copy(out1)
        out2 = F.relu(self.conv2(out1))
        skip2 = copy.copy(out2)
        out3 = F.relu(self.conv3(out2))
        skip3 = copy.copy(out3)
        out4 = F.relu(self.conv4(out3))
        return [skip1, skip2, skip3, out4]

    def top_down_pathway(self, feature_maps):
        skip1, skip2, skip3, out = feature_maps
        out = F.relu(self.up1(out))
        pred1 = copy.copy(out)
        out = out + F.relu(self.skipconv3(skip3))
        out = F.relu(self.up2(out))
        pred2 = copy.copy(out)
        out = out + F.relu(self.skipconv2(skip2))
        out = F.relu(self.up3(out))
        pred3 = copy.copy(out)
        out = out + F.relu(self.skipconv1(skip1))
        return [pred1, pred2, pred3, out]

    def forward(self, input):
        feature_maps = self.bottom_up_pathway(input)
        return self.top_down_pathway(feature_maps)


def __main__():
    pass


if __name__ == "__main__":
    __main__()
