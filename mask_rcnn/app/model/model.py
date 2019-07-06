

"""

Simple lightweight U-Net

"""

import numpy as np
import tensorflow as tf
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D
from keras.layers import UpSampling2D, Cropping2D
from keras.models import Model
from keras.optimizers import Adam

# TODO: Ignoring batch normalisation for the meanwhile


def get_simple_unet(input_size=(512, 512)):

    """


    :param input_size:
    :return:

    """

    # l = [16, 32, 64, 128, 64, 32, 16]
    l = [32, 64, 128, 256, 128, 64, 32]


    inputs = Input((input_size[0], input_size[1], 3))

    c1 = Conv2D(l[0], (3, 3), activation='relu', padding='same')(inputs)   # 68x68 (virtual viewing window of unet)
    c1 = Conv2D(l[0], (3, 3), activation='relu', padding='same')(c1)       # 66x66
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)                              # 64x64

    c2 = Conv2D(l[1], (3, 3), activation='relu', padding='same')(p1)       # 32x32
    c2 = Conv2D(l[1], (3, 3), activation='relu', padding='same')(c2)       # 30x30
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)                              # 28x28

    c3 = Conv2D(l[2], (3, 3), activation='relu', padding='same')(p2)       # 14x14
    c3 = Conv2D(l[2], (3, 3), activation='relu', padding='same')(c3)       # 12x12
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)                              # 10x10

    # -----
    c4 = Conv2D(l[3], (3, 3), activation='relu', padding='same')(p3)       # 5x5
    c4 = Conv2D(l[3], (3, 3), activation='relu', padding='same')(c4)       # 3x3
    # ----

    up5 = UpSampling2D(size=(2, 2))(c4)
    m5 = concatenate([c3, up5], axis=-1)
    c5 = Conv2D(l[4], (3, 3), activation='relu', padding='same')(m5)
    c5 = Conv2D(l[4], (3, 3), activation='relu', padding='same')(c5)

    up6 = UpSampling2D(size=(2, 2))(c5)
    m6 = concatenate([c2, up6], axis=-1)
    c6 = Conv2D(l[5], (3, 3), activation='relu', padding='same')(m6)
    c6 = Conv2D(l[5], (3, 3), activation='relu', padding='same')(c6)

    up7 = UpSampling2D(size=(2, 2))(c6)
    m7 = concatenate([c1, up7], axis=-1)
    c7 = Conv2D(l[6], (3, 3), activation='relu', padding='same')(m7)
    c7 = Conv2D(l[6], (3, 3), activation='relu', padding='same')(c7)

    c8 = Conv2D(1, (1, 1), padding='same')(c7)

    model = Model(inputs=inputs, outputs=c8)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy')

    return model

