import keras
from keras import layers
import keras.backend as K
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import data_processing



def make_generator(learning_rate):
    """
    Final Expected Size is 224x336
    """
    model = keras.Sequential()
    model.add(layers.Dense(14*14*1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((14,14,1024)))
    model.add(layers.Conv2DTranspose(filters=512, kernel_size=5, strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(filters=256, kernel_size=5, strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(filters=128, kernel_size=5, strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(filters=64, kernel_size=5, strides=(2,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(filters=32, kernel_size=5, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(filters=3, kernel_size=5, padding='same', activation='tanh'))
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=learning_rate))
    return model

def make_discriminator(learning_rate):
    model = keras.Sequential()
    model.add(layers.Conv2D(filters=256, kernel_size=5, strides=(2, 2), input_shape=[224, 336, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(filters=128, kernel_size=5, strides=(2, 2)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(filters=64, kernel_size=5, strides=(2, 2)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=learning_rate))
    return model




def __main__():
    parser = argparse.ArgumentParser(description="A Program to automatically")
    parser.add_argument("--batch_size", type=int, help="Batch Size to use when training", default=64)
    parser.add_argument("--learning_rate", type=float, help="Learning Rate to use when training, float between [0,1)", default=.001)

    args = parser.parse_args()
    # Instantiate Generator and Discriminator
    generator = make_generator(args.learning_rate)
    discriminator = make_discriminator(args.learning_rate)
    sample_generator = data_processing.CardGenerator(image_path, args.batch_size, shuffle=True)

    # Use train on batch to train discriminator and generator indepedently
    input_noise = K.random.normal([args.batch_size, 100])

    fake_batch = generator(input_noise)
    real_batch = sample_generator.get_sample()

    batch = 


if __name__ == "__main__":
    __main__()
