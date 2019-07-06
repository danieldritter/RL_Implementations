
"""

NUCLEI DETECTION - run training

"""

from app.model.model import get_simple_unet
import time
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.models import Model
# from keras.preprocessing.image import ImageEnhance
import matplotlib.pyplot as plt
import model_keras

def prediction_comparison(model, X_test, y_test):
    """
    Prediction comparison
    :param model:
    :param X_test:
    :param y_test:
    :return:
    """

    def DSC(gt, pred):

        pred = pred > 0.5

        TP = np.sum(np.logical_and(gt, pred))
        DSC1 = 2 * TP / (np.sum(gt) + np.sum(pred))

        return DSC1

    DSC_vec = []
    for ii in range(len(X_test)):

        Xi = X_test[ii]
        Xi = np.expand_dims(Xi, axis=0)
        y_pred = model.predict(Xi)

        DSC_vec.append(DSC(y_test[ii], y_pred[0]))

    return np.mean(DSC_vec)


def train():

    """
    Training model
    :return:

    """

    start_time = time.time()

    # Epochs
    for e in range(options['epochs']):
        print("Epoch: ", e)

        # Augmentation

        # TODO: test ZCA whitening

        # we create two instances with the same arguments
        data_gen_args = dict(rotation_range=10.,
                             width_shift_range=0.05,
                             height_shift_range=0.05,
                             vertical_flip=True,
                             horizontal_flip=True,
                             zoom_range=0.1,
                             fill_mode='constant',
                             cval=0)

        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)

        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = np.random.randint(0, 100)
        image_datagen.fit(X, augment=True, seed=seed)
        mask_datagen.fit(y, augment=True, seed=seed)

        image_generator = image_datagen.flow(X, batch_size=options['batch_size'], seed=seed)
        mask_generator = mask_datagen.flow(y, batch_size=options['batch_size'], seed=seed)

        train_generator = zip(image_generator, mask_generator)

        batches = 0
        loss = 0
        for X_batch, Y_batch in train_generator:

            # plt.figure()
            # plt.imshow(X_batch[0])
            # plt.contour(Y_batch[0][:, :, 0])
            # plt.show()

            l1 = model.fit(X_batch, Y_batch,
                           batch_size=options['batch_size'],
                           verbose=0)
            batches += 1
            loss += l1.history['loss'][0]

            if batches >= np.ceil(len(X) / options['batch_size']):
                break

        loss = loss / batches

        print(loss)

        # Prediction
        test_dsc = prediction_comparison(model, X_test, y_test)
        print("Test DSC: ", test_dsc)

        if e % 10 == 0:
            if not options['rescale_half']:
                model.save_weights('saved_models/nuclei_' + str(e) + '.h5')
            else:
                model.save_weights('saved_models/nuclei_rescale_' + str(e) + '.h5')


if __name__ == "__main__":

    input_tensor = Input(shape=(1000, 1000, 3))
    network = model_keras.MaskRCNN()
    out = network.forward(input_tensor)
    model = Model(inputs=input_tensor, outputs=out)
    model.compile("Adam", loss="mean_squared_error")

    options = {}

    options['epochs'] = 400
    options['data_folder'] = './data/MoNuSeg_Training_Data/'
    options['batch_size'] = 2
    options['rescale_half'] = True

    if not options['rescale_half']:
        data = np.load(options['data_folder'] + 'data_train_rescale.npz')
    else:
        data = np.load(options['data_folder'] + 'data_train_rescale.npz')

    X = data['X']
    y = data['y']
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    if not options['rescale_half']:
        data = np.load(options['data_folder'] + 'data_test.npz')
    else:
        data = np.load(options['data_folder'] + 'data_test_rescale.npz')

    X_test = data['X']
    y_test = data['y']
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    train()
