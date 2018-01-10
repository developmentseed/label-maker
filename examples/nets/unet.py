"""
unet.py

@author: developmentseed

Example using UNet deep learning model for pixel-wise image segmentation of
images obtained from this repo.
"""

import os
import os.path as op
from datetime import datetime as dt

import numpy as np
import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (Input, Conv2D, MaxPooling2D, concatenate,
                          UpSampling2D)


def load_data(data_dir, zero_mean=True, unit_var=True):
    """Helper to load data and print basic information"""
    # Load data, preshuffled and split between train and test sets
    npz = np.load(op.join(data_dir, 'data.npz'))
    x_train, y_train = npz['x_train'], npz['y_train']
    x_test, y_test = npz['x_test'], npz['y_test']

    # If needed
    if zero_mean:
        mean = np.mean(x_train, axis=(0, 1))
        x_train -= mean
        y_train -= mean
    if unit_var:
        std = np.std(x_train, axis=(0, 1))
        x_train /= std
        y_train /= std

    # Set mask images as 0-1 (to match model's sigmoid output)
    x_test /= 255
    y_test /= 255

    print('Loaded {} training examples and {} testing examples'.format(
        x_train.shape[0], x_test.shape[0]))

    return x_train, y_train, x_test, y_test


def get_f1_score(y_true, y_pred, smooth=1.):
    """Non-binary F1 Score (aka Sorenson-Dice) coeff
    Notes
    -----
    Definition here:
        https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    This mirrors what others in the ML community have been using where, in the
    denominator, the pred and true vectors are summed (not squared prior to summing)
    """

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    total = K.sum(y_true_f + y_pred_f)

    return (2. * intersection + smooth) / (total + smooth)


def get_f1_dist(y_true, y_pred):
    """Helper to turn the F1 score into a loss"""
    return -1 * get_f1_score(y_true, y_pred)


def get_unet(input_shape, kernel_size=3, strides=1, pool_size=2):
    """Return vanilla u-net model ready for training.

    Parameters
    ----------
    input_shape: iterable
        Should be tuple, list, or array specifying (img_width,
        img_height, n_channels). Keras will prepend batch dimension
    kernel_size: int or tuple
        Size of kernel for 2D convolutions. Should be 2 or 3 (default).
    strides: int or tuple
        Size of stride for 2D convolutions. Default is 1.
    pool_size: tuple
        Size of kernal for max-pooling and upsampling. Default is 2.

    Returns
    -------
    unet_model: Keras.models.Model
        Compiled model ready for training.
    """
    conv_params = dict(activation='relu', kernel_size=kernel_size,
                       strides=strides, padding='same',
                       kernel_initializer='he_uniform')

    inputs = Input(input_shape)

    conv1 = Conv2D(32, **conv_params)(inputs)
    conv1 = Conv2D(32, **conv_params)(inputs)
    pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

    conv2 = Conv2D(64, **conv_params)(pool1)
    conv2 = Conv2D(64, **conv_params)(conv2)
    pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

    conv3 = Conv2D(128, **conv_params)(pool2)
    conv3 = Conv2D(128, **conv_params)(conv3)
    pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

    conv4 = Conv2D(256, **conv_params)(pool3)
    conv4 = Conv2D(256, **conv_params)(conv4)
    pool4 = MaxPooling2D(pool_size=pool_size)(conv4)

    conv5 = Conv2D(512, **conv_params)(pool4)
    conv5 = Conv2D(512, **conv_params)(conv5)

    # Upsample with same kernel size as pooling layer
    up6 = concatenate([UpSampling2D(size=pool_size)(conv5), conv4])
    conv6 = Conv2D(256, **conv_params)(up6)
    conv6 = Conv2D(256, **conv_params)(conv6)

    up7 = concatenate([UpSampling2D(size=pool_size)(conv6), conv3])
    conv7 = Conv2D(128, **conv_params)(up7)
    conv7 = Conv2D(128, **conv_params)(conv7)

    up8 = concatenate([UpSampling2D(size=pool_size)(conv7), conv2])
    conv8 = Conv2D(64, **conv_params)(up8)
    conv8 = Conv2D(64, **conv_params)(conv8)

    up9 = concatenate([UpSampling2D(size=pool_size)(conv8), conv1])
    conv9 = Conv2D(32, **conv_params)(up9)
    conv9 = Conv2D(32, **conv_params)(conv9)

    conv10 = Conv2D(1, kernel_size=1, activation='sigmoid')(conv9)

    unet_model = keras.models.Model(inputs=inputs, outputs=conv10)
    unet_model.compile(optimizer=Adam(lr=1e-4), loss=get_f1_dist,
                       metrics=[get_f1_score])

    return unet_model


if __name__ == '__main__':

    ###################################
    # Get training/testing data
    ###################################
    npz_dir = op.join(os.environ['BUILDS_DIR'], 'ml-data-generation', 'data')
    x_train, y_train, x_test, y_test = load_data(npz_dir)

    ###################################
    # Create the model
    ###################################
    # Define some training parameters
    K.set_image_data_format('channels_last')  # Ensure image format is right
    img_rows, img_cols, n_bands = 256, 256, 3
    epochs = 3
    batch_size = 16
    start_time = dt.now().strftime("%m%d_%H%M%S")

    model = get_unet(input_shape=(img_rows, img_cols, n_bands))

    # Define callbacks to save the model after each epoch
    ckpt_fpath = op.join(start_time + '_L{val_loss:.2f}_E{epoch:02d}.hdf5')
    callbacks_list = [ModelCheckpoint(ckpt_fpath, monitor='val_loss',
                                      save_best_only=True)]
    ###################################
    # Train model
    ###################################
    # Define image generator
    datagen = ImageDataGenerator(
        #rescale=1./255, # Use this if not using standardizing loaded data
        horizontal_flip=True, vertical_flip=True)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=int(x_train.shape[0] / batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        verbose=1)

    final_score = model.evaluate(x_test, y_test, verbose=0)
    print('Final test loss: {:0.4f}'.format(final_score[0]))

    ###################################
    # Plot results
    ###################################
