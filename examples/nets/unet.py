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
from scipy.ndimage.filters import gaussian_filter1d
import keras
from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import (Input, Conv2D, MaxPooling2D, concatenate,
                          UpSampling2D, Cropping2D)

from label_maker.utils.plot_utils import plot_overlays


def load_data(data_dir, zero_mean=True, unit_var=True):
    """Helper to load data and print basic information"""
    # Load data, preshuffled and split between train and test sets
    npz = np.load(op.join(data_dir, 'data_SA_segmentation.npz'))
    x_train = npz['x_train'].astype(np.float)
    x_test = npz['x_test'].astype(np.float)
    y_train = npz['y_train'].astype(np.float)
    y_test = npz['y_test'].astype(np.float)

    print('Loaded {} training examples and {} testing examples'.format(
        x_train.shape[0], x_test.shape[0]))

    return x_train, y_train, x_test, y_test


def expand_mask(Y_mask, sigma=1):
    """Widen single-pixel-wide segmentation masks"""

    # Seg maps are 1 pix wide; need to give roads some width
    # Use gaussian filter as poor man's way to accomplish this
    Y_mask_expanded = gaussian_filter1d(Y_mask, sigma=sigma, axis=1)
    Y_mask_expanded = gaussian_filter1d(Y_mask_expanded, sigma=sigma, axis=2)
    Y_mask_expanded[Y_mask_expanded > 0.1] = 1

    return Y_mask_expanded

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
    total = K.sum(K.square(y_true_f) + K.square(y_pred_f))

    return (2. * intersection + smooth) / (total + smooth)


def get_f1_dist(y_true, y_pred):
    """Helper to turn the F1 score into a loss"""
    return -1 * get_f1_score(y_true, y_pred)


def get_unet(input_shape, strides=1, pool_size=2):
    """Return vanilla u-net model ready for training.

    Parameters
    ----------
    input_shape: iterable
        Should be tuple, list, or array specifying (img_width,
        img_height, n_channels). Keras will prepend batch dimension
    strides: int or tuple
        Size of stride for 2D convolutions. Default is 1.
    pool_size: tuple
        Size of kernal for max-pooling and upsampling. Default is 2.

    Returns
    -------
    unet_model: Keras.models.Model
        Compiled model ready for training.
    """
    conv_params = dict(activation='relu', kernel_size=2,
                       strides=strides, padding='valid',
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

    up4 = concatenate([UpSampling2D(size=pool_size)(conv3),
                       Cropping2D([(2, 3), (2, 3)])(conv2)])
    conv4 = Conv2D(64, **conv_params)(up4)
    conv4 = Conv2D(64, **conv_params)(conv4)

    up5 = concatenate([UpSampling2D(size=pool_size)(conv4),
                       Cropping2D([(9, 10), (9, 10)])(conv1)])
    conv5 = Conv2D(32, **conv_params)(up5)
    conv5 = Conv2D(32, **conv_params)(conv5)

    conv6 = Conv2D(1, kernel_size=1, activation='sigmoid')(conv5)

    unet_model = keras.models.Model(inputs=inputs, outputs=conv6)
    unet_model.compile(optimizer=Adam(lr=5e-4), loss=get_f1_dist,
                       metrics=[get_f1_score])

    return unet_model


if __name__ == '__main__':

    ###################################
    # Get training/testing data
    ###################################
    npz_dir = op.join(os.environ['BUILDS_DIR'], 'label-maker', 'data')
    X_train, Y_train, X_test, Y_test = load_data(npz_dir)
    Y_train = expand_mask(Y_train)
    Y_test = expand_mask(Y_test)

    ###################################
    # Create the model
    ###################################
    # Define some training parameters
    K.set_image_data_format('channels_last')  # Ensure image format is right
    img_rows, img_cols, n_bands = 256, 256, 3
    epochs = 5
    batch_size = 32
    cb = 11  # Crop border (to make size of Y masks match size of predictions)
    start_time = dt.now().strftime("%m%d_%H%M%S")
    Y_train = Y_train[:, cb:-cb, cb:-cb, :]
    Y_test = Y_test[:, cb:-cb, cb:-cb, :]

    model = get_unet(input_shape=(img_rows, img_cols, n_bands))

    ###################################
    # Train model
    ###################################
    # Define image generator
    datagen = ImageDataGenerator(
        #rescale=1./255, # Use this if you don't standardize data
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=(1, 1.1))
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=int(X_train.shape[0] / batch_size),
                        epochs=epochs,
                        validation_data=datagen.flow(X_test, Y_test, batch_size=batch_size),
                        validation_steps=int(X_test.shape[0] / batch_size),
                        verbose=1)

    print('Training complete')

    ###################################
    # Plot results
    ###################################
    n_preds = 9  # Number of images to show

    # Plot segmentation predictions
    # Use same generator so preprocessing is identical
    X_test_plot, Y_test_plot = np.copy(X_test[:n_preds, ...]), np.copy(Y_test[:n_preds, ...])
    X_test_plot -= datagen.mean
    X_test_plot /= datagen.std

    seg_preds = model.predict(X_test_plot)

    seg_plot = plot_overlays(X_test[:n_preds, cb:-cb, cb:-cb, :],
                             seg_preds, img_a=0.5, mask_a=0.5)
    seg_plot.savefig(op.join(os.environ['BUILDS_DIR'], 'label-maker',
                             'examples', 'images', 'seg_prediction.png'), dpi=150)

    # Plot original mask and original images
    mask_plot = plot_overlays(X_test[:n_preds, cb:-cb, cb:-cb, :],
                              Y_test_plot, 0.75, 0.5, False)
    mask_plot.savefig(op.join(os.environ['BUILDS_DIR'], 'label-maker',
                              'examples', 'images', 'seg_labels.png'), dpi=150)
    orig_plot = plot_overlays(X_test[:n_preds, cb:-cb, cb:-cb, :],
                              Y_test_plot, 1, 0.01, False)
    orig_plot.savefig(op.join(os.environ['BUILDS_DIR'], 'label-maker',
                              'examples', 'images', 'seg_orig_images.png'), dpi=150)
