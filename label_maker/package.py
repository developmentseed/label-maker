# pylint: disable=unused-argument
"""Generate an .npz file containing arrays for training machine learning algorithms"""

from os import path as op
from urllib.parse import urlparse
import numpy as np
from PIL import Image


def package_directory(dest_folder, classes, imagery, ml_type, seed=False, train_size=0.8, **kwargs):
    """Generate an .npz file containing arrays for training machine learning algorithms

    Parameters
    ------------
    dest_folder: str
        Folder to save labels, tiles, and final numpy arrays into
    classes: list
        A list of classes for machine learning training. Each class is defined as a dict
        with two required properties:
          - name: class name
          - filter: A Mapbox GL Filter.
        See the README for more details
    imagery: str
        Imagery template to download satellite images from.
        Ex: http://a.tiles.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.jpg?access_token=ACCESS_TOKEN
    ml_type: str
        Defines the type of machine learning. One of "classification", "object-detection", or "segmentation"
    train_percent: float
        Portion of the data to use in training, the remainder is used as test data (default 0.8)
    **kwargs: dict
        Other properties from CLI config passed as keywords to other utility functions
    """
    # if a seed is given, use it
    if seed:
        np.random.seed(seed)

    # open labels file, create tile array
    labels_file = op.join(dest_folder, 'labels.npz')
    labels = np.load(labels_file)
    tile_names = [tile for tile in labels.files]
    tile_names.sort()
    tiles = np.array(tile_names)
    np.random.shuffle(tiles)

    # find maximum number of features in advance so numpy shapes match
    if ml_type == 'object-detection':
        max_features = 0
        for tile in labels.files:
            features = len(labels[tile])
            if features > max_features:
                max_features = features

    x_vals = []
    y_vals = []

    # open the images and load those plus the labels into the final arrays
    o = urlparse(imagery)
    _, image_format = op.splitext(o.path)
    for tile in tiles:
        image_file = op.join(dest_folder, 'tiles', '{}{}'.format(tile, image_format))
        try:
            img = Image.open(image_file)
        except FileNotFoundError:
            print('Couldn\'t open {}, skipping'.format(image_file))
            continue

        np_image = np.array(img)
        img.close()

        x_vals.append(np_image)
        if ml_type == 'classification':
            y_vals.append(labels[tile])
        elif ml_type == 'object-detection':
            # zero pad object-detection arrays
            cl = labels[tile]
            y_vals.append(np.concatenate((cl, np.zeros((max_features - len(cl), 5)))))
        elif ml_type == 'segmentation':
            y_vals.append(labels[tile][..., np.newaxis])  # Add grayscale channel

    # split into train and test
    split_index = int(len(x_vals) * train_size)

    # convert lists to numpy arrays
    x_vals = np.array(x_vals, dtype=np.uint8)
    y_vals = np.array(y_vals, dtype=np.uint8)

    print('Saving packaged file to {}'.format(op.join(dest_folder, 'data.npz')))
    np.savez(op.join(dest_folder, 'data.npz'),
             x_train=x_vals[:split_index, ...],
             y_train=y_vals[:split_index, ...],
             x_test=x_vals[split_index:, ...],
             y_test=y_vals[split_index:, ...])
