# pylint: disable=unused-argument
"""Generate an .npz file containing arrays for training machine learning algorithms"""

from os import path as op
from urllib.parse import urlparse
import numpy as np
import rasterio
from PIL import Image

from label_maker.utils import is_tif, get_image_format


def package_directory(dest_folder, classes, imagery, ml_type, seed=False,
                      split_names=('train', 'test'), split_vals=(0.8, .2),
                      **kwargs):
    """Generate an .npz file containing arrays for training machine learning algorithms

    Parameters
    ------------
    dest_folder: str
        Folder to save labels, tiles, and final numpy arrays into
    classes: list
        A list of classes for machine learning training. Each class is defined
        as a dict with two required properties:
          - name: class name
          - filter: A Mapbox GL Filter.
        See the README for more details
    imagery: str
        Imagery template to download satellite images from.
        Ex: http://a.tiles.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.jpg?access_token=ACCESS_TOKEN
    ml_type: str
        Defines the type of machine learning. One of "classification",
        "object-detection", or "segmentation"
    seed: int
        Random generator seed. Optional, use to make results reproducible.
    split_vals: tuple
        Percentage of data to put in each catagory listed in split_names. Must
        be floats and must sum to one. Default: (0.8, 0.2)
    split_names: tupel
        Default: ('train', 'test')
        List of names for each subset of the data.
    **kwargs: dict
        Other properties from CLI config passed as keywords to other utility
        functions.
    """
    # if a seed is given, use it
    if seed:
        np.random.seed(seed)

    if len(split_names) != len(split_vals):
        raise ValueError('`split_names` and `split_vals` must be the same '
                         'length. Please update your config.')
    if not np.isclose(sum(split_vals), 1):
        raise ValueError('`split_vals` must sum to one. Please update your config.')

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
    if is_tif(imagery):  # if a TIF is provided, use jpg as tile format
        img_dtype = rasterio.open(imagery).profile['dtype']
        image_format = '.tif'

    else:
        img_dtype = np.uint8
        image_format = get_image_format(imagery, kwargs)

    for tile in tiles:
        image_file = op.join(dest_folder, 'tiles', '{}{}'.format(tile, image_format))
        try:
            img = rasterio.open(image_file)
        except FileNotFoundError:
            # we often don't download images for each label (e.g. background tiles)
            continue
        except OSError:
            print('Couldn\'t open {}, skipping'.format(image_file))
            continue

        i = np.array(img.read())
        np_image = np.moveaxis(i, 0, 2)
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

    # Convert lists to numpy arrays

    #TO-DO flexible x_val dtype
    x_vals = np.array(x_vals, dtype=img_dtype)
    y_vals = np.array(y_vals, dtype=np.uint8)

    # Get number of data samples per split from the float proportions
    split_n_samps = [len(x_vals) * val for val in split_vals]

    if np.any(split_n_samps == 0):
        raise ValueError('Split must not generate zero samples per partition. '
                         'Change ratio of values in config file.')

    # Convert into a cumulative sum to get indices
    split_inds = np.cumsum(split_n_samps).astype(np.integer)

    # Exclude last index as `np.split` handles splitting without that value
    split_arrs_x = np.split(x_vals, split_inds[:-1])
    split_arrs_y = np.split(y_vals, split_inds[:-1])

    save_dict = {}

    for si, split_name in enumerate(split_names):
        save_dict['x_{}'.format(split_name)] = split_arrs_x[si]
        save_dict['y_{}'.format(split_name)] = split_arrs_y[si]

    np.savez(op.join(dest_folder, 'data.npz'), **save_dict)
    print('Saving packaged file to {}'.format(op.join(dest_folder, 'data.npz')))
    print('Image dtype written in npz matches input image dtype: {}'.format(img_dtype))
