# pylint: disable=unused-argument
"""Generate an .npz file containing arrays for training machine learning algorithms"""

import concurrent.futures
from os import makedirs, path as op
from random import shuffle

import numpy as np

from label_maker.utils import get_image_function

def download_images(dest_folder, classes, imagery, ml_type, background_ratio, threadcount, imagery_offset=False, **kwargs):
    """Download satellite images specified by a URL and a label.npz file
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
    background_ratio: float
        Determines the number of background images to download in single class problems. Ex. A value
        of 1 will download an equal number of background images to class images.
    threadcount: int
        Number of threads to use for faster parallel image download
    imagery_offset: list
        An optional list of integers representing the number of pixels to offset imagery. Ex. [15, -5] will
        move the images 15 pixels right and 5 pixels up relative to the requested tile bounds
    **kwargs: dict
        Other properties from CLI config passed as keywords to other utility functions
    """
    # open labels file
    labels_file = op.join(dest_folder, 'labels.npz')
    tiles = np.load(labels_file)

    # create tiles directory
    tiles_dir = op.join(dest_folder, 'tiles')
    if not op.isdir(tiles_dir):
        makedirs(tiles_dir)

    # find tiles which have any matching class
    def class_test(value):
        """Determine if a label matches a given class index"""
        if ml_type == 'object-detection':
            return len(value)
        elif ml_type == 'segmentation':
            return np.sum(value) > 0
        elif ml_type == 'classification':
            return value[0] == 0
        return None
    class_tiles = [tile for tile in tiles.files if class_test(tiles[tile])]

    # for classification problems, we also get background
    # tiles up to len(class_tiles) * config.get('background_ratio')
    background_tiles = []
    if ml_type == 'classification':
        limit = len(class_tiles) * background_ratio
        background_tiles_full = [tile for tile in tiles.files if tile not in class_tiles]
        shuffle(background_tiles_full)
        background_tiles = background_tiles_full[:limit]

    # download tiles
    tiles = class_tiles + background_tiles
    print('Downloading {} tiles to {}'.format(len(tiles), tiles_dir))

    # get image acquisition function based on imagery string
    image_function = get_image_function(imagery)
    kwargs['imagery_offset'] = imagery_offset

    with concurrent.futures.ThreadPoolExecutor(max_workers=threadcount) as executor:
        [executor.submit(image_function, tile, imagery, tiles_dir, kwargs) for tile in tiles]
        executor.shutdown(wait=True)
