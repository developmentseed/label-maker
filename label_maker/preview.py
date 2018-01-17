# pylint: disable=unused-argument
"""Produce imagery examples for specified classes"""

from os import path as op
from os import makedirs
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image, ImageDraw

from label_maker.utils import url, class_match

def preview(dest_folder, number, classes, imagery, ml_type, **kwargs):
    """Produce imagery examples for specified classes

    Parameters
    ------------
    dest_folder: str
        Folder to save labels and example tiles into
    number: int
        Number of preview images to download per class
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
    **kwargs: dict
        Other properties from CLI config passed as keywords to other utility functions
    """
    # open labels file
    labels_file = op.join(dest_folder, 'labels.npz')
    tiles = np.load(labels_file)

    # create example tiles directory
    examples_dir = op.join(dest_folder, 'examples')
    if not op.isdir(examples_dir):
        makedirs(examples_dir)

    # find examples tiles for each class and download
    print('Writing example images to {}'.format(examples_dir))
    o = urlparse(imagery)
    _, image_format = op.splitext(o.path)
    for i, cl in enumerate(classes):
        # create class directory
        class_dir = op.join(dest_folder, 'examples', cl.get('name'))
        if not op.isdir(class_dir):
            makedirs(class_dir)

        class_tiles = [t for t in tiles.files if class_match(ml_type, tiles[t], i + 1)]
        class_tiles = class_tiles[:number]
        print('Downloading {} tiles for class {}'.format(len(class_tiles), cl.get('name')))
        for tile in class_tiles:
            r = requests.get(url(tile.split('-'), imagery))
            tile_img = op.join(dest_folder, 'examples', cl.get('name'),
                               '{}{}'.format(tile, image_format))
            open(tile_img, 'wb').write(r.content)
            if ml_type == 'object-detection':
                img = Image.open(tile_img)
                draw = ImageDraw.Draw(img)
                for box in tiles[tile]:
                    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline='red')
                img.save(tile_img)
            elif ml_type == 'segmentation':
                final = Image.new('RGB', (256, 256))
                img = Image.open(tile_img)
                mask = Image.fromarray(tiles[tile] * 255)
                final.paste(img, mask)
                final.save(tile_img)
