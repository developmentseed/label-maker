# pylint: disable=unused-argument,too-many-nested-blocks
"""Create label data from OSM QA tiles for specified classes"""

import sys
from os import makedirs, path as op
from subprocess import run, Popen, PIPE
import json
from functools import partial

import numpy as np
import mapbox_vector_tile
from shapely.geometry import shape, mapping, Polygon
from shapely.errors import TopologicalError
from rasterio.features import rasterize
from geojson import Feature, FeatureCollection as fc
from mercantile import tiles, feature, Tile
from PIL import Image, ImageDraw
from tilepie import tilereduce

import label_maker
from label_maker.utils import class_match
from label_maker.filter import create_filter
from label_maker.palette import class_color

# declare a global accumulator so the workers will have access
tile_results = dict()

# clip all geometries to a tile
clip_mask = Polygon(((0, 0), (0, 255), (255, 255), (255, 0), (0, 0)))

def make_labels(dest_folder, zoom, country, classes, ml_type, bounding_box, sparse, **kwargs):
    """Create label data from OSM QA tiles for specified classes

    Perform the following operations:
    - If necessary, re-tile OSM QA Tiles to the specified zoom level
    - Iterate over all tiles within the bounding box and produce a label for each
    - Save the label file as labels.npz
    - Create an output for previewing the labels (GeoJSON or PNG depending upon ml_type)

    Parameters
    ------------
    dest_folder: str
        Folder to save labels and example tiles into
    zoom: int
        The zoom level to create tiles at
    country: str
        The OSM QA Tile extract to download. The value should be a country string matching a value found in
        `label_maker/countries.txt`
    classes: list
        A list of classes for machine learning training. Each class is defined as a dict
        with two required properties:
          - name: class name
          - filter: A Mapbox GL Filter.
        See the README for more details
    ml_type: str
        Defines the type of machine learning. One of "classification", "object-detection", or "segmentation"
    bounding_box: list
        The bounding box to create images from. This should be given in the form: `[xmin, ymin, xmax, ymax]`
        as longitude and latitude values between `[-180, 180]` and `[-90, 90]` respectively
    sparse: boolean
        Limit the total background tiles to write based on `background_ratio` kwarg.
    geojson: str
        Filepath to optional geojson label input
    **kwargs: dict
        Other properties from CLI config passed as keywords to other utility functions
    """

    mbtiles_file = op.join(dest_folder, '{}.mbtiles'.format(country))
    mbtiles_file_zoomed = op.join(dest_folder, '{}-z{!s}.mbtiles'.format(country, zoom))

    if not op.exists(mbtiles_file_zoomed):
        filtered_geo = kwargs.get('geojson') or op.join(dest_folder, '{}.geojson'.format(country))
        fast_parse = []
        if not op.exists(filtered_geo):
            fast_parse = ['-P']
            print('Retiling QA Tiles to zoom level {} (takes a bit)'.format(zoom))
            ps = Popen(['tippecanoe-decode', '-c', '-f', mbtiles_file], stdout=PIPE)
            stream_filter_fpath = op.join(op.dirname(label_maker.__file__), 'stream_filter.py')
            run([sys.executable, stream_filter_fpath, json.dumps(bounding_box)],
                stdin=ps.stdout, stdout=open(filtered_geo, 'w'))
            ps.wait()
        run(['tippecanoe', '--no-feature-limit', '--no-tile-size-limit'] + fast_parse +
            ['-l', 'osm', '-f', '-z', str(zoom), '-Z', str(zoom), '-o',
             mbtiles_file_zoomed, filtered_geo])

    # Call tilereduce
    print('Determining labels for each tile')
    mbtiles_to_reduce = mbtiles_file_zoomed
    tilereduce(dict(zoom=zoom, source=mbtiles_to_reduce, bbox=bounding_box,
                    args=dict(ml_type=ml_type, classes=classes)),
               _mapper, _callback, _done)

    # Add empty labels to any tiles which didn't have data
    empty_label = _create_empty_label(ml_type, classes)
    for tile in tiles(*bounding_box, [zoom]):
        index = '-'.join([str(i) for i in tile])
        global tile_results
        if tile_results.get(index) is None:
            tile_results[index] = empty_label

    # Print a summary of the labels
    _tile_results_summary(ml_type, classes)

    # If the --sparse flag is provided, limit the total background tiles to write
    if sparse:
        pos_examples, neg_examples = [], []
        for k in tile_results.keys():
            # if we don't match any class, this is a negative example
            if not sum([class_match(ml_type, tile_results[k], i + 1) for i, c in enumerate(classes)]):
                neg_examples.append(k)
            else:
                pos_examples.append(k)

        # Choose random subset of negative examples
        n_neg_ex = int(kwargs['background_ratio'] * len(pos_examples))
        neg_examples = np.random.choice(neg_examples, n_neg_ex, replace=False).tolist()

        tile_results = {k: tile_results.get(k) for k in pos_examples + neg_examples}
        print('Using sparse mode; subselected {} background tiles'.format(n_neg_ex))

    # write out labels as numpy arrays
    labels_file = op.join(dest_folder, 'labels.npz')
    print('Writing out labels to {}'.format(labels_file))
    np.savez(labels_file, **tile_results)

    # write out labels as GeoJSON or PNG
    if ml_type == 'classification':
        features = []
        for tile, label in tile_results.items():
            feat = feature(Tile(*[int(t) for t in tile.split('-')]))
            features.append(Feature(geometry=feat['geometry'],
                                    properties=dict(label=label.tolist())))
        json.dump(fc(features), open(op.join(dest_folder, 'classification.geojson'), 'w'))
    elif ml_type == 'object-detection':
        label_folder = op.join(dest_folder, 'labels')
        if not op.isdir(label_folder):
            makedirs(label_folder)
        for tile, label in tile_results.items():
            # if we have at least one bounding box label
            if bool(label.shape[0]):
                label_file = '{}.png'.format(tile)
                img = Image.new('RGB', (256, 256))
                draw = ImageDraw.Draw(img)
                for box in label:
                    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=class_color(box[4]))
                print('Writing {}'.format(label_file))
                img.save(op.join(label_folder, label_file))
    elif ml_type == 'segmentation':
        label_folder = op.join(dest_folder, 'labels')
        if not op.isdir(label_folder):
            makedirs(label_folder)
        for tile, label in tile_results.items():
            # if we have any class pixels
            if np.sum(label):
                label_file = '{}.png'.format(tile)
                visible_label = np.array([class_color(l) for l in np.nditer(label)]).reshape(256, 256, 3)
                img = Image.fromarray(visible_label.astype(np.uint8))
                print('Writing {}'.format(label_file))
                img.save(op.join(label_folder, label_file))


def _mapper(x, y, z, data, args):
    """Iterate over OSM QA Tiles and return a label for each tile

    Iterate over the .mbtiles input. Decode the provided vector tile. Depending upon the
    desired type of eventual machine learning training, return a label list:
    - For 'object-detection' tasks, each list element is a bounding box like [xmin, ymin, xmax, ymax, class_index].
      There is one list element for each feature matching the provided classes.
    - For 'classification' tasks, the entire list is a "one-hot" vector representing which class
      the tile matches

    Parameters
    ------------
    x, y, z: int
        tile indices
    data: str
        Encoded vector tile data
    args: dict
        Additional arguments passed to the tile worker

    Returns
    ---------
    label: tuple
        The first element is a tile index of the form x-y-z. The second element is a list
        representing the label of the tile

    """
    ml_type = args.get('ml_type')
    classes = args.get('classes')

    if data is None:
        return ('{!s}-{!s}-{!s}'.format(x, y, z), _create_empty_label(ml_type, classes))

    tile = mapbox_vector_tile.decode(data)
    # for each class, determine if any features in the tile match

    if tile['osm']['features']:
        if ml_type == 'classification':
            class_counts = np.zeros(len(classes) + 1, dtype=np.int)
            for i, cl in enumerate(classes):
                ff = create_filter(cl.get('filter'))
                class_counts[i + 1] = int(bool([f for f in tile['osm']['features'] if ff(f)]))
            # if there are no classes, activate the background
            if np.sum(class_counts) == 0:
                class_counts[0] = 1
            return ('{!s}-{!s}-{!s}'.format(x, y, z), class_counts)
        elif ml_type == 'object-detection':
            bboxes = _create_empty_label(ml_type, classes)
            for feat in tile['osm']['features']:
                for i, cl in enumerate(classes):
                    ff = create_filter(cl.get('filter'))
                    if ff(feat):
                        geo = shape(feat['geometry'])
                        if cl.get('buffer'):
                            geo = geo.buffer(cl.get('buffer'), 4)
                        bb = _pixel_bbox(geo.bounds) + [i + 1]
                        bboxes = np.append(bboxes, np.array([bb]), axis=0)
            return ('{!s}-{!s}-{!s}'.format(x, y, z), bboxes)
        elif ml_type == 'segmentation':
            geos = []
            for feat in tile['osm']['features']:
                for i, cl in enumerate(classes):
                    ff = create_filter(cl.get('filter'))
                    if ff(feat):
                        feat['geometry']['coordinates'] = _convert_coordinates(feat['geometry']['coordinates'])
                        geo = shape(feat['geometry'])
                        try:
                            geo = geo.intersection(clip_mask)
                        except TopologicalError as e:
                            print(e, 'skipping')
                            break
                        if cl.get('buffer'):
                            geo = geo.buffer(cl.get('buffer'), 4)
                        if not geo.is_empty:
                            geos.append((mapping(geo), i + 1))
            result = rasterize(geos, out_shape=(256, 256))
            return ('{!s}-{!s}-{!s}'.format(x, y, z), result)
    return ('{!s}-{!s}-{!s}'.format(x, y, z), np.array())

def _convert_coordinates(coords):
    # for points, return the coordinates converted
    if isinstance(coords[0], int):
        return list(map(_pixel_bounds_convert, enumerate(coords)))
    # for other geometries, recurse
    return list(map(_convert_coordinates, coords))

def _pixel_bbox(bb):
    """Convert a bounding box in 0-4096 to pixel coordinates"""
    # this will have coordinates in xmin, ymin, xmax, ymax order
    # because we flip the yaxis, we also need to reorder
    converted = list(map(_pixel_bounds_convert, enumerate([bb[0], bb[3], bb[2], bb[1]])))
    return _buffer_bbox(converted)

def _buffer_bbox(bb, buffer=4):
    """Buffer a bounding box in pixel coordinates"""
    return list(map(
        _clamp,
        [
            bb[0] - buffer,
            bb[1] - buffer,
            bb[2] + buffer,
            bb[3] + buffer
        ]
    ))

def _clamp(coordinate):
    """restrict a single coordinate to 0-255"""
    return max(0, min(255, coordinate))

def _pixel_bounds_convert(x):
    """Convert a single 0-4096 coordinate to a pixel coordinate"""
    (i, b) = x
    # input bounds are in the range 0-4096 by default: https://github.com/tilezen/mapbox-vector-tile
    # we want them to match our fixed imagery size of 256
    pixel = round(b * 255. / 4096) # convert to tile pixels
    return pixel if (i % 2 == 0) else 255 - pixel # flip the y axis

def _callback(tile_label):
    """Attach tile labels to a global tile_results dict"""
    if not tile_label:
        return
    global tile_results
    (tile, label) = tile_label
    tile_results[tile] = label

def _done():
    pass

def _bbox_class(class_index):
    """Create a function to determine if a bounding box label matches a given class"""
    def bc(x):
        """Determine if a bounding box label matches a given class"""
        return x[4] == class_index
    return bc

def _tile_results_summary(ml_type, classes):
    print('---')
    labels = list(tile_results.values())
    all_tiles = list(tile_results.keys())
    if ml_type == 'object-detection':
        # for each class, show number of features and number of tiles
        for i, cl in enumerate(classes):
            cl_features = len([bb for l in labels for bb in l if bb[4] == i + 1])
            cl_tiles = len([l for l in labels if len(list(filter(_bbox_class(i + 1), l)))]) # pylint: disable=cell-var-from-loop
            print('{}: {} features in {} tiles'.format(cl.get('name'), cl_features, cl_tiles))
    elif ml_type == 'classification':
        class_tile_counts = list(np.sum(labels, axis=0))
        for i, cl in enumerate(classes):
            print('{}: {} tiles'.format(cl.get('name'), int(class_tile_counts[i + 1])))
    elif ml_type == 'segmentation':
        for i, cl in enumerate(classes):
            count = len([l for l in labels if class_match(ml_type, l, i + 1)])
            print('{}: {} tiles'.format(cl.get('name'), count))

    print('Total tiles: {}'.format(len(all_tiles)))

def _create_empty_label(ml_type, classes):
    if ml_type == 'classification':
        label = np.zeros(len(classes) + 1, dtype=np.int)
        label[0] = 1
        return label
    elif ml_type == 'object-detection':
        return np.empty((0, 5), dtype=np.int)
    elif ml_type == 'segmentation':
        return np.zeros((256, 256), dtype=np.int)
    return None
