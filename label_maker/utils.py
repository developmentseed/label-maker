# pylint: disable=unused-argument
"""Provide utility functions"""
from os import path as op
from urllib.parse import urlparse, parse_qs

from mercantile import bounds
from pyproj import Proj, transform
from PIL import Image
import numpy as np
import requests
import rasterio

def url(tile, imagery):
    """Return a tile url provided an imagery template and a tile"""
    return imagery.replace('{x}', tile[0]).replace('{y}', tile[1]).replace('{z}', tile[2])

def class_match(ml_type, label, i):
    """Determine if a label matches a given class index"""
    if ml_type == 'classification':
        return label[i] > 0
    elif ml_type == 'object-detection':
        return len(list(filter(lambda bb: bb[4] == i, label)))
    elif ml_type == 'segmentation':
        return np.count_nonzero(label == i)
    return None

def download_tile_tms(tile, imagery, folder, *args):
    """Download a satellite image tile from a tms endpoint"""
    o = urlparse(imagery)
    _, image_format = op.splitext(o.path)
    r = requests.get(url(tile.split('-'), imagery))
    tile_img = op.join(folder, '{}{}'.format(tile, image_format))
    with open(tile_img, 'wb')as w:
        w.write(r.content)
    return tile_img

def get_tile_tif(tile, imagery, folder, imagery_offset):
    """
    Read a GeoTIFF with a window corresponding to a TMS tile

    The TMS tile bounds are converted to the GeoTIFF source CRS. That bounding
    box is converted to a pixel window which is read from the GeoTIFF. For
    remote files which are internally tiled, this will take advantage of HTTP
    GET Range Requests to avoid downloading the entire file. See more info at:
    http://www.cogeo.org/in-depth.html
    """
    bound = bounds(*[int(t) for t in tile.split('-')])
    imagery_offset = imagery_offset or [0, 0]
    with rasterio.open(imagery) as src:
        x_res, y_res = src.transform[0], src.transform[4]
        p1 = Proj({'init': 'epsg:4326'})
        p2 = Proj(**src.crs)

        # offset our imagery in the "destination pixel" space
        offset_bound = dict()
        deg_per_pix_x = (bound.west - bound.east) / 256
        deg_per_pix_y = (bound.north - bound.south) / 256

        offset_bound['west'] = bound.west + imagery_offset[0] * deg_per_pix_x
        offset_bound['east'] = bound.east + imagery_offset[0] * deg_per_pix_x
        offset_bound['north'] = bound.north + imagery_offset[1] * deg_per_pix_y
        offset_bound['south'] = bound.south + imagery_offset[1] * deg_per_pix_y

        # project tile boundaries from lat/lng to source CRS
        tile_ul_proj = transform(p1, p2, offset_bound['west'], offset_bound['north'])
        tile_lr_proj = transform(p1, p2, offset_bound['east'], offset_bound['south'])
        # get origin point from the TIF
        tif_ul_proj = (src.bounds.left, src.bounds.top)

        # use the above information to calculate the pixel indices of the window
        top = int((tile_ul_proj[1] - tif_ul_proj[1]) / y_res)
        left = int((tile_ul_proj[0] - tif_ul_proj[0]) / x_res)
        bottom = int((tile_lr_proj[1] - tif_ul_proj[1]) / y_res)
        right = int((tile_lr_proj[0] - tif_ul_proj[0]) / x_res)

        window = ((top, bottom), (left, right))

        # read the first three bands (assumed RGB) of the TIF into an array
        data = np.empty(shape=(3, 256, 256)).astype(src.profile['dtype'])
        for k in (1, 2, 3):
            src.read(k, window=window, out=data[k - 1], boundless=True)

        # save
        tile_img = op.join(folder, '{}{}'.format(tile, '.jpg'))
        img = Image.fromarray(np.moveaxis(data, 0, -1), mode='RGB')
        img.save(tile_img)

    return tile_img

def get_tile_wms(tile, imagery, folder, imagery_offset):
    """
    Read a WMS endpoint with query parameters corresponding to a TMS tile

    Converts the tile boundaries to the spatial reference system (SRS) specified
    by the WMS query parameter.
    """
    # retrieve the necessary parameters from the query string
    query_dict = parse_qs(imagery.lower())
    image_format = query_dict['format'][0].split('/')[1]
    wms_srs = query_dict['srs'][0]

    # find our tile bounding box
    bound = bounds(*[int(t) for t in tile.split('-')])
    p1 = Proj({'init': 'epsg:4326'})
    p2 = Proj({'init': wms_srs})

    # project the tile bounding box from lat/lng to WMS SRS
    tile_ll_proj = transform(p1, p2, bound.west, bound.south)
    tile_ur_proj = transform(p1, p2, bound.east, bound.north)
    bbox = tile_ll_proj + tile_ur_proj

    # request the image with the transformed bounding box and save
    wms_url = imagery.replace('{bbox}', ','.join([str(b) for b in bbox]))
    r = requests.get(wms_url)
    tile_img = op.join(folder, '{}.{}'.format(tile, image_format))
    with open(tile_img, 'wb') as w:
        w.write(r.content)
    return tile_img


def is_tif(imagery):
    """Determine if an imagery path leads to a valid tif"""
    valid_drivers = ['GTiff', 'VRT']
    try:
        with rasterio.open(imagery) as test_ds:
            if test_ds.meta['driver'] not in valid_drivers:
                # rasterio can open path, but it is not a tif
                valid_tif = False
            else:
                valid_tif = True
    except rasterio._err.CPLE_HttpResponseError: #pylint: disable=protected-access
        # rasterio cannot open the path. this is the case for a
        # tile service
        valid_tif = False

    return valid_tif

def is_wms(imagery):
    """Determine if an imagery path is a WMS endpoint"""
    return '{bbox}' in imagery

def get_image_function(imagery):
    """Return the correct image downloading function based on the imagery string"""
    if is_tif(imagery):
        return get_tile_tif
    if is_wms(imagery):
        return get_tile_wms
    return download_tile_tms
