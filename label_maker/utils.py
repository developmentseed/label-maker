# pylint: disable=unused-argument
"""Provide utility functions"""
import os
from os import path as op
from urllib.parse import urlparse, parse_qs

from mercantile import bounds, Tile, children
from PIL import Image
import io
import numpy as np
import requests
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform, transform_bounds
from rasterio.windows import Window

WGS84_CRS = CRS.from_epsg(4326)

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

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

def get_image_format(imagery, kwargs):
    if kwargs.get('tms_image_format'):
        image_format =  kwargs.get('tms_image_format')
    else:
        o = urlparse(imagery)
        _, image_format = op.splitext(o.path)
    return image_format

def download_tile_tms(tile, imagery, folder, kwargs):
    """Download a satellite image tile from a tms endpoint"""

    image_format = get_image_format(imagery, kwargs)

    if os.environ.get('ACCESS_TOKEN'):
        token = os.environ.get('ACCESS_TOKEN')
        imagery = imagery.format_map(SafeDict(ACCESS_TOKEN=token))

    r = requests.get(url(tile.split('-'), imagery),
                     auth=kwargs.get('http_auth'))
    tile_img = op.join(folder, '{}{}'.format(tile, image_format))
    tile = tile.split('-')

    over_zoom = kwargs.get('over_zoom')
    if over_zoom:
        new_zoom = over_zoom + kwargs.get('zoom')
        # get children
        child_tiles = children(int(tile[0]), int(tile[1]), int(tile[2]), zoom=new_zoom)
        child_tiles.sort()

        new_dim = 256 * (2 * over_zoom)

        w_lst = []
        for i in range (2 * over_zoom):
            for j in range(2 * over_zoom):
                window = Window(i * 256, j * 256, 256, 256)
                w_lst.append(window)

        # request children
        with rasterio.open(tile_img, 'w', driver='jpeg', height=new_dim,
                        width=new_dim, count=3, dtype=rasterio.uint8) as w:
                for num, t in enumerate(child_tiles):
                    t = [str(t[0]), str(t[1]), str(t[2])]
                    r = requests.get(url(t, imagery),
                                    auth=kwargs.get('http_auth'))
                    img = np.array(Image.open(io.BytesIO(r.content)), dtype=np.uint8)
                    try:
                        img = img.reshape((256, 256, 3)) # 4 channels returned from some endpoints, but not all
                    except ValueError:
                        img = img.reshape((256, 256, 4))
                    img = img[:, :, :3]
                    img = np.rollaxis(img, 2, 0)
                    w.write(img, window=w_lst[num])
    else:
        r = requests.get(url(tile, imagery),
                         auth=kwargs.get('http_auth'))
        with open(tile_img, 'wb')as w:
            w.write(r.content)
    return tile_img

def get_tile_tif(tile, imagery, folder, kwargs):
    """
    Read a GeoTIFF with a window corresponding to a TMS tile

    The TMS tile bounds are converted to the GeoTIFF source CRS. That bounding
    box is converted to a pixel window which is read from the GeoTIFF. For
    remote files which are internally tiled, this will take advantage of HTTP
    GET Range Requests to avoid downloading the entire file. See more info at:
    http://www.cogeo.org/in-depth.html
    """
    bound = bounds(*[int(t) for t in tile.split('-')])
    imagery_offset = kwargs.get('imagery_offset') or [0, 0]
    with rasterio.open(imagery) as src:
        profile = src.profile
        x_res, y_res = src.transform[0], src.transform[4]

        # offset our imagery in the "destination pixel" space
        offset_bound = dict()
        deg_per_pix_x = (bound.west - bound.east) / 256
        deg_per_pix_y = (bound.north - bound.south) / 256

        offset_bound['west'] = bound.west + imagery_offset[0] * deg_per_pix_x
        offset_bound['east'] = bound.east + imagery_offset[0] * deg_per_pix_x
        offset_bound['north'] = bound.north + imagery_offset[1] * deg_per_pix_y
        offset_bound['south'] = bound.south + imagery_offset[1] * deg_per_pix_y

        # project tile boundaries from lat/lng to source CRS
        x, y = transform(WGS84_CRS, src.crs, [offset_bound['west']], [offset_bound['north']])
        tile_ul_proj = x[0], y[0]

        x, y = transform(WGS84_CRS, src.crs, [offset_bound['east']], [offset_bound['south']])
        tile_lr_proj = x[0], y[0]

        # get origin point from the TIF
        tif_ul_proj = (src.bounds.left, src.bounds.top)

        # use the above information to calculate the pixel indices of the window
        top = int((tile_ul_proj[1] - tif_ul_proj[1]) / y_res)
        left = int((tile_ul_proj[0] - tif_ul_proj[0]) / x_res)
        bottom = int((tile_lr_proj[1] - tif_ul_proj[1]) / y_res)
        right = int((tile_lr_proj[0] - tif_ul_proj[0]) / x_res)

        window = ((top, bottom), (left, right))

        # read the first three bands (assumed RGB) of the TIF into an array
        band_indices = kwargs.get('band_indices', (1, 2, 3))
        band_count = len(band_indices)

        arr_shape = (band_count, 256, 256)
        data = np.empty(shape=(arr_shape)).astype(profile['dtype'])

        for i, k in enumerate(band_indices):
            src.read(k, window=window, out=data[i], boundless=True)
        # save
        tile_img = op.join(folder, '{}{}'.format(tile, '.tif'))
        with rasterio.open(tile_img, 'w', driver='GTiff', height=256,
                width=256, count=band_count, dtype=profile['dtype']) as w:
                w.write(data)
    return tile_img

def get_tile_wms(tile, imagery, folder, kwargs):
    """
    Read a WMS endpoint with query parameters corresponding to a TMS tile

    Converts the tile boundaries to the spatial/coordinate reference system
    (SRS or CRS) specified by the WMS query parameter.
    """
    # retrieve the necessary parameters from the query string
    query_dict = parse_qs(imagery.lower())
    image_format = query_dict.get('format')[0].split('/')[1]
    wms_version = query_dict.get('version')[0]
    if wms_version == '1.3.0':
        wms_srs = query_dict.get('crs')[0]
    else:
        wms_srs = query_dict.get('srs')[0]

    # find our tile bounding box
    bound = bounds(*[int(t) for t in tile.split('-')])
    xmin, ymin, xmax, ymax = transform_bounds(WGS84_CRS, CRS.from_string(wms_srs), *bound, densify_pts=21)

    # project the tile bounding box from lat/lng to WMS SRS
    bbox = (
        [ymin, xmin, ymax, xmax] if wms_version == "1.3.0" else [xmin, ymin, xmax, ymax]
    )

    # request the image with the transformed bounding box and save
    wms_url = imagery.replace('{bbox}', ','.join([str(b) for b in bbox]))
    r = requests.get(wms_url, auth=kwargs.get('http_auth'))
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
    except rasterio.errors.RasterioIOError:
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
