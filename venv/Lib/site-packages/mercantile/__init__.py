"""Web mercator XYZ tile utilities"""

from collections import namedtuple
import math


__version__ = '1.0.0'

__all__ = [
    'Bbox', 'LngLat', 'LngLatBbox', 'Tile', 'bounding_tile', 'bounds',
    'children', 'feature', 'lnglat', 'parent', 'quadkey', 'quadkey_to_tile',
    'tile', 'tiles', 'ul', 'xy_bounds']


Tile = namedtuple('Tile', ['x', 'y', 'z'])
"""An XYZ web mercator tile

Attributes
----------
x, y, z : int
    x and y indexes of the tile and zoom level z.
"""


LngLat = namedtuple('LngLat', ['lng', 'lat'])
"""A longitude and latitude pair

Attributes
----------
lng, lat : float
    Longitude and latitude in decimal degrees east or north.
"""


LngLatBbox = namedtuple('LngLatBbox', ['west', 'south', 'east', 'north'])
"""A geographic bounding box

Attributes
----------
west, south, east, north : float
    Bounding values in decimal degrees.
"""


Bbox = namedtuple('Bbox', ['left', 'bottom', 'right', 'top'])
"""A web mercator bounding box

Attributes
----------
left, bottom, right, top : float
    Bounding values in meters.
"""


class MercantileError(Exception):
    """Base exception"""


class InvalidLatitudeError(MercantileError):
    """Raised when math errors occur beyond ~85 degrees N or S"""


def ul(*tile):
    """Returns the upper left longitude and latitude of a tile

    Parameters
    ----------
    tile : Tile or ints
        May be be either an instance of Tile or 3 ints, X, Y, Z.

    Returns
    -------
    LngLat

    Examples
    --------

    >>> ul(Tile(x=0, y=0, z=1))
    LngLat(lng=-180.0, lat=85.0511287798066)

    >>> mercantile.ul(1, 1, 1)
    LngLat(lng=0.0, lat=0.0)
    """
    if len(tile) == 1:
        tile = tile[0]
    xtile, ytile, zoom = tile
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return LngLat(lon_deg, lat_deg)


def bounds(*tile):
    """Returns the bounding box of a tile

    Parameters
    ----------
    tile : Tile or ints
        May be be either an instance of Tile or 3 ints, X, Y, Z.

    Returns
    -------
    LngLatBBox
    """
    if len(tile) == 1:
        tile = tile[0]
    xtile, ytile, zoom = tile
    a = ul(xtile, ytile, zoom)
    b = ul(xtile + 1, ytile + 1, zoom)
    return LngLatBbox(a[0], b[1], b[0], a[1])


def truncate_lnglat(lng, lat):
    if lng > 180.0:
        lng = 180.0
    elif lng < -180.0:
        lng = -180.0
    if lat > 90.0:
        lat = 90.0
    elif lat < -90.0:
        lat = -90.0
    return lng, lat


def xy(lng, lat, truncate=False):
    """Convert longitude and latitude to web mercator x, y

    Parameters
    ----------
    lng, lat : float
        Longitude and latitude in decimal degrees.
    truncate : bool, optional
        Whether to truncate or clip inputs to web mercator limits.

    Returns
    -------
    x, y : float
    """
    if truncate:
        lng, lat = truncate_lnglat(lng, lat)
    x = 6378137.0 * math.radians(lng)
    y = 6378137.0 * math.log(
        math.tan((math.pi * 0.25) + (0.5 * math.radians(lat))))
    return x, y


def lnglat(x, y, truncate=False):
    """Convert web mercator x, y to longitude and latitude

    Parameters
    ----------
    x, y : float
        web mercator coordinates in meters.
    truncate : bool, optional
        Whether to truncate or clip inputs to web mercator limits.

    Returns
    -------
    LngLat
    """
    R2D = 180 / math.pi
    A = 6378137.0
    lng, lat = (
        x * R2D / A,
        ((math.pi * 0.5) - 2.0 * math.atan(math.exp(-y / A))) * R2D)
    if truncate:
        lng, lat = truncate_lnglat(lng, lat)
    return LngLat(lng, lat)


def xy_bounds(*tile):
    """Get the web mercator bounding box of a tile

    Parameters
    ----------
    tile : Tile or ints
        May be be either an instance of Tile or 3 ints, X, Y, Z.

    Returns
    -------
    Bbox
    """
    if len(tile) == 1:
        tile = tile[0]
    xtile, ytile, zoom = tile
    left, top = xy(*ul(xtile, ytile, zoom))
    right, bottom = xy(*ul(xtile + 1, ytile + 1, zoom))
    return Bbox(left, bottom, right, top)


def tile(lng, lat, zoom, truncate=False):
    """Get the tile containing a longitude and latitude

    Parameters
    ----------
    lng, lat : float
        A longitude and latitude pair in decimal degrees.
    zoom : int
        The web mercator zoom level.
    truncate : bool, optional
        Whether or not to truncate inputs to limits of web mercator.

    Returns
    -------
    Tile
    """
    if truncate:
        lng, lat = truncate_lnglat(lng, lat)
    lat = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int(math.floor((lng + 180.0) / 360.0 * n))

    try:
        ytile = int(math.floor((1.0 - math.log(
            math.tan(lat) + (1.0 / math.cos(lat))) / math.pi) / 2.0 * n))
    except ValueError:
        raise InvalidLatitudeError(
            "Y can not be computed for latitude {} radians".format(lat))
    else:
        return Tile(xtile, ytile, zoom)


def quadkey(*tile):
    """Get the quadkey of a tile

    Parameters
    ----------
    tile: Tile or ints

    Returns
    -------
    str
    """
    if len(tile) == 1:
        tile = tile[0]
    xtile, ytile, zoom = tile
    qk = []
    for z in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (z - 1)
        if xtile & mask:
            digit += 1
        if ytile & mask:
            digit += 2
        qk.append(str(digit))
    return ''.join(qk)


def quadkey_to_tile(qk):
    """Get the tile corresponding to a quadkey

    Parameters
    ----------
    qk : str
        The quadkey

    Returns
    -------
    Tile
    """
    xtile, ytile = 0, 0
    for i, digit in enumerate(reversed(qk)):
        mask = 1 << i
        if digit == '1':
            xtile = xtile | mask
        elif digit == '2':
            ytile = ytile | mask
        elif digit == '3':
            xtile = xtile | mask
            ytile = ytile | mask
        elif digit != '0':
            raise ValueError("Unexpected quadkey digit: %r", digit)
    return Tile(xtile, ytile, i + 1)


def tiles(west, south, east, north, zooms, truncate=False):
    """Get the tiles intersecting a geographic bounding box

    Parameters
    ----------
    west, south, east, north : float
        Bounding values in decimal degrees.
    zooms : int or sequence of ints
        One or more zoom levels.
    truncate : bool, optional
        Whether or not to truncate inputs to web mercator limits.

    Yields
    ------
    Tile
    """
    if truncate:
        west, south = truncate_lnglat(west, south)
        east, north = truncate_lnglat(east, north)
    if west > east:
        bbox_west = (-180.0, south, east, north)
        bbox_east = (west, south, 180.0, north)
        bboxes = [bbox_west, bbox_east]
    else:
        bboxes = [(west, south, east, north)]
    for w, s, e, n in bboxes:

        # Clamp bounding values.
        w = max(-180.0, w)
        s = max(-85.051129, s)
        e = min(180.0, e)
        n = min(85.051129, n)

        for z in zooms:
            ll = tile(w, s, z)
            ur = tile(e, n, z)

            # Clamp left x and top y at 0.
            llx = 0 if ll.x < 0 else ll.x
            ury = 0 if ur.y < 0 else ur.y

            for i in range(llx, min(ur.x + 1, 2 ** z)):
                for j in range(ury, min(ll.y + 1, 2 ** z)):
                    yield Tile(i, j, z)


def parent(*tile):
    """Get the parent of a tile

    The parent is the tile of one zoom level lower that contains the
    given "child" tile.

    Parameters
    ----------
    tile : Tile or ints
        May be be either an instance of Tile or 3 ints, X, Y, Z.

    Returns
    -------
    Tile
    """
    if len(tile) == 1:
        tile = tile[0]
    xtile, ytile, zoom = tile
    # Algorithm ported directly from https://github.com/mapbox/tilebelt.
    if xtile % 2 == 0 and ytile % 2 == 0:
        return Tile(xtile // 2, ytile // 2, zoom - 1)
    elif xtile % 2 == 0:
        return Tile(xtile // 2, (ytile - 1) // 2, zoom - 1)
    elif not xtile % 2 == 0 and ytile % 2 == 0:
        return Tile((xtile - 1) // 2, ytile // 2, zoom - 1)
    else:
        return Tile((xtile - 1) // 2, (ytile - 1) // 2, zoom - 1)


def children(*tile):
    """Get the four children of a tile

    Parameters
    ----------
    tile : Tile or ints
        May be be either an instance of Tile or 3 ints, X, Y, Z.

    Returns
    -------
    list
    """
    if len(tile) == 1:
        tile = tile[0]
    xtile, ytile, zoom = tile
    return [
        Tile(xtile * 2, ytile * 2, zoom + 1),
        Tile(xtile * 2 + 1, ytile * 2, zoom + 1),
        Tile(xtile * 2 + 1, ytile * 2 + 1, zoom + 1),
        Tile(xtile * 2, ytile * 2 + 1, zoom + 1)]


def rshift(val, n):
    return (val % 0x100000000) >> n


def bounding_tile(*bbox, **kwds):
    """Get the smallest tile containing a geographic bounding box

    NB: when the bbox spans lines of lng 0 or lat 0, the bounding tile
    will be Tile(x=0, y=0, z=0).

    Parameters
    ----------
    bbox : floats
        South, west, east, north bounding values in decimal degrees.

    Returns
    -------
    Tile
    """
    if len(bbox) == 2:
        bbox += bbox
    w, s, e, n = bbox
    truncate = bool(kwds.get('truncate'))
    if truncate:
        w, s = truncate_lnglat(w, s)
        e, n = truncate_lnglat(e, n)
    # Algorithm ported directly from https://github.com/mapbox/tilebelt.

    try:
        tmin = tile(w, s, 32, truncate=truncate)
        tmax = tile(e, n, 32, truncate=truncate)
    except InvalidLatitudeError:
        return Tile(0, 0, 0)

    cell = tmin[:2] + tmax[:2]
    z = _getBboxZoom(*cell)
    if z == 0:
        return Tile(0, 0, 0)
    x = rshift(cell[0], (32 - z))
    y = rshift(cell[1], (32 - z))
    return Tile(x, y, z)


def _getBboxZoom(*bbox):
    MAX_ZOOM = 28
    for z in range(0, MAX_ZOOM):
        mask = 1 << (32 - (z + 1))
        if ((bbox[0] & mask) != (bbox[2] & mask) or
                (bbox[1] & mask) != (bbox[3] & mask)):
            return z
    return MAX_ZOOM


def feature(
        tile, fid=None, props=None, projected='geographic', buffer=None,
        precision=None):
    """Get the GeoJSON feature corresponding to a tile

    Parameters
    ----------
    tile : Tile or ints
        May be be either an instance of Tile or 3 ints, X, Y, Z.
    fid : str, optional
        A feature id.
    props : dict, optional
        Optional extra feature properties.
    projected : str, optional
        Non-standard web mercator GeoJSON can be created by passing
        'mercator'.
    buffer : float, optional
        Optional buffer distance for the GeoJSON polygon.
    precision : int, optional
        GeoJSON coordinates will be truncated to this number of decimal
        places.

    Returns
    -------
    dict
    """
    west, south, east, north = bounds(tile)
    if projected == 'mercator':
        west, south = xy(west, south, truncate=False)
        east, north = xy(east, north, truncate=False)
    if buffer:
        west -= buffer
        south -= buffer
        east += buffer
        north += buffer
    if precision and precision >= 0:
        west, south, east, north = (
            round(v, precision) for v in (west, south, east, north))
    bbox = [
        min(west, east), min(south, north),
        max(west, east), max(south, north)]
    geom = {
        'type': 'Polygon',
        'coordinates': [[
            [west, south],
            [west, north],
            [east, north],
            [east, south],
            [west, south]]]}
    xyz = str(tile)
    feat = {
        'type': 'Feature',
        'bbox': bbox,
        'id': xyz,
        'geometry': geom,
        'properties': {'title': 'XYZ tile %s' % xyz}}
    if props:
        feat['properties'].update(props)
    if fid:
        feat['id'] = fid
    return feat
