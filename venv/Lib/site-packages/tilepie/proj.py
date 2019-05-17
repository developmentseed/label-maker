## Adapted from https://github.com/makinacorpus/landez
## GNU LESSER GENERAL PUBLIC LICENSE Mathieu Leplatre <mathieu.leplatre@makina-corpus.com>

from math import pi, sin, log, exp, atan, tan, ceil
from gettext import gettext as _

DEFAULT_TILE_SIZE = 256
DEG_TO_RAD = pi/180
RAD_TO_DEG = 180/pi
MAX_LATITUDE = 85.0511287798
EARTH_RADIUS = 6378137

def minmax (a,b,c):
    a = max(a,b)
    a = min(a,c)
    return a

class InvalidCoverageError(Exception):
    """ Raised when coverage bounds are invalid """
    pass

class GoogleProjection(object):

    NAME = 'EPSG:3857'

    """
    Transform Lon/Lat to Pixel within tiles
    Originally written by OSM team : http://svn.openstreetmap.org/applications/rendering/mapnik/generate_tiles.py
    """
    def __init__(self, tilesize=DEFAULT_TILE_SIZE, levels = [0], scheme='wmts'):
        if not levels:
            raise InvalidCoverageError(_("Wrong zoom levels."))
        self.Bc = []
        self.Cc = []
        self.zc = []
        self.Ac = []
        self.levels = levels
        self.maxlevel = max(levels) + 1
        self.tilesize = tilesize
        self.scheme = scheme
        c = tilesize
        for d in range(self.maxlevel):
            e = c/2;
            self.Bc.append(c/360.0)
            self.Cc.append(c/(2 * pi))
            self.zc.append((e,e))
            self.Ac.append(c)
            c *= 2

    def project_pixels(self,ll,zoom):
        d = self.zc[zoom]
        e = round(d[0] + ll[0] * self.Bc[zoom])
        f = minmax(sin(DEG_TO_RAD * ll[1]),-0.9999,0.9999)
        g = round(d[1] + 0.5*log((1+f)/(1-f))*-self.Cc[zoom])
        return (e,g)

    def unproject_pixels(self,px,zoom):
        e = self.zc[zoom]
        f = (px[0] - e[0])/self.Bc[zoom]
        g = (px[1] - e[1])/-self.Cc[zoom]
        h = RAD_TO_DEG * ( 2 * atan(exp(g)) - 0.5 * pi)
        if self.scheme == 'tms':
            h = - h
        return (f,h)

    def tile_at(self, zoom, position):
        """
        Returns a tuple of (z, x, y)
        """
        x, y = self.project_pixels(position, zoom)
        return (zoom, int(x/self.tilesize), int(y/self.tilesize))

    def tile_bbox(self, tile_indices):
        """
        Returns the WGS84 bbox of the specified tile
        """
        (z, x, y) = tile_indices
        topleft = (x * self.tilesize, (y + 1) * self.tilesize)
        bottomright = ((x + 1) * self.tilesize, y * self.tilesize)
        nw = self.unproject_pixels(topleft, z)
        se = self.unproject_pixels(bottomright, z)
        return nw + se

    def project(self, coords):
        """
        Returns the coordinates in meters from WGS84
        """
        (lng, lat) = coords
        x = lng * DEG_TO_RAD
        lat = max(min(MAX_LATITUDE, lat), -MAX_LATITUDE)
        y = lat * DEG_TO_RAD
        y = log(tan((pi / 4) + (y / 2)))
        return (x*EARTH_RADIUS, y*EARTH_RADIUS)

    def unproject(self, xy):
        """
        Returns the coordinates from position in meters
        """
        (x, y) = xy
        lng = x/EARTH_RADIUS * RAD_TO_DEG
        lat = 2 * atan(exp(y/EARTH_RADIUS)) - pi/2 * RAD_TO_DEG
        return (lng, lat)

    def tileslist(self, bbox):
        if len(bbox) != 4:
            raise InvalidCoverageError(_("Wrong format of bounding box."))
        xmin, ymin, xmax, ymax = bbox
        if abs(xmin) > 180 or abs(xmax) > 180 or \
           abs(ymin) > 90 or abs(ymax) > 90:
            raise InvalidCoverageError(_("Some coordinates exceed [-180,+180], [-90, 90]."))

        if xmin >= xmax or ymin >= ymax:
            raise InvalidCoverageError(_("Bounding box format is (xmin, ymin, xmax, ymax)"))

        ll0 = (xmin, ymax)  # left top
        ll1 = (xmax, ymin)  # right bottom

        l = []
        for z in self.levels:
            px0 = self.project_pixels(ll0,z)
            px1 = self.project_pixels(ll1,z)

            for x in range(int(px0[0]/self.tilesize),
                           int(ceil(px1[0]/self.tilesize))):
                if (x < 0) or (x >= 2**z):
                    continue
                for y in range(int(px0[1]/self.tilesize),
                               int(ceil(px1[1]/self.tilesize))):
                    if (y < 0) or (y >= 2**z):
                        continue
                    if self.scheme == 'tms':
                        y = ((2**z-1) - y)
                    l.append((z, x, y))
        return l