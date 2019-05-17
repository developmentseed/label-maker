## Adapted from https://github.com/makinacorpus/landez
## GNU LESSER GENERAL PUBLIC LICENSE Mathieu Leplatre <mathieu.leplatre@makina-corpus.com>

DEFAULT_TILE_SIZE = 256
DEFAULT_TILE_SCHEME = 'wmts'

import os
import sqlite3
import logging
from tilepie.proj import GoogleProjection

logger = logging.getLogger(__name__)

class ExtractionError(Exception):
    """ Raised when extraction of tiles from specified MBTiles has failed """
    pass

class InvalidFormatError(Exception):
    """ Raised when reading of MBTiles content has failed """
pass

def flip_y(y, z):
  return 2 ** z - 1 - y

class MBTilesReader():
  def __init__(self, filename, tilesize=None):
      self.basename = ''
      self.filename = filename
      self.basename = os.path.basename(self.filename)
      self._con = None
      self._cur = None

  def _query(self, sql, *args):
      """ Executes the specified `sql` query and returns the cursor """
      if not self._con:
          logger.debug(("Open MBTiles file '%s'") % self.filename)
          self._con = sqlite3.connect(self.filename)
          self._cur = self._con.cursor()
      sql = ' '.join(sql.split())
      logger.debug(("Execute query '%s' %s") % (sql, args))
      try:
          self._cur.execute(sql, *args)
      except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
          raise InvalidFormatError(("%s while reading %s") % (e, self.filename))
      return self._cur

  def metadata(self):
      rows = self._query('SELECT name, value FROM metadata')
      rows = [(row[0], row[1]) for row in rows]
      return dict(rows)

  def zoomlevels(self):
      rows = self._query('SELECT DISTINCT(zoom_level) FROM tiles ORDER BY zoom_level')
      return [int(row[0]) for row in rows]

  def tile(self, z, x, y):
      logger.debug(("Extract tile %s") % ((z, x, y),))
      tms_y = flip_y(int(y), int(z))
      rows = self._query('''SELECT tile_data FROM tiles
                            WHERE zoom_level=? AND tile_column=? AND tile_row=?;''', (z, x, tms_y))
      t = rows.fetchone()
      if not t:
          raise ExtractionError(("Could not extract tile %s from %s") % ((z, x, y), self.filename))
      return t[0]

  def tileslist(self, bbox, zoomlevels):
    proj = GoogleProjection(DEFAULT_TILE_SIZE, zoomlevels, DEFAULT_TILE_SCHEME)
    return proj.tileslist(bbox)