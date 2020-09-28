import geojson
import mercantile
import shapely

from supermercado import burntiles
from label_maker.label import *


def id_tiles(geojson_path, **kwargs):
    """
    returns list of x,y,z tiles that comprise the project bounds geojson
    """
    with open('/Users/marthamorrissey/Downloads/map.geojson') as f:
        gj = geojson.load(f)
        features = gj['features'][0]
    tiles = burntiles.burn([features], kwargs.get('zoom'))
    tiles = [mercantile.Tile(*tile) for tile in tiles]

    #subset the tiles based on background ratio

def label_tile_match_gj(possible_tiles, labels_geojson):
    #TO-DO add sparse, add backgroundratio
    match_dict = {}
    tiles_geom =  [shapely.geometry.box(*mercantile.bounds(t)) for t in possible_tiles]

    for tile_geom in tiles_geom:
        match = [geom for geom in shape_lst if tile_geom.intersects(geom)]
        if match:
            match_dict.update({tile_geom:match})



def make_dataset(dest_folder, classes, imagery, ml_type, sparse, threadcount, seed=False,
                      split_names=('train', 'test'), split_vals=(0.8, .2),
                      **kwargs):


    # identify all candidates tiles (project bounds or input geojson bounds + zoom)

    # get labels for all tiles (optionally write out xarray pkl with tile name + label, need to add flag for this)

    # get images (optionally write out tiles pngs/jpegs, need to add flag for this)

    # match images and labels

    # write out final pkl x-array (index= x-y-z, columns= label, img, split)