import pickle

import geojson
import mercantile
import shapely

from supermercado import burntiles
from label_maker.label import *
from label_maker.images import *

def make_dataset(dest_folder, classes, imagery, ml_type, sparse, threadcount, seed=False,
                      split_names=('train', 'test'), split_vals=(0.8, .2),
                      **kwargs):


    # identify all candidates tiles (project bounds or input geojson bounds + zoom)

    # get labels for all tiles (optionally write out xarray pkl with tile name + label, need to add flag for this)

    # get images (optionally write out tiles pngs/jpegs, need to add flag for this)

    # match images and labels, #shuffle order

    #put into df, convert to x-array

    # write out final pkl x-array (index= x-y-z, columns= label, img, split-have split be optional?)
    with open(op.join(dest_folder, 'data.pickle'), 'wb') as f:
        pickle.dump(ds, f, protocol=-1)