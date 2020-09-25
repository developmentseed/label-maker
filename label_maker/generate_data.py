

from label_maker.label import *

def make_dataset(dest_folder, classes, imagery, ml_type, sparse, seed=False,
                      split_names=('train', 'test'), split_vals=(0.8, .2),
                      **kwargs):

    # get labels (optionally write out xarray pkl with tile name + label)

    # get images (optionally write out tiles pngs/jpegs)

    # match images and labels

    # write out final pkl xarray