"""Provide utility functions"""
import numpy as np

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
