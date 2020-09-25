"""Expose a schema for use with cerberus for config validation"""
import os.path as op
import label_maker

module_dir = op.dirname(label_maker.__file__)  # Get module home directory

class_schema = {
    'type': 'dict',
    'schema': {'name': {'type': 'string', 'required': True},
               'filter': {'type': 'list', 'required': True},
               'buffer': {'type': 'float'}}
}

lat_schema = {'type': 'float', 'min': -90, 'max': 90}
lon_schema = {'type': 'float', 'min': -180, 'max': 180}

schema = {
    'geojson': {'type': 'string'},
    'project_bounds_geojson': {'type': 'string'},
    'zoom': {'type': 'integer', 'required': True},
    'classes': {'type': 'list', 'schema': class_schema, 'required': True},
    'imagery': {'type': 'string', 'required': True},
    'http_auth': {'type': 'list', 'schema': {'type': 'string'}},
    'background_ratio': {'type': 'float'},
    'ml_type': {'allowed': ['classification', 'object-detection', 'segmentation'], 'required': True},
    'seed': {'type': 'integer'},
    'imagery_offset': {'type': 'list', 'schema': {'type': 'integer'}, 'minlength': 2, 'maxlength': 2},
    'split_vals': {'type': 'list', 'schema': {'type': 'float'}},
    'split_names': {'type': 'list', 'schema': {'type': 'string'}},
    'tms_image_format': {'type': 'string'},
    'over_zoom': {'type': 'integer', 'min': 1},
    'band_indices': {'type': 'list'}
}
