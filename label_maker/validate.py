"""Expose a schema for use with cerberus for config validation"""
import os.path as op
import label_maker

countries = []
module_dir = op.dirname(label_maker.__file__)  # Get module home directory

with open(op.join(module_dir, 'countries.txt')) as f:
    lines = f.readlines()
    for line in lines:
        countries.append(line.strip())

class_schema = {
    'type': 'dict',
    'schema': {'name': {'type': 'string', 'required': True},
               'filter': {'type': 'list', 'required': True},
               'buffer': {'type': 'float'}}
}

lat_schema = {'type': 'float', 'min': -90, 'max': 90}
lon_schema = {'type': 'float', 'min': -180, 'max': 180}

schema = {
    'country': {'type': 'string', 'allowed': countries, 'required': True},
    'bounding_box': {'type': 'list', 'items': [lon_schema, lat_schema, lon_schema, lat_schema], 'required': True},
    'zoom': {'type': 'integer', 'required': True},
    'classes': {'type': 'list', 'schema': class_schema, 'required': True},
    'imagery': {'type': 'string', 'required': True},
    'background_ratio': {'type': 'float'},
    'ml_type': {'allowed': ['classification', 'object-detection', 'segmentation'], 'required': True},
    'seed': {'type': 'integer'}
}
