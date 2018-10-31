"""Tests for validate.py"""
import unittest
import json
import copy

from cerberus import Validator
from label_maker.validate import schema

v = Validator(schema)

class TestValidate(unittest.TestCase):
    """Tests for configuration validation"""
    def test_passing(self):
        """Test a good configuration"""
        with open('test/fixtures/validation/passing.json') as config_file:
            config = json.load(config_file)
            valid = v.validate(config)
            self.assertTrue(valid)

    def test_required(self):
        """Test for all required keys"""
        with open('test/fixtures/validation/passing.json') as config_file:
            config = json.load(config_file)
            for key in ['zoom', 'classes', 'imagery', 'ml_type']:
                bad_config = copy.deepcopy(config)
                bad_config.pop(key)
                valid = v.validate(bad_config)
                self.assertFalse(valid)

    def test_geojson(self):
        """Test an alternate configuration with geojson input"""
        with open('test/fixtures/validation/geojson.json') as config_file:
            config = json.load(config_file)
            valid = v.validate(config)
            self.assertTrue(valid)

    def test_country(self):
        """Test country not in list fails"""
        with open('test/fixtures/validation/passing.json') as config_file:
            config = json.load(config_file)
            config['country'] = 'not_a_real_place'
            valid = v.validate(config)
            self.assertFalse(valid)

    def test_class(self):
        """Test that bad class construction fails"""
        with open('test/fixtures/validation/passing.json') as config_file:
            config = json.load(config_file)
            config['classes'] = [dict(filter=list())]
            valid = v.validate(config)
            self.assertFalse(valid)
            config['classes'] = [dict(name='')]
            valid = v.validate(config)
            self.assertFalse(valid)
            config['classes'] = [dict(name='', filter='')]
            valid = v.validate(config)
            self.assertFalse(valid)
            config['classes'] = [dict(name=5, filter=list())]
            valid = v.validate(config)
            self.assertFalse(valid)

    def test_bounds(self):
        """Test that bad bounds fail"""
        with open('test/fixtures/validation/passing.json') as config_file:
            config = json.load(config_file)
            config['bounding_box'] = [-181, 0, 0, 0]
            valid = v.validate(config)
            self.assertFalse(valid)
            config['bounding_box'] = [0, -91, 0, 0]
            valid = v.validate(config)
            self.assertFalse(valid)
            config['bounding_box'] = [0, 0, 181, 0]
            valid = v.validate(config)
            self.assertFalse(valid)
            config['bounding_box'] = [0, 0, 0, 91]
            valid = v.validate(config)
            self.assertFalse(valid)


if __name__ == '__main__':
    unittest.main()
