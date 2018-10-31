"""Test that the following CLI command returns the expected outputs
label-maker labels --dest integration-cl --config test/fixtures/integration/config.geojson.json"""
import unittest
import json
from os import makedirs
from shutil import copyfile, rmtree
import subprocess

import numpy as np

class TestClassificationLabelGeoJSON(unittest.TestCase):
    """Tests for classification label creation"""
    @classmethod
    def setUpClass(cls):
        makedirs('integration-cl')
        copyfile('test/fixtures/integration/labels.geojson', 'integration-cl/labels.geojson')

    @classmethod
    def tearDownClass(cls):
        rmtree('integration-cl')

    def test_cli(self):
        """Verify stdout, geojson, and labels.npz produced by CLI"""
        # our command line output should look like this
        expected_output = """Determining labels for each tile
---
Water Tower: 1 tiles
Building: 1 tiles
Farmland: 0 tiles
Ruins: 1 tiles
Parking: 1 tiles
Roads: 8 tiles
Total tiles: 9
Writing out labels to integration-cl/labels.npz
"""

        cmd = 'label-maker labels --dest integration-cl --config test/fixtures/integration/config.geojson.json'
        cmd = cmd.split(' ')
        with subprocess.Popen(cmd, universal_newlines=True, stdout=subprocess.PIPE) as p:
            self.assertEqual(expected_output, p.stdout.read())

        # our labels should look like this
        expected_labels = {
            '62092-50162-17': np.array([1, 0, 0, 0, 0, 0, 0]),
            '62092-50163-17': np.array([0, 0, 0, 0, 0, 0, 1]),
            '62092-50164-17': np.array([0, 0, 0, 0, 0, 0, 1]),
            '62093-50162-17': np.array([0, 0, 0, 0, 0, 0, 1]),
            '62093-50164-17': np.array([0, 0, 0, 0, 0, 0, 1]),
            '62094-50162-17': np.array([0, 0, 0, 0, 0, 0, 1]),
            '62094-50164-17': np.array([0, 0, 0, 0, 0, 0, 1]),
            '62094-50163-17': np.array([0, 1, 1, 0, 0, 0, 1]),
            '62093-50163-17': np.array([0, 0, 0, 0, 1, 1, 1])
        }

        labels = np.load('integration-cl/labels.npz')
        self.assertEqual(len(labels.files), len(expected_labels.keys()))  # First check number of tiles
        for tile in labels.files:
            self.assertTrue(np.array_equal(expected_labels[tile], labels[tile]))  # Now, content

        # our GeoJSON looks like the fixture
        with open('test/fixtures/integration/classification.geojson') as fixture:
            with open('integration-cl/classification.geojson') as geojson_file:
                expected_geojson = json.load(fixture)
                geojson = json.load(geojson_file)

                for feature in geojson['features']:
                    self.assertTrue(feature in expected_geojson['features'])
