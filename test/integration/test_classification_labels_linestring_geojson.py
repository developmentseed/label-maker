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
        copyfile('test/fixtures/integration/labels-linestring.geojson', 'integration-cl/labels.geojson')

    @classmethod
    def tearDownClass(cls):
        rmtree('integration-cl')

    def test_cli(self):
        """Verify stdout, geojson, and labels.npz produced by CLI"""
        # our command line output should look like this
        expected_output = """Determining labels for each tile
---
tertiary: 9 tiles
motorway: 0 tiles
primary: 0 tiles
secondary: 3 tiles
residential: 5 tiles
unclassified: 0 tiles
Total tiles: 12
Writing out labels to integration-cl/labels.npz
"""

        cmd = 'label-maker labels --dest integration-cl --config test/fixtures/integration/config-linestring.geojson.json'
        cmd = cmd.split(' ')
        with subprocess.Popen(cmd, universal_newlines=True, stdout=subprocess.PIPE) as p:
            self.assertEqual(expected_output, p.stdout.read())

        # our labels should look like this

        expected_labels = {
        '184101-116932-18': np.array([0, 1, 0, 0, 0, 0, 0]),
        '184101-116931-18': np.array([0, 1, 0, 0, 0, 0, 0]),
        '184101-116934-18': np.array([0, 1, 0, 0, 0, 1, 0]),
        '184103-116932-18': np.array([0, 0, 0, 0, 0, 1, 0]),
        '184103-116933-18': np.array([0, 0, 0, 0, 1, 1, 0]),
        '184102-116932-18': np.array([0, 1, 0, 0, 0, 1, 0]),
        '184102-116931-18': np.array([0, 1, 0, 0, 0, 0, 0]),
        '184101-116933-18': np.array([0, 1, 0, 0, 1, 1, 0]),
        '184102-116934-18': np.array([0, 1, 0, 0, 0, 0, 0]),
        '184103-116934-18': np.array([0, 1, 0, 0, 0, 0, 0]),
        '184102-116933-18': np.array([0, 1, 0, 0, 1, 0, 0]),
        '184103-116931-18': np.array([1, 0, 0, 0, 0, 0, 0])
        }

        labels = np.load('integration-cl/labels.npz')
        self.assertEqual(len(labels.files), len(expected_labels.keys()))  # First check number of tiles
        for tile in labels.files:
            self.assertTrue(np.array_equal(expected_labels[tile], labels[tile]))  # Now, content

        # our GeoJSON looks like the fixture
        with open('test/fixtures/integration/classification_linestring.geojson') as fixture:
            with open('integration-cl/classification.geojson') as geojson_file:
                expected_geojson = json.load(fixture)
                geojson = json.load(geojson_file)

                self.assertCountEqual(expected_geojson, geojson)
