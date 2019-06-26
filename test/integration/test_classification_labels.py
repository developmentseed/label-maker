"""Test that the following CLI command returns the expected outputs
label-maker labels --dest integration-cl --config test/fixtures/integration/config.integration.json"""
import unittest
import json
from os import makedirs
from shutil import copyfile, rmtree
import subprocess

import numpy as np

class TestClassificationLabel(unittest.TestCase):
    """Tests for classification label creation"""
    @classmethod
    def setUpClass(cls):
        makedirs('integration-cl')
        copyfile('test/fixtures/integration/portugal-z10.mbtiles', 'integration-cl/portugal-z10.mbtiles')
        copyfile('test/fixtures/integration/spain-z10.mbtiles', 'integration-cl/spain-z10.mbtiles')

    @classmethod
    def tearDownClass(cls):
        rmtree('integration-cl')

    def test_cli(self):
        """Verify stdout, geojson, and labels.npz produced by CLI"""
        # our command line output should look like this
        expected_output = """Determining labels for each tile
---
Water Tower: 0 tiles
Building: 0 tiles
Farmland: 11 tiles
Ruins: 0 tiles
Parking: 2 tiles
Roads: 0 tiles
Total tiles: 24
Writing out labels to integration-cl/labels.npz
Determining labels for each tile
---
Water Tower: 0 tiles
Building: 1 tiles
Farmland: 11 tiles
Ruins: 0 tiles
Parking: 5 tiles
Roads: 0 tiles
Total tiles: 24
Writing out labels to integration-cl/labels.npz
"""

        cmd = 'label-maker labels --dest integration-cl --config test/fixtures/integration/config.integration.json'
        cmd = cmd.split(' ')
        with subprocess.Popen(cmd, universal_newlines=True, stdout=subprocess.PIPE) as p:
            self.assertEqual(expected_output, p.stdout.read())

        # our labels should look like this
        expected_labels = {
            '491-396-10': np.array([0, 0, 0, 0, 0, 0, 0]),
            '491-397-10': np.array([0, 0, 0, 0, 0, 0, 0]),
            '492-395-10': np.array([0, 0, 0, 0, 0, 0, 0]),
            '491-395-10': np.array([0, 0, 0, 0, 0, 0, 0]),
            '492-394-10': np.array([0, 0, 0, 0, 0, 0, 0]),
            '491-394-10': np.array([0, 0, 0, 1.5, 0, 0, 0]),
            '488-395-10': np.array([0, 0, 0, 12, 0, 0, 0]),
            '489-396-10': np.array([0, 0, 0, 1.5, 0, 0, 0]),
            '489-397-10': np.array([0, 0, 0, 0, 0, 0, 0]),
            '490-395-10': np.array([0, 0, 0, 0, 0, 0, 0]),
            '490-396-10': np.array([0, 0, 0, 0, 0, 0, 0]),
            '490-394-10': np.array([0, 0, 0, 6, 0, 0, 0]),
            '488-396-10': np.array([0, 0, 0, 11, 0, 0, 0]),
            '488-394-10': np.array([0, 0, 0, 1, 0, 0.5, 0]),
            '488-397-10': np.array([0, 0, 0, 7, 0, 0, 0]),
            '489-395-10': np.array([0, 0, 0, 2.5, 0, 0, 0]),
            '489-394-10': np.array([0, 0, 0, 0, 0, 1, 0]),
            '490-397-10': np.array([0, 0, 0, 1, 0, 0, 0]),
            '492-396-10': np.array([0, 0, 0, 0, 0, 0, 0]),
            '492-397-10': np.array([0, 0, 1.5, 56, 0, 0.5, 0]),
            '493-394-10': np.array([0, 0, 0, 0, 0, 0, 0]),
            '493-395-10': np.array([0, 0, 0, 0, 0, 0, 0]),
            '493-396-10': np.array([0, 0, 0, 17.5, 0, 0.5, 0]),
            '493-397-10': np.array([0, 0, 0, 0, 0, 1, 0])
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

                self.assertCountEqual(expected_geojson, geojson)
