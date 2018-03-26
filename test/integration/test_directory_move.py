"""Test that the following CLI command returns the expected outputs outside the current directory
label-maker labels --dest integration-cl --config test/fixtures/integration/config.integration.json"""
import unittest
import json
from os import makedirs, chdir
from shutil import copyfile, rmtree
import subprocess

import numpy as np

class TestOutsideDirectory(unittest.TestCase):
    """Tests for classification label creation outside of the current directory"""
    @classmethod
    def setUpClass(cls):
        makedirs('integration-cl')
        copyfile('test/fixtures/integration/portugal-z17.mbtiles', 'integration-cl/portugal-z17.mbtiles')


    @classmethod
    def tearDownClass(cls):
        rmtree('integration-cl')

    def test_cli(self):
        """Verify geojson and labels.npz produced by CLI"""

        # first move outside the directory
        chdir('..')
        directory = 'label-maker'

        cmd = 'label-maker labels --dest {}/integration-cl --config {}/test/fixtures/integration/config.integration.json'.format(directory, directory)
        cmd = cmd.split(' ')
        subprocess.run(cmd, universal_newlines=True)

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

        # move back into the directory
        chdir(directory)

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
