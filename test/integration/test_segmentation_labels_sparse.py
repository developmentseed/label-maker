"""Test that the following CLI command returns the expected outputs
label-maker labels --dest integration-sg --config test/fixtures/integration/config.integration.segmentation.json --sparse"""
import unittest
from os import makedirs
from shutil import copyfile, rmtree
import subprocess

import numpy as np

class TestSegmentationLabelSparse(unittest.TestCase):
    """Tests for segmentation label creation (sparse)"""
    @classmethod
    def setUpClass(cls):
        makedirs('integration-sg')
        copyfile('test/fixtures/integration/portugal-z17.mbtiles', 'integration-sg/portugal-z17.mbtiles')

    @classmethod
    def tearDownClass(cls):
        rmtree('integration-sg')

    def test_cli(self):
        """Verify labels.npz produced by CLI"""
        cmd = 'label-maker labels --dest integration-sg --config test/fixtures/integration/config.integration.segmentation_sparse.json --sparse'
        cmd = cmd.split(' ')
        subprocess.run(cmd, universal_newlines=True)

        # our labels should look like this
        expected_sums = {
            '62092-50164-17': 13500,
            '62093-50162-17': 36822,
            '62093-50164-17': 2400,
            '62094-50162-17': 21234,
            '62094-50164-17': 19146,
            '62094-50163-17': 21613,
            '62093-50163-17': 31568
        }

        labels = np.load('integration-sg/labels.npz')
        self.assertEqual(len(labels.files), len(expected_sums.keys()))  # First check number of tiles
        for tile in labels.files:
            self.assertEqual(expected_sums[tile], np.sum(labels[tile]))  # Now, sums
