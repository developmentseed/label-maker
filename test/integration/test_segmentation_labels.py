"""Test that the following CLI command returns the expected outputs
label-maker labels --dest integration-sg --config test/fixtures/integration/config.integration.segmentation.json"""
import unittest
from os import makedirs
from shutil import copyfile, rmtree
import subprocess

import numpy as np

class TestSegmentationLabel(unittest.TestCase):
    """Tests for segmentation label creation"""
    @classmethod
    def setUpClass(cls):
        makedirs('integration-sg')
        copyfile('test/fixtures/integration/portugal-z10.mbtiles', 'integration-sg/portugal-z10.mbtiles')
        copyfile('test/fixtures/integration/spain-z10.mbtiles', 'integration-sg/spain-z10.mbtiles')

    @classmethod
    def tearDownClass(cls):
        rmtree('integration-sg')

    def test_cli(self):
        """Verify labels.npz produced by CLI"""
        cmd = 'label-maker labels --dest integration-sg --config test/fixtures/integration/config.integration.segmentation.json'
        cmd = cmd.split(' ')
        subprocess.run(cmd, universal_newlines=True)

        # our labels should look like this
        expected_sums = {
            '492-394-10': 10,
            '490-396-10': 1,
            '491-394-10': 1,
            '490-394-10': 2,
            '490-395-10': 1,
            '489-396-10': 1,
            '488-395-10': 10,
            '488-396-10': 3,
            '488-397-10': 4,
            '489-397-10': 11,
            '488-394-10': 2,
            '490-397-10': 5,
            '489-394-10': 1,
            '492-395-10': 2,
            '492-396-10': 6,
            '492-397-10': 9,
            '493-394-10': 4,
            '493-395-10': 2,
            '489-395-10': 0,
            '491-395-10': 0,
            '491-396-10': 0,
            '491-397-10': 0,
            '493-396-10': 0,
            '493-397-10': 0

        }

        labels = np.load('integration-sg/labels.npz')
        self.assertEqual(len(labels.files), len(expected_sums.keys()))  # First check number of tiles
        for tile in labels.files:
            self.assertEqual(expected_sums[tile], np.sum(labels[tile]))  # Now, sums
