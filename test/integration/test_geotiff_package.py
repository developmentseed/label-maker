"""Test that the following CLI command returns the expected outputs
label-maker package -d integration-tif/sao_tome -c test/fixtures/integration/config.intergration.geotiff_package.json"""
import unittest
from os import makedirs
from shutil import copyfile, rmtree
import subprocess

import numpy as np

class TestObjectDetectionPackage(unittest.TestCase):
    """Tests for local GeoTIFF package creation"""
    @classmethod
    def setUpClass(cls):
        makedirs('integration-tif')
        copyfile('test/fixtures/drone.tif', 'integration-tif/drone.tif')
        copyfile('test/fixtures/integration/labels-tif.npz', 'integration-tif/labels.npz')

    @classmethod
    def tearDownClass(cls):
        rmtree('integration-tif')

    def test_cli(self):
        """Verify data.npz produced by CLI"""
        cmd = 'label-maker images -d integration-tif -c test/fixtures/integration/config.intergration.geotiff_package.json'
        cmd = cmd.split(' ')
        subprocess.run(cmd, universal_newlines=True)

        cmd = 'label-maker package -d integration-tif -c test/fixtures/integration/config.intergration.geotiff_package.json'
        cmd = cmd.split(' ')
        subprocess.run(cmd, universal_newlines=True)

        data = np.load('integration-tif/data.npz')

        self.assertEqual(data['x_train'].shape, (3, 256, 256, 3))
        self.assertEqual(data['x_test'].shape, (1, 256, 256, 3))

        # validate our label data with exact matches in shape
        self.assertEqual(data['y_train'].shape, (3, 3))
        self.assertEqual(data['y_test'].shape, (1, 3))
