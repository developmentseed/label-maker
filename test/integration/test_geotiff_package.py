"""Test that the following CLI command returns the expected outputs
label-maker package -d integration-od/sao_tome -c test/fixtures/integration/config.intergration.geotiff_package.json"""
import unittest
from os import makedirs
from shutil import copytree, rmtree
import subprocess

import numpy as np

class TestObjectDetectionPackage(unittest.TestCase):
    """Tests for local GeoTIFF package creation"""
    @classmethod
    def setUpClass(cls):
        makedirs('integration-od')
        copytree('test/fixtures/integration/geotiff', 'integration-od/geotiff')
        copytree('test/fixtures/integration/sao_tome', 'integration-od/sao_tome')

    @classmethod
    def tearDownClass(cls):
        rmtree('integration-od')

    def test_cli(self):
        """Verify data.npz produced by CLI"""
        cmd = 'label-maker package -d integration-od/sao_tome -c test/fixtures/integration/config.intergration.geotiff_package.json'
        cmd = cmd.split(' ')
        subprocess.run(cmd, universal_newlines=True)

        data = np.load('integration-od/sao_tome/data.npz')

        self.assertEqual(data['x_train'].shape, (332, 256, 256, 3))
        self.assertEqual(data['x_test'].shape, (83, 256, 256, 3))

        # validate our label data with exact matches in shape
        self.assertEqual(data['y_train'].shape, (332, 3))
        self.assertEqual(data['y_test'].shape, (83, 3))
