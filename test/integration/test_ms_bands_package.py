"""Test that the following CLI command returns the expected outputs
label-maker package -d integration-tif/sao_tome -c test/fixtures/integration/config.intergration.geotiff_package.json"""
import unittest
from os import makedirs
from shutil import copyfile, rmtree
import subprocess

import numpy as np

class TestClassificationPackage(unittest.TestCase):
    """Tests for local GeoTIFF package creation"""
    @classmethod
    def setUpClass(cls):
        makedirs('integration')
        copyfile('test/fixtures/ms-img.tif', 'integration/ms-img.tif')
        copyfile('test/fixtures/integration/labels-ms.npz', 'integration/labels.npz')

    @classmethod
    def tearDownClass(cls):
        rmtree('integration')

    def test_cli(self):
        """Verify data.npz produced by CLI"""
        cmd = 'label-maker images -d integration -c test/fixtures/integration/config.intergration.bands.json'
        cmd = cmd.split(' ')
        subprocess.run(cmd, universal_newlines=True)

        cmd = 'label-maker package -d integration -c test/fixtures/integration/config.intergration.bands.json'
        cmd = cmd.split(' ')
        subprocess.run(cmd, universal_newlines=True)

        data = np.load('integration/data.npz')

`       # validate our label data with exact matches in shape`
        self.assertEqual(data['x_train'].shape, (12, 256, 256, 2))
        self.assertEqual(data['x_test'].shape, (4, 256, 256, 2))

        # validate our label data with exact matches in shape
        self.assertEqual(data['y_train'].shape, (4, 3))
        self.assertEqual(data['y_test'].shape, (12, 3))

        #validate img dtype
        self.assertEqual(np.uint16, data['y_train'].dtype)