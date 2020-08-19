"""Test that the following CLI command returns the expected outputs
label-maker package -d integration-tif/sao_tome -c test/fixtures/integration/config.intergration.geotiff_package.json"""
import unittest
from os import makedirs
from shutil import copyfile, copytree, rmtree
import subprocess
import os

import numpy as np

class TestClassificationPackage(unittest.TestCase):
    """Tests for local GeoTIFF package creation"""
    @classmethod
    def setUpClass(cls):
        makedirs('integration-ms')
        copyfile('test/fixtures/integration/ms_img.tif', 'integration-ms/ms-img.tif')
        copyfile('test/fixtures/integration/roads_ms.geojson', 'integration-ms/ms-roads.geojson')
        copyfile('test/fixtures/integration/labels-ms.npz', 'integration-ms/labels.npz')

    @classmethod
    def tearDownClass(cls):
        rmtree('integration-ms')

    def test_cli(self):
        """Verify data.npz produced by CLI"""
        cmd = 'label-maker images --dest integration-ms --config test/fixtures/integration/config.integration.bands.json'
        cmd = cmd.split(' ')
        subprocess.run(cmd, universal_newlines=True)

        cmd = 'label-maker package --dest integration-ms --config test/fixtures/integration/config.integration.bands.json'
        cmd = cmd.split(' ')
        subprocess.run(cmd, universal_newlines=True)

        print(os.listdir('integration-ms/'))
        data = np.load('integration-ms/data.npz')

        # validate our label data with exact matches in shape
        self.assertEqual(data['x_train'].shape, (8, 256, 256, 2))
        self.assertEqual(data['x_test'].shape, (3, 256, 256, 2))

        # validate our label data with exact matches in shape
        self.assertEqual(data['y_train'].shape, (8, 3))
        self.assertEqual(data['y_test'].shape, (3, 3))

        #validate img dtype
        self.assertEqual(np.uint16, data['x_train'].dtype)