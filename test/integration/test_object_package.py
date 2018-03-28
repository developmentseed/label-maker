"""Test that the following CLI command returns the expected outputs
label-maker package -d integration-od -c test/fixtures/integration/config.integration.object_detection.json"""
import unittest
from os import makedirs
from shutil import copyfile, copytree, rmtree
import subprocess

import numpy as np

class TestObjectDetectionPackage(unittest.TestCase):
    """Tests for object detection package creation"""
    @classmethod
    def setUpClass(cls):
        makedirs('integration-od')
        copyfile('test/fixtures/integration/labels-od.npz', 'integration-od/labels.npz')
        copytree('test/fixtures/integration/tiles', 'integration-od/tiles')

    @classmethod
    def tearDownClass(cls):
        rmtree('integration-od')

    def test_cli(self):
        """Verify data.npz produced by CLI"""
        cmd = 'label-maker package -d integration-od -c test/fixtures/integration/config.integration.object_detection.json'
        cmd = cmd.split(' ')
        subprocess.run(cmd, universal_newlines=True)

        data = np.load('integration-od/data.npz')

        self.assertEqual(np.sum(data['x_train']), 144752757)
        self.assertEqual(np.sum(data['x_test']), 52758414)
        self.assertEqual(data['x_train'].shape, (6, 256, 256, 3))
        self.assertEqual(data['x_test'].shape, (2, 256, 256, 3))

        # validate our label data with exact matches in shape
        self.assertEqual(data['y_train'].shape, (6, 16, 5))
        self.assertEqual(data['y_test'].shape, (2, 16, 5))
