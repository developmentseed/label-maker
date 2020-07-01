"""Test that the following CLI command returns the expected outputs
label-maker package --dest integration-cl --config test/fixtures/integration/config.integration.json"""
import unittest
from os import makedirs
from shutil import copyfile, copytree, rmtree
import subprocess

import numpy as np


class TestClassificationPackage(unittest.TestCase):
    """Tests for classification package creation"""

    @classmethod
    def setUpClass(cls):
        makedirs('integration-cl')
        copyfile('test/fixtures/integration/labels-cl.npz', 'integration-cl/labels.npz')
        copytree('test/fixtures/integration/tiles', 'integration-cl/tiles')

        makedirs('integration-cl-split')
        copyfile('test/fixtures/integration/labels-cl.npz', 'integration-cl-split/labels.npz')
        copytree('test/fixtures/integration/tiles', 'integration-cl-split/tiles')


        makedirs('integration-cl-overzoom')
        copyfile('test/fixtures/integration/labels-cl.npz', 'integration-cl-overzoom/labels.npz')

        makedirs('integration-cl-img-f')
        copyfile('test/fixtures/integration/labels-cl-img-f.npz', 'integration-cl-img-f/labels.npz')
        copytree('test/fixtures/integration/tiles_png', 'integration-cl-img-f/tiles')

    @classmethod
    def tearDownClass(cls):
        rmtree('integration-cl')
        rmtree('integration-cl-split')
        rmtree('integration-cl-overzoom')
        rmtree('integration-cl-img-f')

    def test_cli(self):
        """Verify data.npz produced by CLI"""
        cmd = 'label-maker package --dest integration-cl --config test/fixtures/integration/config.integration.json'
        cmd = cmd.split(' ')
        subprocess.run(cmd, universal_newlines=True)

        data = np.load('integration-cl/data.npz')
        # validate our image data with sums and shapes
        self.assertEqual(np.sum(data['x_train']), 144752757)
        self.assertEqual(np.sum(data['x_test']), 52758414)
        self.assertEqual(data['x_train'].shape, (6, 256, 256, 3))
        self.assertEqual(data['x_test'].shape, (2, 256, 256, 3))

        # validate our label data with exact matches
        expected_y_train = np.array(
            [[0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 1],
             [0, 1, 1, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 1]]
        )
        self.assertTrue(np.array_equal(data['y_train'], expected_y_train))

        expected_y_test = np.array(
            [[0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 1]]
        )
        self.assertTrue(np.array_equal(data['y_test'], expected_y_test))

    def test_cli_3way_split(self):
        """Verify data.npz produced by CLI when split into train/test/val"""

        cmd = 'label-maker package --dest integration-cl-split --config test/fixtures/integration/config_3way.integration.json'
        cmd = cmd.split(' ')
        subprocess.run(cmd, universal_newlines=True)

        data = np.load('integration-cl-split/data.npz')

        # validate our image data with shapes
        self.assertEqual(data['x_train'].shape, (5, 256, 256, 3))
        self.assertEqual(data['x_test'].shape, (2, 256, 256, 3))
        self.assertEqual(data['x_val'].shape, (1, 256, 256, 3))

        # validate label data with shapes
        self.assertEqual(data['y_train'].shape, (5, 7))
        self.assertEqual(data['y_test'].shape, (2, 7))
        self.assertEqual(data['y_val'].shape, (1, 7))

    def test_overzoom(self):
        """Verify data.npz produced by CLI when overzoom is used"""
        cmd = 'label-maker images --dest integration-cl-overzoom --config test/fixtures/integration/config_overzoom.integration.json'
        cmd = cmd.split(' ')
        subprocess.run(cmd, universal_newlines=True)

        cmd = 'label-maker package --dest integration-cl-overzoom --config test/fixtures/integration/config_overzoom.integration.json'
        cmd = cmd.split(' ')
        subprocess.run(cmd, universal_newlines=True)

        data = np.load('integration-cl-overzoom/data.npz')

        self.assertEqual(data['x_train'].shape, (6, 512, 512, 3))
        self.assertEqual(data['x_test'].shape, (2, 512, 512, 3))
        self.assertEqual(data['x_val'].shape, (1, 512, 512, 3))

    def test_tms_img_format(self):
        """Verify data.npz produced by CLI"""

        cmd = 'label-maker package --dest integration-cl-img-f --config test/fixtures/integration/config_tms_format_img.json'
        cmd = cmd.split(' ')
        subprocess.run(cmd, universal_newlines=True)

        data = np.load('integration-cl-img-f/data.npz')

        # validate our image data with shapes
        self.assertEqual(data['x_train'].shape, (9, 256, 256, 3))
        self.assertEqual(data['x_test'].shape, (3, 256, 256, 3))

        # validate label data with shapes
        self.assertEqual(data['y_train'].shape, (9, 3))
        self.assertEqual(data['y_test'].shape, (3, 3))
