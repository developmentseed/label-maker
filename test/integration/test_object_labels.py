"""Test that the following CLI command returns the expected outputs
label-maker labels -d integration-od -c test/fixtures/integration/config.integration.object_detection.json"""
import unittest
from os import makedirs
from shutil import copyfile, rmtree
import subprocess

import numpy as np

class TestObjectDetectionLabel(unittest.TestCase):
    """Tests for object detection label creation"""
    @classmethod
    def setUpClass(cls):
        makedirs('integration-od')
        copyfile('test/fixtures/integration/portugal-z17.mbtiles', 'integration-od/portugal-z17.mbtiles')

    @classmethod
    def tearDownClass(cls):
        rmtree('integration-od')

    def test_cli(self):
        """Verify labels.npz produced by CLI"""
        cmd = 'label-maker labels -d integration-od -c test/fixtures/integration/config.integration.object_detection.json'
        cmd = cmd.split(' ')
        subprocess.run(cmd, universal_newlines=True)

        labels = np.load("integration-od/labels.npz")

        expected_bboxes = dict()
        expected_bboxes['62092-50162-17'] = np.empty((0, 5))
        expected_bboxes['62092-50163-17'] = np.array([
            [209, 192, 259, 259, 6], [251, 251, 259, 259, 6]
        ])
        expected_bboxes['62092-50164-17'] = np.array([
            [209, -4, 250, 28, 6], [242, -4, 259, 28, 6],
            [222, 13, 235, 66, 6], [87, 20, 250, 259, 6]
        ])
        expected_bboxes['62093-50162-17'] = np.array([
            [81, 145, 128, 259, 6], [124, -4, 218, 259, 6],
            [207, -4, 247, 153, 6], [140, 108, 193, 259, 6],
            [125, 236, 152, 259, 6], [162, 177, 176, 216, 6],
            [170, 151, 214, 179, 6], [141, 166, 244, 259, 6],
            [203, 88, 259, 186, 6]
        ])
        expected_bboxes['62093-50163-17'] = np.array([
            [81, -4, 125, 15, 6], [117, -4, 133, 17, 6],
            [119, -4, 151, 36, 6], [125, -4, 140, 7, 6],
            [141, -4, 187, 7, 6], [64, 32, 91, 60, 4],
            [84, 50, 106, 64, 6], [111, 9, 127, 26, 6],
            [111, 18, 127, 35, 6], [84, 15, 119, 52, 6],
            [74, 6, 129, 69, 5], [93, 24, 123, 46, 6],
            [88, 27, 127, 93, 6], [-4, 85, 96, 213, 6],
            [-2, 85, 96, 259, 6], [115, 38, 259, 100, 6]
        ])
        expected_bboxes['62094-50162-17'] = np.array([
            [67, -4, 172, 248, 6], [-4, 172, 90, 259, 6],
            [91, 170, 259, 227, 6]
        ])
        expected_bboxes['62093-50164-17'] = np.array([
            [-4, -4, 12, 22, 6], [207, 158, 259, 195, 6]
        ])
        expected_bboxes['62094-50163-17'] = np.array([
            [73, -4, 259, 78, 6], [30, 166, 60, 196, 1],
            [30, 166, 60, 196, 2], [203, 129, 259, 259, 6],
            [-4, 90, 259, 138, 6]
        ])
        expected_bboxes['62094-50164-17'] = np.array([
            [158, -4, 216, 82, 6], [-4, 108, 147, 173, 6],
            [139, 74, 254, 143, 6], [240, 90, 259, 232, 6]
        ])

        self.assertEqual(len(labels.files), len(expected_bboxes.keys())) # First check the number of tiles
        for tile in labels.files:
            self.assertTrue(np.array_equal(expected_bboxes[tile], labels[tile]))  # Now, bboxes
