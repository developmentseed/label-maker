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
            [209, 192, 255, 255, 6], [253, 251, 255, 255, 6]
        ])
        expected_bboxes['62092-50164-17'] = np.array([
            [209, 0, 250, 28, 6], [242, 0, 255, 28, 6],
            [222, 13, 235, 66, 6], [87, 20, 250, 255, 6]
        ])
        expected_bboxes['62093-50162-17'] = np.array([
            [81, 145, 128, 255, 6], [124, 0, 218, 255, 6],
            [207, 0, 247, 153, 6], [140, 108, 193, 255, 6],
            [125, 236, 152, 255, 6], [162, 177, 176, 216, 6],
            [170, 151, 214, 179, 6], [141, 166, 244, 255, 6],
            [203, 88, 255, 186, 6]
        ])
        expected_bboxes['62093-50163-17'] = np.array([
            [81, 0, 125, 15, 6], [117, 0, 133, 17, 6],
            [119, 0, 151, 36, 6], [125, 0, 140, 7, 6],
            [141, 0, 187, 7, 6], [64, 32, 91, 60, 4],
            [84, 50, 106, 64, 6], [111, 9, 127, 26, 6],
            [111, 18, 127, 35, 6], [84, 15, 119, 52, 6],
            [74, 6, 129, 69, 5], [93, 24, 123, 46, 6],
            [88, 27, 127, 93, 6], [0, 85, 96, 213, 6],
            [0, 85, 96, 255, 6], [115, 38, 255, 100, 6]
        ])
        expected_bboxes['62094-50162-17'] = np.array([
            [67, 0, 172, 248, 6], [0, 172, 90, 255, 6],
            [91, 170, 255, 227, 6]
        ])
        expected_bboxes['62093-50164-17'] = np.array([
            [0, 0, 12, 22, 6], [207, 158, 255, 195, 6]
        ])
        expected_bboxes['62094-50163-17'] = np.array([
            [73, 0, 255, 78, 6], [30, 166, 60, 196, 1],
            [30, 166, 60, 196, 2], [203, 129, 255, 255, 6],
            [0, 90, 255, 138, 6]
        ])
        expected_bboxes['62094-50164-17'] = np.array([
            [158, 0, 216, 82, 6], [0, 108, 147, 173, 6],
            [139, 74, 254, 143, 6], [240, 90, 255, 232, 6]
        ])

        self.assertEqual(len(labels.files), len(expected_bboxes.keys())) # First check the number of tiles
        for tile in labels.files:
            self.assertTrue(np.array_equal(expected_bboxes[tile], labels[tile]))  # Now, bboxes
