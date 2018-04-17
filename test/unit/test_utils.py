"""Tests for utils.py"""
import os
from os import path as op, makedirs
import unittest
import numpy as np
from PIL import Image

from label_maker.utils import url, class_match, get_tile_tif

class TestUtils(unittest.TestCase):
    """Tests for utility functions"""
    def test_url(self):
        """Test for url templating"""
        imagery = 'https://api.imagery.com/{z}/{x}/{y}.jpg'
        tile = '1-2-3'.split('-')
        filled = 'https://api.imagery.com/3/1/2.jpg'
        self.assertEqual(url(tile, imagery), filled)

    def test_class_match_classification(self):
        """Test class match function for classification problems"""
        ml_type = 'classification'
        class_index = 2
        passing = np.array([0, 0, 1])
        failing = np.array([0, 1, 0])
        self.assertTrue(class_match(ml_type, passing, class_index))
        self.assertFalse(class_match(ml_type, failing, class_index))

    def test_class_match_object(self):
        """Test class match function for object detection problems"""
        ml_type = 'object-detection'
        class_index = 2
        passing = np.array([[0, 0, 0, 0, 2]])
        failing = np.array([[0, 0, 0, 0, 1]])
        self.assertTrue(class_match(ml_type, passing, class_index))
        self.assertFalse(class_match(ml_type, failing, class_index))

    def test_class_match_segmentation(self):
        """Test class match function for segmentation problems"""
        ml_type = 'segmentation'
        class_index = 2
        passing = np.ones((256, 256), dtype=np.int) * 2
        failing = np.ones((256, 256), dtype=np.int)
        self.assertTrue(class_match(ml_type, passing, class_index))
        self.assertFalse(class_match(ml_type, failing, class_index))

    def test_get_tile_tif(self):
        """Test reading of tile from geotiff"""
        tile = '1087767-1046604-21'
        # create tiles directory
        dest_folder = 'test'
        tiles_dir = op.join(dest_folder, 'tiles')
        if not op.isdir(tiles_dir):
            makedirs(tiles_dir)

        get_tile_tif(tile, 'test/fixtures/drone.tif', dest_folder, None)
        test_tile = Image.open('test/tiles/{}.jpg'.format(tile))
        fixture_tile = Image.open('test/fixtures/{}.jpg'.format(tile))
        self.assertEqual(test_tile, fixture_tile)

    def test_get_tile_tif_offset(self):
        """Test reading of tile from geotiff with imagery_offset, test fixture"""
        tile = '1087767-1046604-21'
        # create tiles directory
        dest_folder = 'test'
        tiles_dir = op.join(dest_folder, 'tiles')
        if not op.isdir(tiles_dir):
            makedirs(tiles_dir)

        get_tile_tif(tile, 'test/fixtures/drone.tif', dest_folder, [128, 64])
        test_tile = Image.open('test/tiles/{}.jpg'.format(tile))
        fixture_tile = Image.open('test/fixtures/{}_offset.jpg'.format(tile))
        self.assertEqual(test_tile, fixture_tile)

    def test_get_tile_vrt(self):
        """Test reading of tile from a virtual raster"""
        tile = '1087767-1046604-21'
        # create tiles directory
        dest_folder = 'test'
        tiles_dir = op.join(dest_folder, 'tiles')
        if not op.isdir(tiles_dir):
            makedirs(tiles_dir)

        get_tile_tif(tile, 'test/fixtures/drone.vrt', dest_folder, None)
        test_tile = Image.open('test/tiles/{}.jpg'.format(tile))
        fixture_tile = Image.open('test/fixtures/{}.jpg'.format(tile))
        self.assertEqual(test_tile, fixture_tile)

if __name__ == '__main__':
    unittest.main()
