"""Tests for utils.py"""
import os
from os import path as op, makedirs
import shutil
import tempfile
import unittest
import numpy as np
from PIL import Image

from label_maker.utils import url, class_match, get_tile_tif, get_tile_wms, is_tif

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

    def test_is_tif(self):
        """Test identifying tif or vrt files as tif"""
        img_dir = op.join('test', 'fixtures')

        # tif with .tif extension identified as tif
        test_tif = op.join(img_dir, 'drone.tif')
        self.assertTrue(is_tif(test_tif))

        # vrt with .vrt extension identified as tif
        test_vrt = op.join(img_dir, 'drone.vrt')
        self.assertTrue(is_tif(test_vrt))


        # tif with no extension identified as tif
        with tempfile.TemporaryDirectory() as tmpdirname:
            test_tif_no_ext = op.join(tmpdirname, 'drone')
            shutil.copy(test_tif, test_tif_no_ext)
            self.assertTrue(is_tif(test_tif_no_ext))

        # vrt with no extension identified as tif
        with tempfile.TemporaryDirectory() as tmpdirname:
            test_vrt_no_ext = op.join(tmpdirname, 'drone')
            shutil.copy(test_vrt, test_vrt_no_ext)
            self.assertTrue(is_tif(test_vrt_no_ext))



    def test_get_tile_tif(self):
        """Test reading of tile from geotiff"""
        tile = '1087767-1046604-21'
        # create tiles directory
        dest_folder = 'test'
        tiles_dir = op.join(dest_folder, 'tiles')
        if not op.isdir(tiles_dir):
            makedirs(tiles_dir)

        get_tile_tif(tile, 'test/fixtures/drone.tif', tiles_dir, {})
        test_tile = Image.open('test/tiles/{}.tif'.format(tile))
        fixture_tile = Image.open('test/fixtures/{}.tif'.format(tile))
        self.assertEqual(test_tile, fixture_tile)

    def test_get_tile_tif_offset(self):
        """Test reading of tile from geotiff with imagery_offset, test fixture"""
        tile = '1087767-1046604-21'
        # create tiles directory
        dest_folder = 'test'
        tiles_dir = op.join(dest_folder, 'tiles')
        print(tiles_dir)
        if not op.isdir(tiles_dir):
            makedirs(tiles_dir)

        get_tile_tif(tile, 'test/fixtures/drone.tif', tiles_dir,
                     {'imagery_offset': [128, 64]})
        test_tile = Image.open('test/tiles/{}.tif'.format(tile))
        fixture_tile = Image.open('test/fixtures/{}_offset.tif'.format(tile))
        self.assertEqual(test_tile, fixture_tile)

    def test_get_tile_vrt(self):
        """Test reading of tile from a virtual raster"""
        tile = '1087767-1046604-21'
        # create tiles directory
        dest_folder = 'test'
        tiles_dir = op.join(dest_folder, 'tiles')
        if not op.isdir(tiles_dir):
            makedirs(tiles_dir)

        get_tile_tif(tile, 'test/fixtures/drone.vrt', tiles_dir, {})
        test_tile = Image.open('test/tiles/{}.tif'.format(tile))
        fixture_tile = Image.open('test/fixtures/{}.tif'.format(tile))
        self.assertEqual(test_tile, fixture_tile)

    def test_get_tile_wms(self):
        """Test reading of tile from a WMS endpoint"""
        tile = '146-195-9'
        # create tiles directory
        dest_folder = 'test'
        tiles_dir = op.join(dest_folder, 'tiles')
        if not op.isdir(tiles_dir):
            makedirs(tiles_dir)

        nasa_url = 'https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&layers=MODIS_Aqua_CorrectedReflectance_TrueColor&version=1.3.0&crs=EPSG:4326&transparent=false&width=256&height=256&bbox={bbox}&format=image/jpeg&time=2019-03-05'

        get_tile_wms(tile, nasa_url, tiles_dir, {})
        test_tile = Image.open('test/tiles/{}.jpeg'.format(tile))
        fixture_tile = Image.open('test/fixtures/{}.jpeg'.format(tile))
        self.assertEqual(test_tile, fixture_tile)

if __name__ == '__main__':
    unittest.main()
