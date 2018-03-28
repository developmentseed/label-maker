"""Tests for label.py"""
import unittest
import numpy as np

from label_maker.label import _mapper, _convert_coordinates, _pixel_bbox, _pixel_bounds_convert, _bbox_class

class TestLabel(unittest.TestCase):
    """Tests for private label functions"""
    def test_mapper(self):
        """Test for private function _mapper"""
        # test tile data has one building feature with bbox [100, 100, 300, 300] within the tile
        test_tile_data = b'\x1a5\n\x03osm\x12\x18\x12\x02\x00\x00\x18\x03"\x10\t\xc8\x01\xb8>\x1a\x00\x8f\x03\x90\x03\x00\x00\x90\x03\x0f\x1a\x08building"\x05\n\x03yes(\x80 x\x01'
        classes = [dict(name="Building", filter=['has', 'building'])]
        x, y, z = (1, 2, 3)

        # for classification we return the one-hot array
        ml_type = 'classification'
        tile, label = _mapper(x, y, z, test_tile_data, dict(ml_type=ml_type, classes=classes))
        self.assertEqual(tile, ('{!s}-{!s}-{!s}'.format(x, y, z)))
        self.assertTrue(np.array_equal(label, np.array([0, 1], dtype=np.int)))

        # for object-detection we return the bounding box and class
        ml_type = 'object-detection'
        tile, label = _mapper(x, y, z, test_tile_data, dict(ml_type=ml_type, classes=classes))
        self.assertEqual(tile, ('{!s}-{!s}-{!s}'.format(x, y, z)))
        self.assertTrue(np.array_equal(label, np.array([[2, 232, 23, 253, 1]], dtype=np.int)))

        # for segmentation we return the rasterized image
        ml_type = 'segmentation'
        tile, label = _mapper(x, y, z, test_tile_data, dict(ml_type=ml_type, classes=classes))
        match_label = np.zeros((256, 256), dtype=np.int)
        match_label[236:249, 6:19] = 1
        self.assertEqual(tile, ('{!s}-{!s}-{!s}'.format(x, y, z)))
        self.assertTrue(np.array_equal(label, match_label))

    def test_convert_coordinates(self):
        """Test for private function _convert_coordinates"""
        # this is mostly a convenience function for handling multiple list wrapping
        # common with GeoJSON coordinate arrays
        self.assertEqual(_convert_coordinates([1024, 512]), [64, 223])
        self.assertEqual(_convert_coordinates([[1024, 512], [3072, 3584]]), [[64, 223], [191, 32]])
        self.assertEqual(_convert_coordinates([[[1024, 512], [3072, 3584]]]), [[[64, 223], [191, 32]]])

    def test_pixel_bbox(self):
        """Test for private function _pixel_bbox"""
        self.assertEqual(_pixel_bbox([1024, 512, 3072, 3584]), [60, 28, 195, 227])

    def test_pixel_bounds_convert(self):
        """Test for private function _pixel_bounds_convert"""
        # ensure x and y edges match
        self.assertEqual(_pixel_bounds_convert((0, 4096)), 255)
        self.assertEqual(_pixel_bounds_convert((0, 0)), 0)
        self.assertEqual(_pixel_bounds_convert((1, 0)), 255)
        self.assertEqual(_pixel_bounds_convert((1, 4096)), 0)

        # test interior point
        self.assertEqual(_pixel_bounds_convert((0, 2048)), 128)
        self.assertEqual(_pixel_bounds_convert((1, 2048)), 127)

    def test_bbox_class(self):
        """Test for private function _bbox_class"""
        bc = _bbox_class(3)
        passing = [0, 0, 1, 1, 3]
        failing = [0, 0, 1, 1, 2]
        self.assertTrue(bc(passing))
        self.assertFalse(bc(failing))

if __name__ == '__main__':
    unittest.main()
