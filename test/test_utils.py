"""Tests for utils.py"""
import unittest
import numpy as np

from label_maker.utils import url, class_match

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

if __name__ == '__main__':
    unittest.main()
