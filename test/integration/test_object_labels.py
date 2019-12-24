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
        copyfile('test/fixtures/integration/portugal-z10.mbtiles', 'integration-od/portugal-z10.mbtiles')
        copyfile('test/fixtures/integration/spain-z10.mbtiles', 'integration-od/spain-z10.mbtiles')

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

        expected_bboxes['491-396-10'] = np.empty((0,5))
        
        expected_bboxes['491-397-10'] = np.empty((0,5))
        
        expected_bboxes['492-395-10'] = np.array([[243, 50, 251, 58, 2]])
        
        expected_bboxes['491-395-10'] = np.empty((0,5))
        
        expected_bboxes['492-394-10'] = np.array([[ 31, 58, 39, 66, 2],
                                                  [ 28, 119, 36, 127, 2],
                                                  [ 31, 66, 39,  74, 2],
                                                  [ 34, 60, 42, 68, 2],
                                                  [ 36, 65, 44, 73, 2],
                                                  [ 83, 231, 91, 239, 2]]
                                                  )
        
        expected_bboxes['490-394-10'] = np.array([[ 53,  89,  61,  97,   1],
                                                  [203, 123, 211, 131,   1],
                                                  [204, 122, 212, 131,   1]]
                                                  )
                                                  
        expected_bboxes['491-394-10'] = np.empty((0,5))
        
        expected_bboxes['488-396-10'] = np.array([[122,   0, 130,   4,   2],
                                                  [149,  55, 157,  63,   1]]
                                                  )
                                                  
        expected_bboxes['490-396-10'] = np.array([[157,  62, 165,  70,   2],
                                                  [157,  62, 165,  71,   2],
                                                  [158,  60, 166,  68,   2],
                                                  [158,  60, 166,  68,   2],
                                                  [158,  61, 166,  70,   2],
                                                  [158,  62, 166,  70,   2],
                                                  [184, 228, 192, 236,   1]]
                                                  )
        
        expected_bboxes['490-395-10'] = np.empty((0,5))
        
        expected_bboxes['489-396-10'] = np.array([[ 63,  68,  71,  76,   1],
                                                  [229, 253, 237, 255,   1]]
                                                  )
        
        expected_bboxes['488-397-10'] = np.array([[109, 159, 117, 167,   2],
                                                  [109, 159, 118, 167,   2],
                                                  [252, 167, 255, 175,   2]]
                                                  )
                                                  
        expected_bboxes['488-395-10'] = np.array([[ 65, 120,  73, 128,   2],
                                                  [ 99, 169, 107, 177,   1],
                                                  [ 98, 216, 106, 224,   1],
                                                  [139, 139, 147, 147,   2],
                                                  [135, 143, 143, 151,   2],
                                                  [139, 140, 147, 148,   2],
                                                  [122, 251, 130, 255,   2]]
                                                  )
        
        expected_bboxes['489-397-10'] = np.array([[  0, 167,   5, 175,   2],
                                                  [  8, 160,  16, 168,   2],
                                                  [ 24, 176,  32, 184,   2],
                                                  [ 58, 176,  66, 184,   2],
                                                  [150,  63, 158,  71,   2],
                                                  [229,   0, 237,   6,   1]]
                                                  )
                                                  
        expected_bboxes['488-394-10'] = np.array([[ 11, 243,  19, 251,   1],
                                                  [208,  92, 216, 100,   1]]
                                                  )
        
        expected_bboxes['489-394-10'] = np.array([[ 85, 207,  93, 215,   1],
                                                  [134,  92, 142, 100,   1]]
                                                  )
        
        expected_bboxes['489-395-10'] = np.array([[ 48, 204,  57, 212,   1]])
        
        expected_bboxes['490-397-10'] = np.array([[206,  72, 214,  80,   1],
                                                  [193,  95, 201, 103,   1],
                                                  [213, 211, 221, 219,   2],
                                                  [222, 223, 230, 231,   1]]
                                                  )
        
        expected_bboxes['492-396-10'] = np.array([[ 57,   6,  65,  14,   2],
                                                  [ 60,   9,  68,  17,   2],
                                                  [ 79,  21,  87,  29,   2],
                                                  [147,  81, 155,  89,   2],
                                                  [148,  91, 156,  99,   2],
                                                  [135, 101, 143, 109,   2]]
                                                  )
        
        expected_bboxes['492-397-10'] = np.array([[ 40, 231,  48, 239,   1],
                                                  [ 71, 149,  79, 157,   2],
                                                  [134, 147, 143, 156,   2]]
                                                  )
        
        expected_bboxes['493-394-10'] = np.array([[ 13, 186,  21, 194,   2],
                                                  [ 21, 191,  29, 199,   2]]
                                                  )
        
        expected_bboxes['493-395-10'] = np.array([[17, 47, 25, 55,  2]])
        
        expected_bboxes['493-396-10'] = np.empty((0,5))
        
        expected_bboxes['493-397-10'] = np.array([[118, 166, 126, 174,   1],
                                                  [172,  43, 180,  51,   1],
                                                  [209, 133, 217, 141,   1]]
                                                  )

        self.assertEqual(len(labels.files), len(expected_bboxes.keys())) # First check the number of tiles
        for tile in labels.files:
            self.assertTrue(np.array_equal(expected_bboxes[tile], labels[tile]))  # Now, bboxes
