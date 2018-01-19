"""Validate that the output produced by integration testing on 'label-maker package' matches our expectations"""
import numpy as np

data = np.load('integration-od/data.npz')

assert np.sum(data['x_train']) == 144752757
assert np.sum(data['x_test']) == 52758414
assert data['x_train'].shape == (6, 256, 256, 3)
assert data['x_test'].shape == (2, 256, 256, 3)

# validate our label data with exact matches in shape
assert data['y_train'].shape == (6, 16, 5)
assert data['y_test'].shape ==  (2, 16, 5)
