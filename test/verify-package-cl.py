"""Validate that the output produced by integration testing on 'label-maker  package' matches our expectations"""
import numpy as np

data = np.load('integration-cl/data.npz')
# validate our image data with sums and shapes
assert np.sum(data['x_train']) == 144752757
assert np.sum(data['x_test']) == 52758414
assert data['x_train'].shape == (6, 256, 256, 3)
assert data['x_test'].shape == (2, 256, 256, 3)

# validate our label data with exact matches
expected_y_train = np.array(
    [[0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 1],
     [0, 1, 1, 0, 0, 0, 1],
     [0, 0, 0, 0, 1, 1, 1],
     [0, 0, 0, 0, 0, 0, 1]]
)
assert np.array_equal(data['y_train'], expected_y_train)

expected_y_test = np.array(
    [[0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 1]]
)
assert np.array_equal(data['y_test'], expected_y_test)
