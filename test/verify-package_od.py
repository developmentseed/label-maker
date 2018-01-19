import numpy as np

data = np.load('integration-od/data.npz')

assert np.sum(data['x_train']) == 144752757
assert np.sum(data['x_test']) == 52758414
assert data['x_train'].shape == (6, 256, 256, 3)
assert data['x_test'].shape == (2, 256, 256, 3)

# # validate our label data with exact matches in length
expected_y_train_len = 6
assert len(data['y_train']) == expected_y_train_len

expected_y_test_len = 2
assert len(data['y_test']) == expected_y_test_len


# # validate our label data with exact matches in shape

expected_y_train_shape= (6, 16, 5)
assert  data['y_train'].shape == expected_y_train_shape

expected_y_test_shape = (2, 16, 5)
assert data['y_test'].shape == expected_y_test_shape
