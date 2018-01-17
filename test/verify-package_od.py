import numpy as np

data = np.load('integration/data.npz')

assert np.sum(data['x_train']) == 0
assert np.sum(data['x_test']) == 0
assert data['x_train'].shape == (0, )
assert data['x_test'].shape == (0, )


# validate our label data with exact matches
expected_y_train = np.array([])
assert np.array_equal(data['y_train'], expected_y_train)

expected_y_test = np.array([])
assert np.array_equal(data['y_test'], expected_y_test)
