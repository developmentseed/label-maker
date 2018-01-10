"""Validate that the output produced by integration testing on 'label-maker labels' matches our expectations"""
import json
import numpy as np

# our labels should look like this
expected_labels = {
    '62092-50162-17': np.array([1, 0, 0, 0, 0, 0, 0]),
    '62092-50163-17': np.array([0, 0, 0, 0, 0, 0, 1]),
    '62092-50164-17': np.array([0, 0, 0, 0, 0, 0, 1]),
    '62093-50162-17': np.array([0, 0, 0, 0, 0, 0, 1]),
    '62093-50164-17': np.array([0, 0, 0, 0, 0, 0, 1]),
    '62094-50162-17': np.array([0, 0, 0, 0, 0, 0, 1]),
    '62094-50164-17': np.array([0, 0, 0, 0, 0, 0, 1]),
    '62094-50163-17': np.array([0, 1, 1, 0, 0, 0, 1]),
    '62093-50163-17': np.array([0, 0, 0, 0, 1, 1, 1])
}

labels = np.load('integration/labels.npz')
for tile in labels.files:
    assert np.array_equal(expected_labels[tile], labels[tile])

# our GeoJSON looks like the fixture
expected_geojson = json.load(open('test/fixtures/integration/classification.geojson'))
geojson = json.load(open('integration/classification.geojson'))

for feature in geojson['features']:
    assert feature in expected_geojson['features']

# our command line output should look like this
expected_output = """Determining labels for each tile
---
Water Tower: 1 tiles
Building: 1 tiles
Farmland: 0 tiles
Ruins: 1 tiles
Parking: 1 tiles
Roads: 8 tiles
Total tiles: 9
Write out labels to integration/labels.npz
"""

with open('stdout', 'r') as output:
    assert expected_output == output.read()
