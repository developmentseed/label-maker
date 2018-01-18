"""Validate that the output produced by integration testing on 'label-maker labels'
matches our expectations, specifically for segmentation tasks"""
import numpy as np

# our labels should look like this
expected_sums = {
    '62092-50162-17': 0,
    '62092-50163-17': 2760,
    '62092-50164-17': 13728,
    '62093-50162-17': 38004,
    '62093-50164-17': 2688,
    '62094-50162-17': 21630,
    '62094-50164-17': 19440,
    '62094-50163-17': 21895,
    '62093-50163-17': 32198
}

labels = np.load('integration-sg/labels.npz')
for tile in labels.files:
    assert expected_sums[tile] == np.sum(labels[tile])

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
Write out labels to integration-sg/labels.npz
Writing 62092-50163-17.png
Writing 62092-50164-17.png
Writing 62093-50164-17.png
Writing 62094-50162-17.png
Writing 62093-50162-17.png
Writing 62094-50164-17.png
Writing 62094-50163-17.png
Writing 62093-50163-17.png
"""

with open('stdout', 'r') as output:
    assert expected_output == output.read()
