from os import makedirs, path as op
from shutil import copytree
from collections import Counter
import csv

import numpy as np
from PIL import Image

# create a greyscale folder for class labelled images
greyscale_folder = op.join('labels', 'grayscale')
if not op.isdir(greyscale_folder):
    makedirs(greyscale_folder)
labels = np.load('labels.npz')

# write our numpy array labels to images
# remove empty labels because we don't download images for them
keys = labels.keys()
class_freq = Counter()
image_freq = Counter()
for key in keys:
    label = labels[key]
    if np.sum(label):
        label_file = op.join(greyscale_folder, '{}.png'.format(key))
        img = Image.fromarray(label.astype(np.uint8))
        print('Writing {}'.format(label_file))
        img.save(label_file)
        # get class frequencies
        unique, counts = np.unique(label, return_counts=True)
        freq = dict(zip(unique, counts))
        for k, v in freq.items():
            class_freq[k] += v
            image_freq[k] += 1
    else:
        keys.remove(key)

# copy our tiles to a folder with a different name
copytree('tiles', 'images')

# sample the file names and use those to create text files
np.random.shuffle(keys)
split_index = int(len(keys) * 0.8)

with open('train.txt', 'w') as train:
    for key in keys[:split_index]:
        train.write('/data/images/{}.png /data/labels/grayscale/{}.png\n'.format(key, key))

with open('val.txt', 'w') as val:
    for key in keys[split_index:]:
        val.write('/data/images/{}.png /data/labels/grayscale/{}.png\n'.format(key, key))

# write a csv with class frequencies
freqs = [dict(label=k, frequency=v, image_count=image_freq[k]) for k, v in class_freq.items()]
with open('labels/label-stats.csv', 'w') as stats:
    fieldnames = list(freqs[0].keys())
    writer = csv.DictWriter(stats, fieldnames=fieldnames)

    writer.writeheader()
    for f in freqs:
        writer.writerow(f)
