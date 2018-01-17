import numpy as np
import os
from os import makedirs, path as op
import shutil

import pandas as pd


def pre_data_tf():
    """ to prep data for tensorflow object detection tf records"""
    labels = np.load(op.join(os.getcwd(), "labels.npz"))
    tile_names = [tile for tile in labels.files]
    tile_names.sort()
    tiles = np.array(tile_names)

    tf_tiles_info = []

    for tile in tiles:
        width = 256
        height = 256
        bboxes = labels[tile].tolist()
        if len(bboxes) != 0:
            for bbox in bboxes:
                if bbox[4] == 1:
                    cl_str = "Buildings"
                    y = [tile, width, height, cl_str, bbox[0], bbox[1], bbox[2], bbox[3]]
                    tf_tiles_info.append(y)
    split_index = int(len(tf_tiles_info) * 0.8)
    column_name = ['tile', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    df = pd.DataFrame(tf_tiles_info, columns=column_name)
    train_df = df[:split_index]
    test_df = df[split_index:]

    tiles_dir = op.join(os.getcwd(), 'tiles')
    train_dir = op.join(os.getcwd(), 'train')
    test_dir = op.join(os.getcwd(), 'test')
    if not op.isdir(train_dir):
        makedirs(train_dir)
    if not op.isdir(test_dir):
        makedirs(test_dir)
    for tile in train_df['tile']:
        tile_dir = op.join(tiles_dir, '{}.jpg'.format(tile))
        shutil.copy(tile_dir, train_dir)

    for tile in test_df['tile']:
        tile_dir = op.join(tiles_dir, '{}.jpg'.format(tile))
        shutil.copy(tile_dir, test_dir)

    train_df.to_csv(train_dir + "train.csv", index=None)
    test_df.to_csv(test_dir + "test_df.csv", index=None)
    print("{0} training and {1} test images for createing tf records are done!".format(
        len(train_df), len(test_df)))


if __name__ == '__main__':
    train_len = 0.8
    pre_data_tf()
