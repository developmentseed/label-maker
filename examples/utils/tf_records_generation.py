"""
This code was modified on top of Google tensorflow
(https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)

To use this code with Label Maker and Tensor Flow object detection API, this code work similar to  `label-maker package`.
To create a correct training data set for Tensor Flow Object Detecrtion, we recommend you do:
1. After running `label-maker images`, do `git clone https://github.com/tensorflow/models.git`
2. Install TensorFlow object detection by following this: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
3. From your Label Maker, copy `tiles` folder, this code `tf_records_generation.py` and `labels.npz` to Tensorflow object detecrtion directory `tensorflow/models/research/`
4. From directory `tensorflow/models/research/` simply runs:
python3 tf_records_generation.py --label_input=labels.npz \
             --train_rd_path=data/train_buildings.record \
             --test_rd_path=data/test_buildings.record
"""

import os
import io
import numpy as np
from os import makedirs, path as op
import shutil

import tensorflow as tf
from PIL import Image
from utils import dataset_util
from collections import namedtuple


flags = tf.app.flags
flags.DEFINE_string('label_input', '', 'Path to the labels.npz input')
flags.DEFINE_string('train_rd_path', '', 'Path to output TFRecord')
flags.DEFINE_string('test_rd_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# replace this with label map
def class_text_to_int(row_label):
    """read in label data"""
    if row_label == 'building':
        return 1
    return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    """Creates a tf.Example proto from sample buillding image tile.

Args:
 encoded_building_image_data: The jpg encoded data of the building image.

Returns:
 example: The created tf.Example.
"""
    with tf.gfile.GFile(op.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for _, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    """coverts numpy array into tfrecord files, the numpy array is in order of  'filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'"""
    labels = np.load(op.join(os.getcwd(), FLAGS.label_input))
    tile_names = [tile for tile in labels.files]
    tile_names.sort()
    tiles = np.array(tile_names)

    tf_tiles_info = []

    for tile in tiles:
        bboxes = labels[tile].tolist()
        width = 256
        height = 256
        if len(bboxes) != 0:
            for bbox in bboxes:
                if bbox[4] == 1:
                    cl_str = "building"
                    if bbox[0] < 0:
                        bbox[0] = 0
                    if bbox[1] < 0:
                        bbox[1] = 0
                    if  bbox[2] > 256:
                        bbox[2] = 256
                    if  bbox[3] > 256:
                        bbox[3] = 256
                    y = ["{}.jpg".format(tile), width, height, cl_str, bbox[0], bbox[1], bbox[2], bbox[3]]
                    tf_tiles_info.append(y)
    #train_len = 0.8
    tf_array = np.array(tf_tiles_info)
    split_index = int(len(tf_tiles_info) *0.8 )
    tf_train = tf_array[0: split_index]
    tf_test = tf_array[split_index, :]
    print("You have {} training tiles and {} test tiles ready".format(
    len(set(list(tf_train[:,1]))), len(set(list(tf_test[:,1])))))
    tiles_dir = op.join(os.getcwd(), 'tiles')
    train_dir = op.join(os.getcwd(), 'images', 'train')
    test_dir = op.join(os.getcwd(), 'images', 'test')

    if not op.isdir(train_dir):
        makedirs(train_dir)
    if not op.isdir(test_dir):
        makedirs(test_dir)

    for tile in list(tf_train[:,1]):
        tile_dir = op.join(tiles_dir, tile)
        shutil.copy(tile_dir, train_dir)

    # for tile in test_df['filename']:
    for tile in list(tf_test[:,1]):
        tile_dir = op.join(tiles_dir, tile)
        shutil.copy(tile_dir, test_dir)
    ### for train
    writer = tf.python_io.TFRecordWriter(FLAGS.train_rd_path)
    tf_example = create_tf_example(tf_train, train_dir)
    writer.write(tf_example.SerializeToString())
    writer.close()
    output_train= op.join(os.getcwd(),FLAGS.train_rd_path)
    print('Successfully created the TFRecords: {}'.format(output_train))

    ### for test
    writer = tf.python_io.TFRecordWriter(FLAGS.test_rd_path)

    tf_example = create_tf_example(tf_test, test_dir)
    writer.write(tf_example.SerializeToString())
    writer.close()
    output_test = op.join(os.getcwd(),FLAGS.test_rd_path)
    print('Successfully created the TFRecords: {}'.format(output_test))

def _score_converter_fn_with_logit_scale(tf_score_converter_fn, logit_scale):
    def score_converter_fn(logits):
        cr = logit_scale
        cr = tf.constant([[cr]],tf.float32)
        print(logit_scale)
        print(logits)
        scaled_logits = tf.divide(logits, cr, name='scale_logits') #change logit_scale
        return tf_score_converter_fn(scaled_logits, name='convert_scores')
    score_converter_fn.__name__ = '%s_with_logit_scale' % (tf_score_converter_fn.__name__)
    return score_converter_fn


if __name__ == '__main__':
    tf.app.run()
