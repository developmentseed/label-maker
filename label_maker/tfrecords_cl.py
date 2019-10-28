import tensorflow as tf
import numpy as np


def _bytes_feature(value):
    """Returns a bytes_list from a single image numpy array representation from label maker ie npz['x_train'][i]"""

    value = tf.io.serialize_tensor(x)
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a single label array from label maker ie  npz['y_train'][i]."""

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def gen_tf_image_example(img, label):
    feature = {
        'image': _bytes_feature(img),
        'label': _int64_feature(label)}
    return tf.train.Example(features=tf.train.Features(feature=feature))


def convert_to_tfrecords(dest_folder, npz_path):
    # write doc strings
    # confirm both pathes exist

    npz = np.load(npz_path)

    with tf.io.TFRecordWriter(dest_folder + 'train.tfrecords') as writer:
        for i, data in enumerate(npz['x_train']):
            tf_example = gen_tf_image_example(npz['x_train'][i], npz['y_train'][i])
            writer.write(tf_example.SerializeToString())
        print('TFrecords created for train.')

    with tf.io.TFRecordWriter(dest_folder + 'test.tfrecords') as writer:
        for i, data in enumerate(npz['x_test']):
            tf_example = gen_tf_image_example(npz['x_test'][i], npz['y_test'][i])
            writer.write(tf_example.SerializeToString())
        print('TFrecords created for test.')

    try:
        npz['x_val']
        with tf.io.TFRecordWriter(dest_folder + 'val.tfrecords') as writer:
            for i, data in enumerate(npz['x_val']):
                tf_example = gen_tf_image_example(npz['x_val'][i], npz['y_val'][i])
                writer.write(tf_example.SerializeToString())
            print('TFrecords created for val.')
    except KeyError:
        print('TFrecords creation complete')

