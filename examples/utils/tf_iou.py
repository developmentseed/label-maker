"""
Python script to obtain IOU (aslo know as Jaccard index) for Tensorflow object detection.
For defination of IOU, please see: https://en.wikipedia.org/wiki/Jaccard_index.
Note:
The bbox from Label Maker is organized as [xmin, ymin, xmax, ymax],
but the bbox predection from TensorFlow Object Detection is [ymin, xmin, ymax, xmax], and it was
normalized by the image size. For more, see:
https://www.tensorflow.org/versions/r0.12/api_docs/python/image/working_with_bounding_boxes

Usage:
           python tf_iou.py   --model_name=building_od_ssd  \
                                      --path_to_label=data/building_od.pbtxt\
                                      --test_image_path=images/test
"""

import os
from os import makedirs, path as op
import sys
import glob
import six.moves.urllib as urllib
import tensorflow as tf
import tarfile
import pandas as pd

import numpy as np
from collections import defaultdict
from PIL import Image

sys.path.append("..")

from utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('model_name', '', 'Path to frozen detection graph')
flags.DEFINE_string('path_to_label', '', 'Path to label file')
flags.DEFINE_string('test_image_path', '', 'Path to test imgs and output diractory')
FLAGS = flags.FLAGS

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def bb_IOU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0,(xB - xA + 1)) * max(0,(yB - yA + 1))
    # compute the area of both the prediction and ground-truthrectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def pred_bbox():
    pred_bboxes = list()
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_path in test_imgs:
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                  [detection_boxes, detection_scores, detection_classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})
                  ### 256 here is the image size from Label Maker, adjust it according to your input image size.
                bboxe = (boxes*256).astype(np.int)
                bboxe = np.squeeze(bboxe)
                score = np.squeeze(((scores*100).transpose()).astype(np.int))
                ### only keep the bbox that prediction score is higher than 50.
                bboxes = bboxe[score > 50]
                if bboxes.any():
                    bboxes_ls = bboxes.tolist()
                    for bbox in bboxes_ls:
                        # pred_bboxes.append([image_path[-18:],bbox])
                        pred_bboxes.append(bbox)
    return pred_bboxes

def gr_bbox():
    gr_data = np.load("labels.npz")
    tile_names = [tile for tile in gr_data.files]
    tiles = np.array(tile_names)

    gr_bboxes = list()
    for tile in tiles:
        bboxes = gr_data[tile].tolist()
        bbox_info = list()
        if bboxes:
            for bbox in bboxes:
                bbox = [max(0, min(255, x)) for x in bbox[:4]]
                # switching bbox from [xmin, ymin, xmax, ymax] to [ymin, xmin, ymax, xmax]
                bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
            gr_bboxes.append(bbox)
    return gr_bboxes

def get_iou():
    pred_bboxes = pred_bbox()
    gr_bboxes = gr_bbox()
    iou_out = list()
    for pred_box in pred_bboxes:
        for gr_box in gr_bboxes:
            try:
                iou = bb_IOU(pred_box, gr_box)
                ### If iou is large than 0.5, we assume this prediction is acceptable.
                if iou >=0.5:
                    iou_out.append(iou)
            except:
                pass
    return iou_out

if __name__ =='__main__':
    model_name = op.join(os.getcwd(), FLAGS.model_name)
    # Path to frozen detection graph.
    path_to_ckpt = op.join(model_name,  'frozen_inference_graph.pb')
    # Path to the label file
    path_to_label = op.join(os.getcwd(), FLAGS.path_to_label)
    #only train on buildings
    num_classes = 1
    #Directory to test images path
    test_image_path = op.join(os.getcwd(), FLAGS.test_image_path)
    test_imgs = glob.glob(test_image_path + "/*.jpg")

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(path_to_label)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    iou_out = get_iou()
    pred_bboxes = pred_bbox()
    gr_bboxes = gr_bbox()

    print('*'*40)
    print("The precision score is: {}".format(float(len(iou_out)/len(pred_bboxes))))
    
    print("Precision is when the model predicts yes, how often is it correct.")
    print('*'*40)
