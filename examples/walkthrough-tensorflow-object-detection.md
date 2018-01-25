# Example Use: A building object detection with Tensor Flow

We have to walk through all these steps to be able to train a TensorFlow Object Detection and have this results:
![vietnam_od_buildings](https://user-images.githubusercontent.com/14057932/35354064-0cd453f6-0117-11e8-8e6f-96f5619bcdf3.png)


# Download Training dataset
Vietnam has good image through Mapbox Satellite Imagery, so we are going to use the same configure JSON file we used for [another walkthrough](https://github.com/developmentseed/label-maker/blob/tf_object_detection/examples/walkthrough-classification-mxnet-sagemaker.md)
- Install Label Maker by: `pip install label_maker` .
- Create vietnam_tf.json for `object-detection` like shown in following json file.
```json
{
  "country": "vietnam",
  "bounding_box": [105.42,20.75,106.41,21.53],
  "zoom": 15,
  "classes": [
    { "name": "Buildings", "filter": ["has", "building"] }
  ],
  "imagery": "http://a.tiles.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.jpg?access_token=ACCESS_TOKEN",
  "background_ratio": 1,
  "ml_type": "object-detection"
}
```
If you don't know know what is `country`, `bounding_box`, `zoom`, `classes`, `imagery`, `background_ratio` and `ml_type`, please check other two walkthrough examples we provided: [image classification with AWS and Keras](https://github.com/developmentseed/label-maker/blob/tf_object_detection/examples/walkthrough-classification-aws.md) and [image classification with MXNet and SageMaker](https://github.com/developmentseed/label-maker/blob/tf_object_detection/examples/walkthrough-classification-mxnet-sagemaker.md).

# Training dataset generation
We'll follow the [CLI commands from the README](https://github.com/developmentseed/label-maker#command-line-use) but use a separate folder to keep our project well-managed.

```shell
$ label-maker download --dest vietnam_building_tf --config vietnam_tf.json
$ label-maker labels --dest vietnam_building_tf --config vietnam_tf.json
```
These commands will download and retile the OpenStreetMap QA tiles, label maker will find buildings from OpenSreetMap and draw bounding box/boxes, you could visualize the tiles in your `vietnam_building_tf/labels` folder that label maker creates.
The label tiles will be save as an numpy `labels.npz` that you will spot in the `vietnam_building_tf`folder.

![26014-14421-15](https://user-images.githubusercontent.com/14057932/35344642-a130760e-00fb-11e8-85ea-628e00c13dd5.png)  ![26014-14425-15](https://user-images.githubusercontent.com/14057932/35344651-a75e74fe-00fb-11e8-95d3-3685c6fa170f.png)

You could preview how the building bounding boxes are drawn on top of the RGB image tiles by:

```shell
$ label-maker preview -n 10 --dest vietnam_building_tf --config vietnam_tf.json
```
You will see there are ten image tiles download in the foler `vietnam_building_tf/examples/Buildings/`

![25989-14427-15](https://user-images.githubusercontent.com/14057932/35346218-a375d342-00ff-11e8-9ee1-8ff5c4396290.jpg) ![25991-14439-15](https://user-images.githubusercontent.com/14057932/35346236-b2c3e3ac-00ff-11e8-9ad3-250a756e852f.jpg)

The building bounding boxes overlaid with image tiles will be look like above images.
You could tell from the above image tiles that a lot of buildings in Vietnam have been mapped and without bounding boxes yet, which would impact on our model prediction accuracy. If you’d like to help improve the labelling accuracy, [start mapping on OpenStreetMap](https://www.openstreetmap.org/#map=10/20.9755/105.4118).
Download all the image tiles that contain buildings by running
```shell
$ label-maker images --dest vietnam_building_tf --config vietnam_tf.json
```
You will have 385 image tiles in your folder `vietnam_building_tf/tiles`You don't need to run `label-maker package` for this TensorFlow Object Detection task.

Now, you are ready to set up TensorFlow object detection API

# Setup TensorFlow Object Detection API
### Install TensorFlow object detection:
- `git clone https://github.com/tensorflow/models.git`;
- Install TensorFlow Object Detection API by following strictly [this readme](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). You need to have everything installed, and after `python object_detection/builders/model_builder_test.py` past you will be ready to go to next step.

### Create tensorflow TFRecords training.
Tensorflow API supports a variety of file formats. TFRecord file format is a simple record-oriented binary format that many TensorFlow applications use. We have a python code [here on our github repo](TODO...After this branch merge) for you to create TFRecords from `labels.npz` that Label Maker created under `label-maker labels` command.

Follow these steps to create TFRecards
- Create the `tf_records_generation.py` under your Tensorflow object detection directory,which is `models/research/object_detection/`,  from [our github repo](), .
- Copy and paste your `labels.npz` and `tiles` folder from `vietnam_building_tf` to TensorFlow Object Detection API directory `models/research/object_detection/` too.
- From `models/research/object_detection/` directory run:
```shell
python3 tf_records_generation.py --label_input=labels.npz \
             --train_rd_path=data/train_buildings.record \
             --test_rd_path=data/test_buildings.record
```
This code will create a `train_buildings.record` and `test_buildings.record` in a folder called `data` in directory `models/research/object_detection/`

### Object detection model setup
- download a trained model from TensorFlow. Go to [TensorFlow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) to pick up a trained model. We used [`ssd_inception_v2_coco`](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz).

It's the second fastest model after `ssd_mobilenet_v1_coco`.
run this command to unzip the downloaded model:
```shell
tar -xzvf ssd_inception_v2_coco_2017_11_17.tar.gz  |  rm ssd_inception_v2_coco_2017_11_17.tar.gz
```
- [Go here to copy and create  a model configure file](https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_inception_v2_coco.config) for `ssd_inception_v2_coco` as `ssd_inception_v2_coco.config`.

- Create label file, let's name it `building_od.pbtex`, copy and paste following content in the file:

```JSON
item {
  id: 1
  name: 'building'
}

```
- Go back to `ssd_inception_v2_coco.config` and change all the `PATH_TO_BE_CONFIGURED` to the your file directories, e.g. at `fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"` you will need to change `PATH_TO_BE_CONFIGURED` to `ssd_inception_v2_coco_2017_11_17` in the configure file. Don't forget to change `input_path` for `train_buildings.record`, `label_map_path`, and `input_path` for `test_buildings.record` too.

# Train the TensorFlow object detection model
You are now ready to train the model, from the `models/research/object_detection` directory, run:
```shell
python3 train.py --logtostderr \
             --train_dir=training/ \
             --pipeline_config_path=training/ssd_inception_v2_coco.config
```
We created `training` folder to save up all the model checkpoint and output in this task, you could do the same or create a different one.
When model is successfully running you will see:
![screenshot 2018-01-24 14 30 22](https://user-images.githubusercontent.com/14057932/35352821-27df9a6a-0113-11e8-8c44-61d994d34ae6.png)

We ran this model about **10,000 steps**, and it took **37 hours**, if you wanna run a faster model, we recommend to try out `ssd_mobilenet_v1_coco` in TensorFlow model zoo.

# Create your object detection model to visualize
After the 10,000 steps, you will ready to create your own building detection model, simply run:
```shell
python3 export_inference_graph.py --input_type image_tensor \
              --pipeline_config_path training/ssd_inception_v2_coco.config \
              --trained_checkpoint_prefix training/model.ckpt-9575
              --output_directory building_od_ssd_inference_graph
```
Now we just created our building detection model inference graph, and it could visualize with `tensorboard`.
You will see the whole graph by running:
```shell
tensorboard --logdir='training'
```
go to your web browser and paste `http://127.0.0.1:6006/`, you will see.
![screenshot 2018-01-24 14 42 00](https://user-images.githubusercontent.com/14057932/35353302-c6f555c6-0114-11e8-8c7c-1b5334e9c449.png)

# Prediction
We created another code for you to finally run and predict any image tile you have to do the building detection. We saved the test images in a folder called `test_images`. Now [go to our Label Maker repo, copy, paste and save the code as `tf_od_predict.py`](https://github.com/developmentseed/label-maker/blob/tf_object_detection/examples/utils/tf_od_predict.py), simply run:
```shell
python3 tf_od_predict.py --model_name=building_od_ssd_inference_graph \
                                          --path_to_label=data/building_od.pbtxt \
                                          --test_image_path=test_images
```

This code will read through all your test image in `test_images` folder, and output final prediction in the `test_images` too. You will see the final prediction like this:

![26011-14427-15](https://user-images.githubusercontent.com/14057932/35353614-b1709390-0115-11e8-8277-08768034006d.jpg) ![25989-14427-15](https://user-images.githubusercontent.com/14057932/35353624-bacf4846-0115-11e8-9fd0-b3c75cfaaa06.jpg)
