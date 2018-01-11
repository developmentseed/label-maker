# Example Use: Creating a building classifier in Cambodia using MXNet and SageMaker

As one of the tropical countries in Asia, Cambodia has quite a variety of land use types. From the satellite imagery, you could easily spot tropical forests in dark green, deforestation patches that mix green with bare earth, and buildings with a vast variety of roof colors. It would be an interesting challenge to build a building classifier with MXNet and to train in Amazon SageMaker specifically. Amazon SageMaker is a new service from Amazon Web Services (AWS) that enables users to develop, train, deploy and scale machine learning approaches in a fairly straightforward way.

# Download Training Dataset
Before playing with SageMaker, we need to get the training dataset prepared using Label Maker.
- Install: `pip install label_maker`;
- Create cam_buildings.json like shown in following json file;

```json
{
  "country": "cambodia",
  "bounding_box": [104.17785644531249,11.170318336920309,105.260009765625,12.098409789924855],
  "zoom": 15,
  "classes": [
    { "name": "Buildings", "filter": ["has", "building"] }
  ],
  "imagery": "http://a.tiles.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.jpg?access_token=ACCESS_TOKEN",
  "background_ratio": 1,
  "ml_type": "classification"
}
```

We're using the same configuration from [another walkthrough](../examples/walkthrough-classification-aws.md) with changes to location:
- `country` and `bounding_box`: changed to indicated the location in Cambodia to download data from.
- `zoom`: Buildings in Cambodia have quite a variety in size, so zoom 15 (roughly 5m resolution) will allow us to


# Training dataset generation

We'll follow the [CLI commands from the README](https://github.com/developmentseed/label-maker#command-line-use) but use a separate folder to keep our project well-managed.

```bash
$ label-maker download --dest cambodia_building --config cam_buildings.json
$ label-maker labels --dest cambodia_building --config cam_buildings.json
```

These commands will download and retile the OpenStreetMap QA tiles and use it to create our label data as `labels.npz`. We'll also get a file for inspection `classifcation.geojson`:

![cambodia_building](https://user-images.githubusercontent.com/14057932/34792813-3edb8ae6-f617-11e7-96b0-75b4a6b99887.png)
_Purple building tile labels overlaid over [Mapbox Satellite Imagery](https://www.mapbox.com/maps/satellite/)_

Preview the data with

```bash
$ label-maker preview -n 10 --dest cambodia_building --config cam_buildings.json
```

Example satellite images will be at `cambodia_building/examples`

![25927-15322-15](https://user-images.githubusercontent.com/14057932/34792884-83e718ee-f617-11e7-8a5b-fce11b42f4cc.jpg)

When you're ready, download all 560 imagery tiles and package it into our final file:

```bash
$ label-maker images --dest cambodia_building --config cam_buildings.json
$ label-maker package --dest cambodia_building --config cam_buildings.json
```

We'll use the final file `cambodia_building/data.npz` to start training the model on Sagemaker

# Setup Amazon Sagemaker
Here are few steps to follow if you are interested in using it to train an image classification with MXNet:
- You could go to your [AWS console](https://console.aws.amazon.com);
- Log in your account, and go to the [sagemaker home page](https://console.aws.amazon.com/sagemaker/)
- Create a Notebook Instance! ![screenshot 2017-12-20 17 20 42](https://user-images.githubusercontent.com/14057932/34264652-912868ea-e641-11e7-9877-60ede67eb421.png) Click on `Create notebook Instance`. You will have three instance options, `ml.t2.medium`, `ml.m4.xlarge` and `ml.p2.xlarge`, to choose from. We recommend you to use the p2 machine (a gpu machine) to train this image classification.

Once you have your p2 instance notebook set up, you are ready to train a classifier. Specifically, you are going to learn how to plug your own script into Amazon SageMaker MXNet Estimator and train the classifier we prepared for detecting buildings in images.


# Train the model with MXNet on AWS SageMaker
Training a LeNet building classifier using MXNet Estimator:
- Prepare your own training script, and you could use our `mx_lenet_sagemaker.py` here, just slightly modify it; You could see and follow from the [Jupyter Notebook we prepared](https://github.com/developmentseed/ml-data-generation/blob/mx_net/examples/nets/SageMaker_mx-lenet.ipynb).
- Run the script on SageMaker via an MXNet Estimator, use the script Jupyter Notebook `SageMaker_mx-lenet.ipynb` directly.
  - Inside of the MXNet estimator you need to have you entry-point, which is the prepared script `mx_lenet_sagemaker.py`;
  - Your SageMaker `role`, and it could be obtained by `get_execution_role`;
  - The `train_instance_type`, we used and also recommend GPU instance `ml.p2.xlarge"` here;
  - The `train_instance_count` is equal to 1, which means we are gonna train this LeNet on only one machine. Apparently, you could train the model by multiple machines through SageMaker.
  - Pass your training data to `mxnet_estimator.fit()` from a S3 bucket.
