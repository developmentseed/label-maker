# Using `label-maker` with `skynet-train`

## Background

[`skynet-data`](https://github.com/developmentseed/skynet-data/) is a tool developed specifically to prepare data for [`skynet-train`]((https://github.com/developmentseed/skynet-train/)), an implementation of [SegNet](http://mi.eng.cam.ac.uk/projects/segnet/). `skynet-data` predates `label-maker` and prepares data in a very similar way: download OpenStreetMap data and satellite imagery tiles for use in Machine Learning training. Eventually, `skynet-data` will be deprecated as most of it's functionality can be replicated using `label-maker`.

## Preparing data

`skynet-train` requires a few separate files specific to [`caffe`](https://github.com/BVLC/caffe). To create these files, we've created a [utility script](utils/skynet.py) to help connect `label-maker` with [`skynet-train`](https://github.com/developmentseed/skynet-train/). First, prepare segmentation labels and images with `label-maker` by running `download`, `labels`, and `images` from the command line, following instructions from the [other examples](README.md) or the [README](../README.md). Then, in your data folder (the script uses relative paths), run:

```bash
python utils/segnet.py
```

This should create the files (`train.txt`, `val.txt`, and `label-stats.csv`) which are needed for running `skynet-train`

## Training

Now you can mount your data folder as shown in the [`skynet-train` instructions](https://github.com/developmentseed/skynet-train/#quick-start) and training should begin.
