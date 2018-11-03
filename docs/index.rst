.. label-maker documentation master file, created by
   sphinx-quickstart on Sun Sep 16 11:05:39 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Label Maker Documentation
#########################

Label Maker generates training data for ML algorithms focused on overhead imagery (e.g., from satellites or drones). It downloads OpenStreetMap QA Tile information and overhead imagery tiles and saves them as an Numpy `.npz <https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html>`_ file for easy use in ML pipelines. For more details, see the `inaugural blog post <https://developmentseed.org/blog/2018/01/11/label-maker/>`_.

Requirements
============
* `Python 3.6 <https://www.python.org/>`_
* `tippecanoe <https://github.com/mapbox/tippecanoe>`_

Standard pip install
====================

.. code-block:: bash

   pip install label-maker

.. note::

	Label Maker requires ``tippecanoe`` to be available from your command-line. Confirm this before proceeding.

Configuration
=============
Before you can use Label Maker, you must specify inputs to the data-creation process within ``config.json`` file. Below is a simple example. To see the complete list of parameters and options for imagery access, check out the `parameters page <parameters.html>`_.

.. code-block:: json

	{
	  "country": "togo",
	  "bounding_box": [1.09725, 6.05520, 1.34582, 6.30915],
	  "zoom": 12,
	  "classes": [
	    { "name": "Roads", "filter": ["has", "highway"] },
	    { "name": "Buildings", "filter": ["has", "building"] }
	  ],
	  "imagery": "http://a.tiles.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.jpg?access_token=ACCESS_TOKEN",
	  "background_ratio": 1,
	  "ml_type": "classification"
	}

Before using this configuration, make sure to replace ``ACCESS_TOKEN`` with your `Mapbox Access Token <https://www.mapbox.com/help/how-access-tokens-work/>`_

Command line interface (CLI)
============================

Label Maker is most easily used as a command line tool. There are five commands documented below. You should run them in order as each operation builds on the previous one and commands accept two flags:

	``-d`` or ``--dest``: string
		Directory for storing output files. Defaults to ``'./data'``
	``-c`` or ``--config``: string
		Location of ``config.json`` file. Defaults to ``'./config.json'``

CLI Step 1: download
^^^^^^^^^^^^^^^^^^^^
Download and unzip OSM QA tiles containing feature information.

.. code-block:: bash

	$ label-maker download
	Saving QA tiles to data/ghana.mbtiles
	   100%     18.6 MiB       1.8 MiB/s            0:00:00 ETA

CLI Step 2: labels
^^^^^^^^^^^^^^^^^^
Retiles the OSM data to the desired zoom level, creates label data (``labels.npz``), calculates class statistics, creates visual label files (either GeoJSON or PNG files depending upon ``ml_type``). Requires the mbtiles file from the ``label-maker download`` step.

Accepts one additional flag:

	``-s`` or ``--sparse``: boolean 
		Specifies if features in the class of interest are sparse. If ``True``, only save labels for up to ``n`` background tiles, where ``n`` is equal to ``background_ratio`` times the number of tiles with a class label. Defaults to ``False``.

.. code-block:: bash

	$ label-maker labels
	Determining labels for each tile
	---
	Residential: 638 tiles
	Total tiles: 1189
	Write out labels to data/labels.npz

CLI Step 3: preview (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Downloads example overhead images for each class. Requires the ``labels.npz`` file from the ``label-maker labels`` step.

Accepts one additional flag:

	``-n`` or ``--number``: int
		Specifies the number of examples images to create per class. Defaults to ``5``.

.. code-block:: bash

	$ label-maker preview -n 10
	Writing example images to data/examples
	Downloading 10 tiles for class Residential

CLI Step 4: images
^^^^^^^^^^^^^^^^^^

Downloads all imagery tiles needed to create the training data. Requires the ``labels.npz`` file from the ``label-maker labels`` step.

.. code-block:: bash

	$ label-maker images
	Downloading 1189 tiles to data/tiles

CLI Step 5: package
^^^^^^^^^^^^^^^^^^^
Bundles the images and OSM labels to create a final ``data.npz`` file. Requires the ``labels.npz`` file from the ``label-maker labels`` step and downloaded image tiles from the ``label-maker images`` step.

.. code-block:: bash

	$ label-maker package
	Saving packaged file to data/data.npz

Using the packaged data
=======================
Once you have a create ``data.npz`` file using the above commands, you can use `numpy.load <https://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html>`_ to load it. For example, you can supply the created data to a `Keras <https://keras.io/>`_ ``Model`` as follows:

.. code-block:: bash

	# Load the data, shuffled and split between train and test sets
	npz = np.load('data.npz')
	x_train = npz['x_train']
	y_train = npz['y_train']
	x_test = npz['x_test']
	y_test = npz['y_test']

	# Define your model here, example usage in Keras
	model = Sequential()
	# ...
	model.compile(...)

	# Train
	model.fit(x_train, y_train, batch_size=16, epochs=50)
	model.evaluate(x_test, y_test, batch_size=16)

For more detailed walkthroughs, see the `examples page <examples.html>`_.

Acknowledgements
================

This library builds on the concepts of `skynet-data <https://github.com/developmentseed/skynet-data>`_. It wouldn't be possible without the excellent data from OpenStreetMap and Mapbox under the following licenses:

* OSM QA tile data `copyright OpenStreetMap contributors <http://www.openstreetmap.org/copyright>`_ and licensed under `ODbL <http://opendatacommons.org/licenses/odbl/>`_.
* Mapbox Satellite data can be `traced for noncommercial purposes <https://www.mapbox.com/tos/#%5BYmtMIywt%5D>`_.
* Marc Farra's `tilepie <https://github.com/kamicut/tilepie>`_ to asynchronously process vector tiles.


.. toctree::
   :hidden:
   :maxdepth: 2

   parameters
   examples
   contributing
