Parameters
----------
Here is the full list of configuration parameters you can specify in a ``config.json`` file.

**country**: string
	The `OSM QA Tile <https://osmlab.github.io/osm-qa-tiles/>`_ extract to download. The string should be a country matching a one of the options in ``label_maker/countries.txt``

**bounding_box**: list of floats
	The bounding box to create images from. This should be given in the form: ``[xmin, ymin, xmax, ymax]`` as longitude and latitude values between ``[-180, 180]`` and ``[-90, 90]``, respectively. Values should use the WGS84 datum, with longitude and latitude units in decimal degrees.

**geojson**: string
	An input ``GeoJSON`` file containing labels. Adding this parameter will override the values in the ``country`` and ``bounding_box`` parameters. The ``GeoJSON`` should only contain `Polygon` and not `Multipolygon` or a `GeometryCollection`.

**zoom**: int
	The `zoom level <http://wiki.openstreetmap.org/wiki/Zoom_levels>`_ used to create images. This functions as a rough proxy for resolution. Value should be given as an int on the interval [0, 19].

**classes**: list of dicts
	The training classes. Each class is defined as dict object with two required keys:

 	**name**: string
 		The class name.
 	**filter**: list of strings
 		A `Mapbox GL Filter <https://www.mapbox.com/mapbox-gl-js/style-spec#other-filter>`_ to define any vector features matching this class. Filters are applied with the standalone `featureFilter <https://github.com/mapbox/mapbox-gl-js/tree/master/src/style-spec/feature_filter>`_ from Mapbox GL JS.

**buffer**: int
	Optional paramter to buffer labels in ``'object-detection'`` and ``'segmentation'`` tasks by an arbitrary number of pixels. Accepts both positive and negative integers. It uses `Shapely object.buffer <https://shapely.readthedocs.io/en/latest/manual.html#object.buffer>`_ to calculate the final geometry. You can verify that your buffer options create the desired labels by inspecting the files created in ``data/labels/`` after running the ``label-maker labels`` command.

**imagery**: string
	Label Maker expects to receive imagery tiles that are 256 x 256 pixels. You can specific the source of the imagery with one of:

 		A template string for a tiled imagery service. Note that you will generally need an API key to obtain images and there may be associated costs. The above example requires a `Mapbox access token <https://www.mapbox.com/help/how-access-tokens-work/>`_. Also see `OpenAerialMap <https://openaerialmap.org/>`_ for open imagery.

 		A GeoTIFF file location. Works with local files: ``'http://oin-hotosm.s3.amazonaws.com/593ede5ee407d70011386139/0/3041615b-2bdb-40c5-b834-36f580baca29.tif'``

 		Remote files like a `WMS endpoint <http://www.opengeospatial.org/standards/wms>`_ ``GetMap`` request. Fill out all necessary parameters except ``bbox`` which should be set as ``{bbox}``. Ex: ``'https://basemap.nationalmap.gov/arcgis/services/USGSImageryOnly/MapServer/WMSServer?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1&LAYERS=0&STYLES=&FORMAT=image%2Fjpeg&TRANSPARENT=false&HEIGHT=256&WIDTH=256&SRS=EPSG%3A3857&BBOX={bbox}'``

**background_ratio**: float
	Specify how many background (or "negative") training examples to create when there is only one class specified with the ``classes`` parameter. Label Maker will generate ``background_ratio`` times the number of images matching the one class.

**ml_type**: string
	One of ``'classification'``, ``'object-detection'``, or ``'segmentation'``. This defines the output format for the final label numpy arrays (``y_train`` and ``y_test``).

 	``'classification'``
 		Output is an array of ``len(classes) + 1``. Each array value will be either `1` or `0` based on whether it matches the class at the same index. The additional array element belongs to the background class, which will always be the first element. 

 	``'object-detection'``
 		Output is an array of bounding boxes of the form ``[xmin, ymin, width, height, class_index]``. In this case, the values are pixel values measured from the upper left-hand corner (not latitude and longitude values). Each feature is tested against each class, so if a feature matches two or more classes, it will have the corresponding number of bounding boxes created.

 	``'segmentation'``
 		Output is an array of shape ``(256, 256)`` with values matching the class index label at that position. The classes are applied sequentially according to ``config.json`` so latter classes will be written over earlier class labels if there is overlap.

**imagery_offset**:  list of ints
	An optional list of integers representing the number of pixels to offset imagery. For example ``[15, -5]`` will move the images 15 pixels right and 5 pixels up relative to the requested tile bounds.
