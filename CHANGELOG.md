[0.9.0](https://github.com/developmentseed/label-maker/releases/tag/0.9.0) (2020-08-19)
------------------
- Add the ability to parse non-polygons from the GeoJSON file ([#170](https://github.com/developmentseed/label-maker/pull/170))
- Add `over_zoom` parameter to create higher resolution tiles ([#172](https://github.com/developmentseed/label-maker/pull/172))
- Add `band_indices` parameter to select bands from input TIF file ([#178](https://github.com/developmentseed/label-maker/pull/178))
- Improve test output ([#173](https://github.com/developmentseed/label-maker/pull/173))
- Update documentation


[0.8.0](https://github.com/developmentseed/label-maker/releases/tag/0.8.0) (2020-05-13)
------------------
- Maintenance release
- Add tox for tests/automation
- Automate pypi releases
- Dependency removal (homura/pycurl/pyproj) and upgrades (rasterio/numpy)


[0.7.0](https://github.com/developmentseed/label-maker/releases/tag/0.7.0) (2019-12-13)
------------------
- Fixed bug introduced by HTTP Authentication ([#157](https://github.com/developmentseed/label-maker/pull/157) and [#161](https://github.com/developmentseed/label-maker/pull/161))
- Updated background ratio to work for multiclass problems ([#159](https://github.com/developmentseed/label-maker/pull/159))


[0.6.1](https://github.com/developmentseed/label-maker/releases/tag/0.6.1) (2019-11-11)
------------------
- Added ability to use HTTP Authentication for TMS endpoints ([#152](https://github.com/developmentseed/label-maker/pull/152))


[0.6.0](https://github.com/developmentseed/label-maker/releases/tag/0.6.0) (2019-11-06)
------------------
- Use sys.exectuable in place of python string ([#124](https://github.com/developmentseed/label-maker/pull/124))
- Correct script reference to fix bug in skynet train example ([#129](https://github.com/developmentseed/label-maker/pull/129))
- Add s3 requirement to rasterio ([#137](https://github.com/developmentseed/label-maker/pull/137))
- users can split data into more groups than train and test, for example train/test/validate, and specify the ratio for
each split ([#149](https://github.com/developmentseed/label-maker/pull/149))


[0.5.1](https://github.com/developmentseed/label-maker/releases/tag/0.5.1) (2018-11-12)
------------------
- Skip invalid or empty geometries which prevent segmentation rendering ([#118](https://github.com/developmentseed/label-maker/pull/118))
- Add binder example ([#119](https://github.com/developmentseed/label-maker/pull/119))


[0.5.0](https://github.com/developmentseed/label-maker/releases/tag/0.5.0) (2018-11-05)
------------------
- Accept GeoJSON input labels ([#32](https://github.com/developmentseed/label-maker/pull/32))
- Correct documentation regarding class labels ([#113](https://github.com/developmentseed/label-maker/pull/113))
- Small miscellaneous fixes


[0.4.0](https://github.com/developmentseed/label-maker/releases/tag/0.4.0) (2018-10-04)
------------------
- Read file drivers to determine file type rather than relying on extension ([#80](https://github.com/developmentseed/label-maker/pull/80))
- Add support for WMS endpoints as an imagery source ([#93](https://github.com/developmentseed/label-maker/pull/93))
- Add documentation site ([#108](https://github.com/developmentseed/label-maker/pull/108))
- Fix rendering errors at tile boundaries ([#78](https://github.com/developmentseed/label-maker/pull/78))
- New examples and updates to example code ([#89](https://github.com/developmentseed/label-maker/pull/89), [#91](https://github.com/developmentseed/label-maker/pull/91), [#105](https://github.com/developmentseed/label-maker/pull/105), [#107](https://github.com/developmentseed/label-maker/pull/107))


[0.3.2](https://github.com/developmentseed/label-maker/releases/tag/0.3.2) (2018-05-14)
------------------
- Provide a default value of False for imagery_offset to preview function ([#79](https://github.com/developmentseed/label-maker/pull/79))


[0.3.1](https://github.com/developmentseed/label-maker/releases/tag/0.3.1) (2018-04-19)
------------------
- Add colors for object detection and segmentation labels ([#64](https://github.com/developmentseed/label-maker/pull/64))
- Add support for `vrt` reads ([#71](https://github.com/developmentseed/label-maker/pull/71))
- Add documentation for local testing ([#75](https://github.com/developmentseed/label-maker/pull/75))
- Fix `preview` downloading one too many tiles ([#63](https://github.com/developmentseed/label-maker/pull/63))
- Fix warnings on intentionally missing tiles ([#68](https://github.com/developmentseed/label-maker/pull/68))
- Fix image and tile format inconsistency when packaging GeoTIFF ([#66](https://github.com/developmentseed/label-maker/pull/66))
- Fix function docstrings ([#61](https://github.com/developmentseed/label-maker/pull/61))


[0.3.0](https://github.com/developmentseed/label-maker/releases/tag/0.3.0) (2018-03-29)
------------------
- Add optional `imagery_offset` property to align imagery with label data ([#58](https://github.com/developmentseed/label-maker/pull/58))
- Generate preview tiles faster ([#30](https://github.com/developmentseed/label-maker/pull/30))
- Add support for reading GeoTIFF as the imagery source ([#13](https://github.com/developmentseed/label-maker/pull/13))
- Refactor testing structure ([#29](https://github.com/developmentseed/label-maker/pull/29))
- Bug fix: fix logic for matching the correct tiles when creating segmentation
  labels with the --sparse flag ([#46](https://github.com/developmentseed/label-maker/pull/46))


[0.2.1](https://github.com/developmentseed/label-maker/releases/tag/0.2.1) (2018-02-24)
------------------
- Lower memory usage of stream_filter.py ([#39](https://github.com/developmentseed/label-maker/pull/39))
- Bug fix: print correct object detection labeling summary ([#33](https://github.com/developmentseed/label-maker/pull/33))
- Bug fix: uncompress mbtiles line by line to prevent memory usage issues
  causing large files to fail on the `download` subcommand ([#35](https://github.com/developmentseed/label-maker/pull/35))


[0.2.0](https://github.com/developmentseed/label-maker/releases/tag/0.2.0) (2018-01-19)
------------------
- Add optional `buffer` property to classes to create more accurate
object-detection or segmentation labels ([#10](https://github.com/developmentseed/label-maker/pull/10)).
- Add --sparse flag to limit the size of labels.npz file ([#16](https://github.com/developmentseed/label-maker/pull/16)).
- Add more globally ignored statements to pylint settings ([#24](https://github.com/developmentseed/label-maker/pull/24)).
- Bug fix: correct a variable name in package.py which prevented
  object-detection packaging from running ([#19](https://github.com/developmentseed/label-maker/pull/19)).


[0.1.2](https://github.com/developmentseed/label-maker/releases/tag/0.1.2) (2018-01-11)
------------------
- Bug fix: resolve path issues which prevented it from working outside the
  github cloned repository ([#2](https://github.com/developmentseed/label-maker/pull/2)).


[0.1](https://github.com/developmentseed/label-maker/releases/tag/0.1) (2018-01-10)
------------------
- Initial Release
