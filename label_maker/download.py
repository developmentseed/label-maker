# pylint: disable=unused-argument
"""Download QA Tiles for the selected country."""
from os import path
import tempfile
import gzip
from homura import download


def download_mbtiles(dest_folder, country, **kwargs):
    """Download QA Tiles for the selected country.

    Download a gzipped mbtiles file of all OSM data within a country from S3.
    More details at https://osmlab.github.io/osm-qa-tiles/

    Parameters
    ------------
    dest_folder: str
        Folder to save download into
    country: str
        Country for which to download the OSM QA tiles
    **kwargs: dict
        Other properties from CLI config passed as keywords to other utility functions
    """
    download_file = path.join(dest_folder, '{}.mbtiles'.format(country))
    print('Saving QA tiles to {}'.format(download_file))
    url = 'https://s3.amazonaws.com/mapbox/osm-qa-tiles-production/latest.country/{}.mbtiles.gz'.format(country)
    gz = tempfile.TemporaryDirectory()
    tmp_path = path.join(gz.name, '{}.mbtiles.gz'.format(country))
    download(url=url, path=tmp_path)
    with gzip.open(tmp_path, 'rb') as r:
        with open(download_file, 'wb') as w:
            for line in r:
                w.write(line)
