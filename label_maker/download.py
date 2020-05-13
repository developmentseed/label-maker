# pylint: disable=unused-argument
"""Download QA Tiles for the selected country."""
from os import path
import tempfile
import gzip
import requests
from tqdm import tqdm

def download(url, path):
    """Download url to target path"""
    file_size = int(requests.head(url).headers["Content-Length"])
    header = {"Range": "bytes=%s-%s" % (0, file_size)}
    pbar = tqdm(
        total=file_size,
        unit='B',
        unit_scale=True,
        desc=url.split('/')[-1]
    )
    req = requests.get(url, headers=header, stream=True)
    with(open(path, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size

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
