#!/usr/bin/env python
from os import path as op
import io
from imp import load_source
from setuptools import setup, find_packages

__version__ = load_source('label_maker.version', 'label_maker/version.py').__version__

here = op.abspath(op.dirname(__file__))

# get the dependencies and installs
with io.open(op.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if 'git+' not in x]

setup(
    name='label-maker',
    author='Drew Bollinger',
    author_email='drew@developmentseed.org',
    version=__version__,
    description='Data preparation for satellite machine learning',
    url='https://github.com/developmentseed/label-maker/',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6'
    ],
    keywords='',
    entry_points={
        'console_scripts': ['label-maker=label_maker.main:cli'],
    },
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    install_requires=install_requires,
    dependency_links=dependency_links,
)
