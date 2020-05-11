#!/usr/bin/env python
from os import path as op
import io
from imp import load_source
from setuptools import setup, find_packages

__version__ = load_source('label_maker.version', 'label_maker/version.py').__version__

here = op.abspath(op.dirname(__file__))

extra_reqs = {
    "test": ["pytest", "pytest-cov"],
    "dev": ["pytest", "pytest-cov", "pre-commit"],
}

# get the dependencies and installs
with io.open(op.join(here, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = f.read().split('\n')

# readme
with open('README.md') as f:
    readme = f.read()

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
    extras_require=extra_reqs,
    long_description=readme,
    long_description_content_type="text/markdown"
)
