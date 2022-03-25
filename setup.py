#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.06 #
#####################################################
"""The setup function for pypi."""
# The following is to make nats_bench avaliable on Python Package Index (PyPI)
#
# conda install -c conda-forge twine  # Use twine to upload nats_bench to pypi
#
# python setup.py sdist bdist_wheel
# python setup.py --help-commands
# twine check dist/*
#
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
# https://pypi.org/project/nats-bench
#
# NOTE(xuanyidong):
# local install = `pip install . --force`
# local test = `pytest . -s`
#
# TODO(xuanyidong): upload it to conda
#
import os
from setuptools import setup
from nats_bench import version

NAME = "nats_bench"
REQUIRES_PYTHON = ">=3.6"
DESCRIPTION = "API for NATS-Bench (a dataset/benchmark for neural architecture topology and size)."

VERSION = version()


def read(fname="README.md"):
    with open(
        os.path.join(os.path.dirname(__file__), fname), encoding="utf-8"
    ) as cfile:
        return cfile.read()


# What packages are required for this module to be executed?
REQUIRED = ["numpy>=1.16.5"]

setup(
    name=NAME,
    version=VERSION,
    author="Xuanyi Dong",
    author_email="dongxuanyi888@gmail.com",
    description=DESCRIPTION,
    license="MIT Licence",
    keywords="NAS Dataset API DeepLearning",
    url="https://github.com/D-X-Y/NATS-Bench",
    packages=["nats_bench"],
    install_requires=REQUIRED,
    python_requires=REQUIRES_PYTHON,
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
