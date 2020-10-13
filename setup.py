#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.06 #
#####################################################
"""The setup function for pypi."""
# [2020.08.31] v1.0
import os
from setuptools import setup


def read(fname="README.md"):
  with open(
      os.path.join(os.path.dirname(__file__), fname),
      encoding="utf-8") as cfile:
    return cfile.read()


setup(
    name="nats_bench",
    version="1.0",
    author="Xuanyi Dong",
    author_email="dongxuanyi888@gmail.com",
    description="API for NATS-Bench (a dataset for neural architecture topology and size).",
    license="MIT",
    keywords="NAS Dataset API DeepLearning",
    url="https://github.com/D-X-Y/NATS-Bench",
    packages=["nats_bench"],
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
