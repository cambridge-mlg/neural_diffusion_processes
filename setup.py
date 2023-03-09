# from setuptools import setup, find_packages

# setup(
#   name="neural-diffusion-processes",
#   version="0.1",
#   packages=find_packages(
#     exclude=[]
#   ),
# )

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_namespace_packages, setup

with open("requirements.txt", "r") as file:
    requirements = [line.strip() for line in file]

with open("README.md", "r") as file:
    long_description = file.read()

with open("VERSION", "r") as file:
    version = file.read().strip()

setup(
    name="neural-diffusion-processes",
    version=version,
    author="Emile Mathieu, Vincent Dutordoir",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="",
    # license="Apache License 2.0",
    keywords="",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
