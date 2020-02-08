# ==============================================================================
# PyCrown - Fast raster-based individual tree segmentation for LiDAR data
# ------------------------------------------------------------------------------
# Copyright: 2018, Jan Zörner
# Licence: GNU GPLv3
# ==============================================================================

import os
import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().splitlines()

extensions = [
    Extension(
        'pycrown._crown_dalponte_cython',
        sources=['pycrown/_crown_dalponte_cython.pyx'],
        include_dirs=[np.get_include()],
        optional=True
    )
]

setup(
    name='PyCrown',
    ext_modules=cythonize(extensions),
    version='0.2',
    test_suite='tests',
    packages=['pycrown'],
    install_requires=requirements,
    license='MIT',
    author='Dr. Jan Zörner',
    author_email='zoernerj@landcareresearch.co.nz',
    description="Fast Raster-based Tree Crown Segmentation"
)
