#!/usr/bin/env python
# -*- encoding: utf8 -*-
import glob
import io
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")).read()

setup(
    name="pyflex",
    version="0.0.1a",
    license='GNU General Public License, Version 3 (GPLv3)',
    description="Python port of the FLEXWIN package",
    author="Lion Krischer",
    author_email="krischer@geophysik.uni-muenchen.de",
    url="https://github.com/krischer/pyflex",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(i))[0] for i in glob.glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Utilities",
    ],
    keywords=[
        "seismology", "flexwin", "science", "tomography", "inversion"
    ],
    install_requires=[
        "obspy", "flake8", "pytest", "nose", "future"
    ],
    extras_require={
        "docs": ["sphinx", "ipython", "runipy"]
    }
)
