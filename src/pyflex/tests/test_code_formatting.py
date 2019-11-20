#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests all Python files of the project with flake8. This ensure PEP8 conformance
and some other sanity checks as well.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013-2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from flake8.api import legacy
import inspect
import os


def test_flake8():
    test_dir = os.path.dirname(os.path.abspath(inspect.getfile(
        inspect.currentframe())))
    pyflex_dir = os.path.dirname(test_dir)

    # Possibility to ignore some files and paths.
    ignore_paths = [
        os.path.join(pyflex_dir, "doc"),
        os.path.join(pyflex_dir, ".git")]
    files = []
    for dirpath, _, filenames in os.walk(pyflex_dir):
        ignore = False
        for path in ignore_paths:
            if dirpath.startswith(path):
                ignore = True
                break
        if ignore:
            continue
        filenames = [_i for _i in filenames if
                     os.path.splitext(_i)[-1] == os.path.extsep + "py"]
        if not filenames:
            continue
        for py_file in filenames:
            full_path = os.path.join(dirpath, py_file)
            files.append(full_path)

    # Get the style checker with the default style.
    flake8_style = legacy.get_style_guide()

    report = flake8_style.check_files(files)

    # And no errors occured.
    assert report.total_errors == 0
