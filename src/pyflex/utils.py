#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functionality for pyflex.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future import standard_library

import numpy as np
from scipy.signal import argrelextrema

with standard_library.hooks():
    import itertools


def find_local_extrema(data):
    """
    Function finding local extrema. It can also deal with flat extrema,
    e.g. a flat top or bottom. In that case the first index of all flat
    values will be returned.

    Returns a tuple of maxima and minima indices.
    """
    diff = np.diff(data)
    flats = np.argwhere(diff == 0)

    # Discard neighbouring flat points.
    new_flats = list(flats[0:1])
    for i, j in zip(flats[:-1], flats[1:]):
        if j - i == 1:
            continue
        new_flats.append(j)
    flats = new_flats

    maxima = []
    minima = []

    # Go over each flats position and check if its a maxima/minima.
    for idx in flats:
        l_type = "left"
        r_type = "right"
        for i in itertools.count():
            this_idx = idx - i - 1
            if diff[this_idx] < 0:
                l_type = "minima"
                break
            elif diff[this_idx] > 0:
                l_type = "maxima"
                break
        for i in itertools.count():
            this_idx = idx + i + 1
            if this_idx >= len(diff):
                break
            if diff[this_idx] < 0:
                r_type = "maxima"
                break
            elif diff[this_idx] > 0:
                r_type = "minima"
                break
        if r_type != l_type:
            continue
        if r_type == "maxima":
            maxima.append(int(idx))
        else:
            minima.append(int(idx))

    maxs = set(list(argrelextrema(data, np.greater)[0]))
    mins = set(list(argrelextrema(data, np.less)[0]))

    return np.array(sorted(list(maxs.union(set(maxima)))), dtype="int32"), \
        np.array(sorted(list(mins.union(set(minima)))), dtype="int32")
