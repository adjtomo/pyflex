#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests miscellaneous other things.

Run with pytest.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np

import pyflex


def test_local_extrema_function():
    """
    Tests the local extrema function against flats.
    """
    data = np.array([0, 1, 1, 1, 0, -1, -1, -1, 0, -1, 0, 1, 0, 0])

    maxs, mins = pyflex.utils.find_local_extrema(data)

    # It will always return the left-most value of flat extrema.
    np.testing.assert_array_equal(maxs, np.array([1, 8, 11]))
    np.testing.assert_array_equal(mins, np.array([5, 9]))
