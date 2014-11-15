#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The same STA/LTA as used in Flexwin.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import numpy as np
from scipy.signal import lfilter


def sta_lta(data, dt, min_period):
    """
    STA/LTA as used in SPECFEM.

    :param data: The data array.
    :param dt: The sample interval of the data.
    :param min_period: The minimum period of the data.
    """
    Cs = 10 ** (-dt / min_period)
    Cl = 10 ** (-dt / (12 * min_period))
    TOL = 1e-9

    noise = data.max() / 1E5

    # 1000 samples should be more then enough to "warm up" the STA/LTA.
    extended_syn = np.zeros(len(data) + 1000, dtype=np.float64)
    # copy the original synthetic into the extended array, right justified
    # and add the noise level.
    extended_syn += noise
    extended_syn[-len(data):] += data

    # This piece of codes "abuses" SciPy a bit by "constructing" an IIR
    # filter that does the same as the decaying sum and thus avoids the need to
    # write the loop in Python. The result is a speedup of up to 2 orders of
    # magnitude in common cases without needing to write the loop in C which
    # would have a big impact in the ease of installation of this package.
    # Other than that its quite a cool little trick.
    a = [1.0, -Cs]
    b = [1.0]
    sta = lfilter(b, a, extended_syn)

    a = [1.0, -Cl]
    b = [1.0]
    lta = lfilter(b, a, extended_syn)

    # STA is now STA_LTA
    sta /= lta

    # Apply threshold to avoid division by very small values.
    sta[lta < TOL] = noise
    return sta[-len(data):]
