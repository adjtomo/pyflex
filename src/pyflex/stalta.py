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


def sta_lta(data, dt, min_period):
    """
    STA/LTA as used in SPECFEM. Should potentially be replace by a suitable
    STA/LTA variant in ObsPy as it is really slow right now.

    :param data: The data array.
    :param dt: The sample interval of the data.
    :param min_period: The minimum period of the data.
    """
    Cs = 10 ** (-dt / min_period)
    Cl = 10 ** (-dt / (12 * min_period))
    TOL = 1e-9

    noise = data.max() / 1E5

    # set pre-extension for synthetic data and allocate extended_syn
    n_extend = 1000 * int(min_period / dt)
    extended_syn = np.zeros(len(data) + n_extend, dtype=np.float64)
    # copy the original synthetic into the extended array, right justified
    # and add the noise level.
    extended_syn += noise
    extended_syn[-len(data):] += data

    STA_LTA = np.zeros(len(data), dtype=np.float64)

    sta = 0.0
    lta = 0.0

    # warm up the sta and lta
    for i in xrange(n_extend):
        sta = Cs * sta + extended_syn[i]
        lta = Cl * lta + extended_syn[i]

    # calculate sta/lta for the envelope
    for i in xrange(len(data)):
        sta = Cs * sta + extended_syn[n_extend + i]
        lta = Cl * lta + extended_syn[n_extend + i]
        if lta > TOL:
            STA_LTA[i] = sta / lta
    return STA_LTA
