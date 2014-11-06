#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import collections
import numpy as np

import obspy
import obspy.signal
import obspy.signal.trigger

from . import PyflexError

import warnings

Event = collections.namedtuple("Event",
                               ["latitude", "longitude", "depth_in_m"])

Station = collections.namedtuple("Station", ["latitude", "longitude"])


def select_windows(observed, synthetic, min_period, max_period):
    """
    Function picking windows.

    :param observed: A preprocessed :class:`~obspy.core.trace.Trace` object
        containing the observed data.
    :param observed: A preprocessed :class:`~obspy.core.trace.Trace` object
        containing the synthetic data.
    :param min_period: The minimum period of the already processed data and
        synthetics.
    :param max_period: The maximum period of the already processed data and
        synthetics.
    """
    # Copy to not modify the original data.
    observed = observed.copy()
    synthetic = synthetic.copy()
    _sanity_checks(observed, synthetic)

    # STA/LTA of the synthetics
    STA_LTA = obspy.signal.trigger.recSTALTA(synthetic, 100, 10)


def _sanity_checks(observed, synthetic):
    """
    Perform a number of basic sanity checks to assure the data is valid in a
    certain sense.

    It checks the types of both, the starttime, sampling rate, number of
    samples, ...

    :param observed:
    :param synthetic:
    :return:
    """
    if not isinstance(observed, obspy.Trace):
        raise PyflexError("Observed data must be an ObsPy Trace object.")
    if not isinstance(synthetic, obspy.Trace):
        raise PyflexError("Synthetic data must be an ObsPy Trace object.")

    if observed.stats.npts != synthetic.stats.npts:
        raise PyflexError("Observed and synthetic data must have the same "
                          "number of samples.")

    sr1 = observed.stats.sampling_rate
    sr2 = synthetic.stats.sampling_rate

    if abs(sr1 - sr2) / sr1 >= 1E-5:
        raise PyflexError("Observed and synthetic data must have the same "
                          "sampling rate.")

    if observed.stats.starttime != synthetic.stats.starttime:
        raise PyflexError("Observed and synthetic data must have the same "
                          "starttime.")

    ptp = sorted([observed.data.ptp(), synthetic.data.ptp()])
    if ptp[1] / ptp[0] >= 5:
        warnings.warn("The amplitude difference between data and synthetic "
                      "is fairly large.")

    # Also check the components of the data to avoid silly mistakes of users.
    if len(set(observed.stats.channel[-1].upper(),
               synthetic.stats.channel[-1].upper())) != 1:
        warnings.warn("The orientation code of synthetic and observed data "
                      "is not equal.")

    observed.data = np.ascontiguousarray(observed.data, dtype=np.float64)
    synthetic.data = np.ascontiguousarray(synthetic.data, dtype=np.float64)
