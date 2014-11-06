#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import collections

import obspy.signal
import obspy.signal.trigger

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
    pass
