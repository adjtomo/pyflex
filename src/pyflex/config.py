#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration object for pyflex.

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

from . import PyflexError


class Config(object):
    def __init__(self, min_period, max_period, stalta_waterlevel=0.07,
                 tshift_acceptance_level=10.0, tshift_reference=0.0,
                 dlna_acceptance_level=1.3, dlna_reference=0.0,
                 cc_acceptance_level=0.7, earth_model="ak135",
                 min_surface_wave_velocity=3.0, c_0=1.0, c_1=1.5, c_2=0.0,
                 c_3a=4.0, c_3b=2.5, c_4a=2.0, c_4b=6.0):
        """
        Central configuration object for Pyflex.

        This replaces the old PAR_FILEs and user functions. It has sensible
        defaults for most values but you will probably want to adjust them
        for your given application.

        :param min_period: Minimum period of the filtered input data in
            seconds.
        :type min_period: float

        :param max_period: Maximum period of the filtered input data in
            seconds.
        :type max_period: float

        :param stalta_waterlevel: Water level on the STA/LTA functional. Can
            be either a single value or an array with the same number of
            samples as the data.
        :type stalta_waterlevel: float or :class:`numpy.ndarray`

        :param tshift_acceptance_level:  Maximum allowable cross correlation
            time shift/lag relative to the reference. Can be either a single
            value or an array with the same number of samples as the data.
        :type tshift_acceptance_level: float or :class:`numpy.ndarray`

        :param tshift_reference: Allows a systematic shift of the cross
            correlation time lag.
        :type tshift_reference: float

        :param dlna_acceptance_level: Maximum allowable amplitude ratio
            relative to the reference. Can be either a single value or an
            array with the same number of samples as the data.
        :type dlna_acceptance_level: float or :class:`numpy.ndarray`

        :param dlna_reference: Reference DLNA level. Allows a systematic shift.
        :type dlna_reference: float

        :param cc_acceptance_level: Limit on the normalized cross correlation
            per window. Can be either a single value or an array with the
            same number of samples as the data.
        :type cc_acceptance_level: float or :class:`numpy.ndarray`

        :param earth_model: The earth model used for the traveltime
            calculations. Either ``"ak135"`` or ``"iasp91"``.
        :type earth_model: str

        :param min_surface_wave_velocity: The minimum surface wave velocity
            in km/s. All windows containing data later then this velocity
            will be rejected. Only used if station and event information is
            available.
        :type min_surface_wave_velocity: float

        :param c_0: Fine tuning constant for the rejection of windows based
            on the height of internal minima. Any windows with internal
            minima lower then this value times the STA/LTA water level at
            the window peak will be rejected.
        :type c_0: float

        :param c_1: Fine tuning constant for the minimum acceptable window
            length. This value multiplied by the minimum period will be the
            minimum acceptable window length.
        :type c_1: float

        :param c_2: Fine tuning constant for the maxima prominence
            rejection. Any windows whose minima surrounding the central peak
            are smaller then this value times the central peak will be
            rejected. This value is set to 0 in many cases as it is hard to
            control.
        :type c_2: float

        :param c_3a: Fine tuning constant for the separation height in the
            phase separation rejection stage.
        :type c_3a: float

        :param c_3b: Fine tuning constant for the separation time used in
            the decay function in the phase separation rejection stage.
        :type c_3b: float

        :param c_4a: Fine tuning constant for curtailing windows on the left
            with emergent start/stops and/or codas.
        :type c_4a: float

        :param c_4b: Fine tuning constant for curtailing windows on the right
            with emergent start/stops and/or codas.
        :type c_4b: float
        """
        self.min_period = min_period
        self.max_period = min_period

        self.stalta_waterlevel = stalta_waterlevel
        self.tshift_acceptance_level = tshift_acceptance_level
        self.tshift_reference = tshift_reference
        self.dlna_acceptance_level = dlna_acceptance_level
        self.dlna_reference = dlna_reference
        self.cc_acceptance_level = cc_acceptance_level

        if earth_model.lower() not in ("ak135", "iasp91"):
            raise PyflexError("Earth model must either be 'ak135' or "
                              "'iasp91'.")
        self.earth_model = earth_model.lower()
        self.min_surface_wave_velocity = min_surface_wave_velocity

        self.c_0 = c_0
        self.c_1 = c_1
        self.c_2 = c_2
        self.c_3a = c_3a
        self.c_3b = c_3b
        self.c_4a = c_4a
        self.c_4b = c_4b

    def _convert_to_array(self, npts):
        """
        Internally converts the acceptance and water levels to arrays. Not
        called during initialization as the array length is not yet known.
        """
        attributes = ("stalta_waterlevel", "tshift_acceptance_level",
                      "dlna_acceptance_level", "cc_acceptance_level")
        for name in attributes:
            attr = getattr(self, name)

            if isinstance(attr, collections.Iterable):
                if len(attr) != npts:
                    raise PyflexError(
                        "Config value '%s' does not have the same number of "
                        "samples as the waveforms." % name)
                setattr(self, name, np.array(attr))
                continue

            setattr(self, name, attr * np.ones(npts))
