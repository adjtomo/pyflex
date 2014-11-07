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

from . import PyflexError


class Config(object):
    def __init__(self, min_period, max_period, stalta_base=0.07,
                 tshift_base=10.0, tshift_reference=0.0, dlna_base=1.3,
                 dlna_reference=0.0, cc_base=0.7, earth_model="ak135",
                 min_surface_wave_velocity=3.0,
                 c_0=1.0, c_1=1.5, c_2=0.0, c_3a=4.0, c_3b=2.5, c_4a=2.0,
                 c_4b=6.0):
        """
        Configuration object for pyflex to have one place to store it and to
        give meaningful default values.

        :param min_period: Minimum period of the filtered data in seconds.
        :type min_period: float
        :param max_period: Maximum period of the filtered data in seconds.
        :type max_period: float
        :param stalta_base: Base water level on the STA/LTA functional
        :type stalta_base: float
        :param tshift_base: Maximum allowable time shift relative to the
            reference.
        :type tshift_base: float
        :param tshift_reference: Allows a systematic shift of the CC time
            shift.
        :type tshift_reference: float
        :param dlna_base: Maximum allowable amplitude measurement relative
            to the refernce.
        :type dlna_base: float
        :param dlna_reference: Reference DLNA level. Allows a systematic shift.
        :type dlna_reference: float
        :param cc_base: Limit on CC for acceptable windows.
        :type cc_base: float
        :param earth_model: The earth model used for the traveltime
            calculations. Either ``"ak135"`` or ``"iasp91"``.
        :type earth_model: str
        :param min_surface_wave_velocity: The minimum surface wave velocity
            in km/s. All windows containing data later then this velocity
            will be rejected. Only used if station and event information is
            available.
        :type min_surface_wave_velocity: float
        :param c_0: Fine tuning constant for the window internal minima.
        :type c_0: float
        :param c_1: Fine tuning constant for too small window lengths.
        :type c_1: float
        :param c_2: Fine tuning constant for the maxima prominence rejection.
        :type c_2: float
        :param c_3a: Fine tuning constant for the separation height.
        :type c_3a: float
        :param c_3b: Fine tuning constant for the separation time.
        :type c_3b: float
        :param c_4a: Fine tuning constant for curtail on the left.
        :type c_4a: float
        :param c_4b: Fine tuning constant for curtail on the right.
        :type c_4b: float
        """
        self.min_period = min_period
        self.max_period = min_period
        self.stalta_base = stalta_base
        self.tshift_base = tshift_base
        self.tshift_reference = tshift_reference
        self.dlna_base = dlna_base
        self.dlna_reference = dlna_reference
        self.cc_base = cc_base

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
