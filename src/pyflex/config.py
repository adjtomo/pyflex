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
                 cc_acceptance_level=0.7, s2n_limit=1.5, earth_model="ak135",
                 min_surface_wave_velocity=3.0,
                 max_time_before_first_arrival=50.0,
                 c_0=1.0, c_1=1.5, c_2=0.0,
                 c_3a=4.0, c_3b=2.5, c_4a=2.0, c_4b=6.0,
                 check_global_data_quality=False, snr_integrate_base=3.5,
                 snr_max_base=3.0, noise_start_index=0, noise_end_index=None,
                 signal_start_index=None, signal_end_index=-1,
                 window_weight_fct=None,
                 window_signal_to_noise_type="amplitude",
                 resolution_strategy="interval_scheduling"):
        """
        Central configuration object for Pyflex.

        This replaces the old PAR_FILEs and user functions. It has sensible
        defaults for most values but you will probably want to adjust them
        for your given application.

        If necessary, the acceptance levels/limits can also be set as
        arrays. Each array must have the same number of samples as the
        observed and synthetic data. The corresponding values are then
        evaluated seperately at each necessary point. This enables the full
        emulation of the user functions in the original FLEXWIN code. The
        following basic example illustrates the concept which can become
        arbitrarily complex.

        .. code-block:: python

            stalta_waterlevel = 0.08 * np.ones(npts)
            tshift_acceptance_level = 15.0 * np.ones(npts)
            dlna_acceptance_level = 1.0 * np.ones(npts)
            cc_acceptance_level = 0.80 * np.ones(npts)
            s2n_limit = 1.5 * np.ones(npts)

            # Double all values from a certain index on.
            stalta_waterlevel[2500:] *= 2.0
            tshift_acceptance_level[2500:] += 5.0
            dlna_acceptance_level[2500:] *= 1.5
            cc_acceptance_level[2500:] += 0.5
            s2n_limit[2500:] *= 0.9

            config = Config(
                min_period=10.0, max_period=50.0,
                stalta_waterlevel=stalta_waterlevel,
                tshift_acceptance_level=tshift_acceptance_level,
                dlna_acceptance_level=dlna_acceptance_level,
                cc_acceptance_level=cc_acceptance_level,
                s2n_limit=s2n_limit)


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

        :param s2n_limit: Limit of the signal to noise ratio per window. If
            the maximum amplitude of the window over the maximum amplitude
            of the global noise of the waveforms is smaller than this
            window, then it will be rejected. Can be either a single value
            or an array with the same number of samples as the data.
        :type s2n_limit: float or :class:`numpy.ndarray`

        :param earth_model: The earth model used for the traveltime
            calculations. Either ``"ak135"`` or ``"iasp91"``.
        :type earth_model: str

        :param min_surface_wave_velocity: The minimum surface wave velocity
            in km/s. All windows containing data later then this velocity
            will be rejected. Only used if station and event information is
            available.
        :type min_surface_wave_velocity: float

        :param max_time_before_first_arrival: This is the minimum starttime
            of any window in seconds before the first arrival. No windows will
            have a starttime smaller than this.
        :type max_time_before_first_arrival: float

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

        :param check_global_data_quality: Determines whether or not to check
            the signal to noise ratio of the whole observed waveform. If
            True, no windows will be selected if the signal to noise ratio
            is above the thresholds.
        :param snr_integrate_base: Minimal SNR ratio. If the squared sum of
            the signal normalized by its length over the squared sum of the
            noise normalized by its length is smaller then this value,
            no windows will be chosen for the waveforms. Only used if
            ``check_global_data_quality`` is ``True``.
        :type snr_integrate_base: float
        :param snr_max_base: Minimal amplitude SNR ratio. If the maximum
            amplitude of the signal over the maximum amplitude of the noise
            is smaller than this value no windows will be chosen for the
            waveforms. Only used if  ``check_global_data_quality`` is ``True``.
        :type snr_max_base: float
        :param noise_start_index: Index in the observed data where noise
            starts for the signal to noise calculations.
        :type noise_start_index: int
        :param noise_end_index: Index in the observed data where noise
            ends for the signal to noise calculations. Will be set to the
            time of the first theoretical arrival minus the minimum period
            if not set and event and station information is available.
        :type noise_end_index: int
        :param signal_start_index: Index where the signal starts for the signal
            to noise calculations. Will be set to to the noise end index if
            not given.
        :type signal_start_index: int
        :param signal_end_index: Index where the signal ends for the signal
            to noise calculations.
        :type signal_end_index: int

        :param window_weight_fct: A function returning the weight for a
            specific window as a single number. Directly passed to the
            :class:`~pyflex.window.Window` 's initialization function.
        :type window_weight_fct: function

        :param window_signal_to_noise_type: The type of signal to noise
            ratio used to reject windows. If ``"amplitude"``, then the
            largest amplitude before the arrival is the noise amplitude and
            the largest amplitude in the window is the signal amplitude. If
            ``"energy"`` the time normalized energy is used in both cases.
            The later one is a bit more stable when having random wiggles
            before the first arrival.
        :type window_signal_to_noise_type: str

        :param resolution_strategy: Strategy used to resolve overlaps.
            Possibilities are ``"interval_scheduling"`` and ``"merge"``.
            Interval scheduling will chose the optimal subset of
            non-overlapping windows. Merging will simply merge overlapping
            windows.
        :type resolution_strategy: str
        """
        self.min_period = min_period
        self.max_period = max_period

        self.stalta_waterlevel = stalta_waterlevel
        self.tshift_acceptance_level = tshift_acceptance_level
        self.tshift_reference = tshift_reference
        self.dlna_acceptance_level = dlna_acceptance_level
        self.dlna_reference = dlna_reference
        self.cc_acceptance_level = cc_acceptance_level
        self.s2n_limit = s2n_limit

        if earth_model.lower() not in ("ak135", "iasp91"):
            raise PyflexError("Earth model must either be 'ak135' or "
                              "'iasp91'.")
        self.earth_model = earth_model.lower()
        self.min_surface_wave_velocity = min_surface_wave_velocity
        self.max_time_before_first_arrival = max_time_before_first_arrival

        self.c_0 = c_0
        self.c_1 = c_1
        self.c_2 = c_2
        self.c_3a = c_3a
        self.c_3b = c_3b
        self.c_4a = c_4a
        self.c_4b = c_4b

        self.check_global_data_quality = check_global_data_quality
        self.snr_integrate_base = snr_integrate_base
        self.snr_max_base = snr_max_base
        self.noise_start_index = noise_start_index
        self.noise_end_index = noise_end_index
        self.signal_start_index = noise_start_index
        self.signal_end_index = noise_end_index

        self.window_weight_fct = window_weight_fct

        snr_type = window_signal_to_noise_type.lower()
        if snr_type not in ["amplitude", "energy"]:
            raise PyflexError("The window signal to noise type must be either"
                              "'amplitude' or 'energy'.")
        self.window_signal_to_noise_type = snr_type

        if resolution_strategy.lower() not in ["interval_scheduling", "merge"]:
            raise PyflexError(
                "Invalid resolution strategy. Choose either "
                "'interval_scheduling' or 'merge'.")
        self.resolution_strategy = resolution_strategy.lower()

    def _convert_to_array(self, npts):
        """
        Internally converts the acceptance and water levels to arrays. Not
        called during initialization as the array length is not yet known.
        """
        attributes = ("stalta_waterlevel", "tshift_acceptance_level",
                      "dlna_acceptance_level", "cc_acceptance_level",
                      "s2n_limit")
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
