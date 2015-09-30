#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class managing the actual window selection process.

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

import json
import numpy as np
import obspy
import obspy.station
from obspy.core.util import geodetics
from obspy.signal.filter import envelope
from obspy.taup import getTravelTimes
import os
import warnings

from . import PyflexError, PyflexWarning, utils, logger, Event, Station
from .stalta import sta_lta
from .window import Window
from .interval_scheduling import schedule_weighted_intervals

with standard_library.hooks():
    import itertools


class WindowSelector(object):
    """
    Low level window selector internally used by Pyflex.
    """
    def __init__(self, observed, synthetic, config, event=None, station=None):
        """
        :param observed: The preprocessed, observed waveform.
        :type observed: :class:`~obspy.core.trace.Trace` or single component
            :class:`~obspy.core.stream.Stream`
        :param synthetic: The preprocessed, synthetic waveform.
        :type synthetic: :class:`~obspy.core.trace.Trace` or single component
            :class:`~obspy.core.stream.Stream`
        :param config: Configuration object.
        :type config: :class:`~.config.Config`
        :param event: The event information. Either a Pyflex Event object,
            or an ObsPy Catalog or Event object. If not given this information
            will be extracted from the data traces if either originates from a
            SAC file.
        :type event: A Pyflex :class:`~pyflex.Event` object,  an ObsPy
            :class:`~obspy.core.event.Catalog` object, or an ObsPy
            :class:`~obspy.core.event.Event` object
        :param station: The station information. Either a Pyflex Station
            object, or an ObsPy Inventory. If not given this information
            will be extracted from the data traces if either originates from
            a SAC file.
        :type station: A Pyflex :class:`~pyflex.Station` object or an ObsPy
            :class:`~obspy.station.inventory.Inventory` object
        """
        self.observed = observed
        self.synthetic = synthetic
        self._sanity_checks()

        self.event = event
        self.station = station
        self._parse_event_and_station()

        # Copy to not modify the original data.
        self.observed = self.observed.copy()
        self.synthetic = self.synthetic.copy()
        self.observed.data = np.ascontiguousarray(self.observed.data)
        self.synthetic.data = np.ascontiguousarray(self.synthetic.data)

        self.config = config
        self.config._convert_to_array(npts=self.observed.stats.npts)

        self.ttimes = []
        self.windows = []

    def load(self, filename):
        """
        Load windows from a JSON file and attach them to the current window
        selector object.

        :param filename: The filename or file-like object to load.
        :type filename: str or file-like object
        """
        if hasattr(filename, "read"):
            obj = json.load(filename)
        else:
            if os.path.exists(filename):
                with open(filename, "r") as fh:
                    obj = json.load(fh)
            else:
                obj = json.loads(filename)

        if "windows" not in obj:
            raise ValueError("Not a valid Windows JSON file.")

        windows = obj["windows"]
        window_objects = []

        for win in windows:
            win_obj = Window._load_from_json_content(win)

            # Perform a number of checks.
            if win_obj.channel_id != self.observed.id:
                raise PyflexError(
                    "The window has channel id '%s' whereas the observed "
                    "data has channel id '%s'." % (
                        win_obj.channel_id, self.observed.id))

            if abs(win_obj.dt - self.observed.stats.delta) / \
                    self.observed.stats.delta >= 0.001:
                raise PyflexError(
                    "The sample interval specified in the window is %g whereas"
                    " the sample interval in the observed data is %g." % (
                        win_obj.delta, self.observed_stats.delta))

            if abs(win_obj.time_of_first_sample -
                    self.observed.stats.starttime) > \
                    0.5 * self.observed.stats.delta:
                raise PyflexError(
                    "The window expects the data to start with at %s whereas "
                    "the observed data starts at %s." % (
                        win.time_of_first_sample,
                        self.observed.stats.starttime))
            # Collect in temporary list and not directly attach to not
            # modify the window object in case a later window raises an
            # exception. Either all or nothing.
            window_objects.append(win_obj)
        self.windows.extend(window_objects)
        # Recalculate window criteria.
        for win in self.windows:
            win._calc_criteria(self.observed.data, self.synthetic.data)

    def write(self, filename):
        """
        Write windows to the specified filename or file like object.

        Will be written as a custom JSON file.

        :param filename: Name or object to write to.
        :type filename: str or file-like object
        """
        windows = [_i._get_json_content() for _i in self.windows]

        info = {"windows": windows}

        class WindowEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, obspy.UTCDateTime):
                    return str(obj)
                # Numpy objects also require explicit handling.
                elif isinstance(obj, np.int64):
                    return int(obj)
                elif isinstance(obj, np.int32):
                    return int(obj)
                elif isinstance(obj, np.float64):
                    return float(obj)
                elif isinstance(obj, np.float32):
                    return float(obj)
                # Let the base class default method raise the TypeError
                return json.JSONEncoder.default(self, obj)

        if not hasattr(filename, "write"):
            with open(filename, "wb") as fh:
                j = json.dumps(
                    info, cls=WindowEncoder, sort_keys=True, indent=4,
                    separators=(',', ': '))
                try:
                    fh.write(j)
                except TypeError:
                    fh.write(j.encode())
        else:
            j = json.dumps(
                info, cls=WindowEncoder, sort_keys=True, indent=4,
                separators=(',', ': '))
            try:
                filename.write(j)
            except TypeError:
                filename.write(j.encode())

    def _parse_event_and_station(self):
        """
        Parse the event and station information.
        """
        # Parse the event.
        if self.event and not isinstance(self.event, Event):
            # It might be an ObsPy event catalog.
            if isinstance(self.event, obspy.core.event.Catalog):
                if len(self.event) != 1:
                    raise PyflexError("The event catalog must contain "
                                      "exactly one event.")
                self.event = self.event[0]
            # It might be an ObsPy event object.
            if isinstance(self.event, obspy.core.event.Event):
                if not self.event.origins:
                    raise PyflexError("Event does not contain an origin.")
                origin = self.event.preferred_origin() or self.event.origins[0]
                self.event = Event(latitude=float(origin.latitude),
                                   longitude=float(origin.longitude),
                                   depth_in_m=float(origin.depth),
                                   origin_time=origin.time)
            else:
                raise PyflexError("Could not parse the event. Unknown type.")

        # Parse the station information if it is an obspy inventory object.
        if isinstance(self.station, obspy.station.Inventory):
            net = self.observed.stats.network
            sta = self.observed.stats.station
            # Workaround for ObsPy 0.9.2 Newer version have a get
            # coordinates method...
            for network in self.station:
                if network.code == net:
                    break
            else:
                raise PyflexError("Could not find the network of the "
                                  "observed data in the inventory object.")
            for station in network:
                if station.code == sta:
                    break
            else:
                raise PyflexError("Could not find the station of the "
                                  "observed data in the inventory object.")
            self.station = Station(latitude=float(station.latitude),
                                   longitude=float(station.longitude))

        # Last resort, if either is not set, and the observed or synthetics
        # are sac files, get the information from there.
        if not self.station or not self.event:
            if hasattr(self.observed.stats, "sac"):
                tr = self.observed
                ftype = "observed"
            elif hasattr(self.synthetic.stats, "sac"):
                tr = self.synthetic
                ftype = "synthetic"
            else:
                return
            sac = tr.stats.sac
            values = (sac.evla, sac.evlo, sac.evdp, sac.stla, sac.stlo, sac.b)
            # Invalid value in sac.
            if -12345.0 in values:
                return
            if not self.station:
                self.station = Station(latitude=values[3], longitude=values[4])
                logger.info("Extracted station information from %s SAC file."
                            % ftype)
            if not self.event:
                self.event = Event(
                    latitude=values[0], longitude=values[1],
                    depth_in_m=values[2] * 1000.0,
                    origin_time=self.observed.stats.starttime - values[5])
                logger.info("Extracted event information from %s SAC file." %
                            ftype)

    def calculate_preliminiaries(self):
        """
        Calculates the envelope, STA/LTA and the finds the local extrema.
        """
        logger.info("Calculating envelope of synthetics.")
        self.synthetic_envelope = envelope(self.synthetic.data)
        logger.info("Calculating STA/LTA.")
        self.stalta = sta_lta(self.synthetic_envelope,
                              self.observed.stats.delta,
                              self.config.min_period)
        self.peaks, self.troughs = utils.find_local_extrema(self.stalta)

        if not len(self.peaks) and len(self.troughs):
            return

        if self.ttimes:
            offset = self.event.origin_time - self.observed.stats.starttime
            min_time = self.ttimes[0]["time"] - \
                self.config.max_time_before_first_arrival + offset
            min_idx = int(min_time / self.observed.stats.delta)

            dist_in_km = geodetics.calcVincentyInverse(
                self.station.latitude, self.station.longitude,
                self.event.latitude, self.event.longitude)[0] / 1000.0
            max_time = dist_in_km / self.config.min_surface_wave_velocity + \
                offset + self.config.max_period
            max_idx = int(max_time / self.observed.stats.delta)

            # Reject all peaks and troughs before the minimal allowed start
            # time and after the maximum allowed end time.
            first_trough, last_trough = self.troughs[0], self.troughs[-1]
            self.troughs = self.troughs[(self.troughs >= min_idx) &
                                        (self.troughs <= max_idx)]

            # If troughs have been removed, readd them add the boundaries.
            if len(self.troughs):
                if first_trough != self.troughs[0]:
                    self.troughs = np.concatenate([
                        np.array([min_idx], dtype=self.troughs.dtype),
                        self.troughs])
                if last_trough != self.troughs[-1]:
                    self.troughs = np.concatenate([
                        self.troughs,
                        np.array([max_idx], dtype=self.troughs.dtype)])
            # Make sure peaks are inside the troughs!
            min_trough, max_trough = self.troughs[0], self.troughs[-1]
            self.peaks = self.peaks[(self.peaks > min_trough) &
                                    (self.peaks < max_trough)]

    def select_windows(self):
        """
        Launch the window selection.
        """
        # Fill self.ttimes.
        if self.event and self.station:
            self.calculate_ttimes()

        self.calculate_preliminiaries()

        # Perform all window selection steps.
        self.initial_window_selection()
        # Reject windows based on traveltime if event and station
        # information is given. This will also fill self.ttimes.
        if self.event and self.station:
            self.reject_on_traveltimes()
        else:
            msg = "No rejection based on traveltime possible. Event and/or " \
                  "station information is not available."
            logger.warning(msg)
            warnings.warn(msg, PyflexWarning)

        self.determine_signal_and_noise_indices()
        if self.config.check_global_data_quality:
            self.check_data_quality()
        self.reject_windows_based_on_minimum_length()
        self.reject_on_minima_water_level()
        self.reject_on_prominence_of_central_peak()
        self.reject_on_phase_separation()
        self.curtail_length_of_windows()
        self.remove_duplicates()
        # Call once again as curtailing might change the length of some
        # windows. Very cheap so can easily be called more than once.
        self.reject_windows_based_on_minimum_length()
        self.reject_based_on_signal_to_noise_ratio()
        self.reject_based_on_data_fit_criteria()

        if self.config.resolution_strategy == "interval_scheduling":
            self.schedule_weighted_intervals()
        elif self.config.resolution_strategy == "merge":
            self.merge_windows()
        else:
            raise NotImplementedError

        if self.ttimes:
            self.attach_phase_arrivals_to_windows()

        return self.windows

    def attach_phase_arrivals_to_windows(self):
        """
        Attaches the theoretical phase arrivals to the windows.
        """
        offset = self.event.origin_time - self.observed.stats.starttime
        for win in self.windows:
            left = win.relative_starttime - offset
            right = win.relative_endtime - offset
            win.phase_arrivals = [
                _i for _i in self.ttimes if left <= _i["time"] <= right]

    def merge_windows(self):
        """
        Merge overlapping windows. Will also recalculate the data fit criteria.
        """
        # Sort by starttime.
        self.windows = sorted(self.windows, key=lambda x: x.left)
        windows = [self.windows.pop(0)]
        for right_win in self.windows:
            left_win = windows[-1]
            if (left_win.right + 1) < right_win.left:
                windows.append(right_win)
                continue
            left_win.right = right_win.right
        self.windows = windows

        for win in self.windows:
            # Recenter windows
            win.center = int(win.left + (win.right - win.left) / 2.0)
            # Recalculate criteria.
            win._calc_criteria(self.observed.data, self.synthetic.data)
        logger.info("Merging windows resulted in %i windows." %
                    len(self.windows))

    def determine_signal_and_noise_indices(self):
        """
        Calculate the time range of the noise and the signal respectively if
        not yet specified by the user.
        """
        if self.config.noise_end_index is None:
            if not self.ttimes:
                logger.warning("Cannot calculate the end of the noise as "
                               "event and/or station information is not given "
                               "and thus the theoretical arrival times cannot "
                               "be calculated")
            else:
                self.config.noise_end_index = \
                    int(self.ttimes[0]["time"] - self.config.min_period)
        if self.config.signal_start_index is None:
            self.config.signal_start_index = self.config.noise_end_index

    def reject_based_on_signal_to_noise_ratio(self):
        """
        Rejects windows based on their signal to noise amplitude ratio.
        """
        if self.config.noise_end_index is None:
            logger.warning("Cannot reject windows based on their signal to "
                           "noise ratio. Please give station and event "
                           "information or information about the temporal "
                           "range of the noise.")
            return

        noise = self.observed.data[self.config.noise_start_index:
                                   self.config.noise_end_index]

        if self.config.window_signal_to_noise_type == "amplitude":
            noise_amp = np.abs(noise).max()

            def filter_window_noise(win):
                win_signal = self.observed.data[win.left: win.right]
                win_noise_amp = np.abs(win_signal).max() / noise_amp
                if win_noise_amp < self.config.s2n_limit[win.center]:
                    return False
                return True

        elif self.config.window_signal_to_noise_type == "energy":
            noise_energy = np.sum(noise ** 2) / len(noise)

            def filter_window_noise(win):
                data = self.observed.data[win.left: win.right]
                win_energy = np.sum(data ** 2) / len(data)
                win_noise_amp = win_energy / noise_energy
                if win_noise_amp < self.config.s2n_limit[win.center]:
                    return False
                return True
        else:
            raise NotImplementedError

        self.windows = list(filter(filter_window_noise, self.windows))
        logger.info("SN amplitude ratio window rejection retained %i windows" %
                    len(self.windows))

    def check_data_quality(self):
        """
        Checks the data quality by estimating signal to noise ratios.
        """
        if self.config.noise_end_index is None:
            raise PyflexError(
                "Cannot check data quality as the noise end index is not "
                "given and station and/or event information is not "
                "available so the theoretical arrival times cannot be "
                "calculated.")

        noise = self.observed.data[self.config.noise_start_index:
                                   self.config.noise_end_index]
        signal = self.observed.data[self.config.signal_start_index:
                                    self.config.signal_end_index]

        noise_int = np.sum(noise ** 2) / len(noise)
        noise_amp = np.abs(noise).max()
        signal_int = np.sum(signal ** 2) / len(signal)
        signal_amp = np.abs(signal).max()

        # Calculate ratios.
        snr_int = signal_int / noise_int
        snr_amp = signal_amp / noise_amp

        if snr_int < self.config.snr_integrate_base:
            msg = ("Whole waveform rejected as the integrated signal to "
                   "noise ratio (%f) is above the threshold (%f)." %
                   (snr_int, self.config.snr_integrate_base))
            logger.warn(msg)
            warnings.warn(msg, PyflexWarning)
            return False

        if snr_amp < self.config.snr_max_base:
            msg = ("Whole waveform rejected as the signal to noise amplitude "
                   "ratio (%f) is above the threshold (%f)." % (
                       snr_amp, self.config.snr_max_base))
            logger.warn(msg)
            warnings.warn(msg, PyflexWarning)
            return False

        logger.info("Global SNR checks passed. Integrated SNR: %f, Amplitude "
                    "SNR: %f" % (snr_int, snr_amp))
        return True

    def calculate_ttimes(self):
        """
        Calculate theoretical travel times. Only call if station and event
        information is available!
        """
        dist_in_deg = geodetics.locations2degrees(
            self.station.latitude, self.station.longitude,
            self.event.latitude, self.event.longitude)
        tts = getTravelTimes(dist_in_deg, self.event.depth_in_m / 1000.0,
                             model=self.config.earth_model)
        self.ttimes = sorted(tts, key=lambda x: x["time"])
        logger.info("Calculated travel times.")

    def reject_on_traveltimes(self):
        """
        Reject based on traveltimes. Will reject windows containing only
        data before a minimum period before the first arrival and windows
        only containing data after the minimum allowed surface wave speed.
        Only call if station and event information is available!
        """
        dist_in_km = geodetics.calcVincentyInverse(
            self.station.latitude, self.station.longitude, self.event.latitude,
            self.event.longitude)[0] / 1000.0

        offset = self.event.origin_time - self.observed.stats.starttime

        min_time = self.ttimes[0]["time"] - self.config.min_period + offset
        max_time = dist_in_km / self.config.min_surface_wave_velocity + offset

        self.windows = [win for win in self.windows
                        if (win.relative_endtime >= min_time) and
                        (win.relative_starttime <= max_time)]
        logger.info("Rejection based on travel times retained %i windows." %
                    len(self.windows))

    def initial_window_selection(self):
        """
        Find all possible windows. This is equivalent to the setup_M_L_R()
        function in flexwin.
        """
        for peak in self.peaks:
            # only continue if there are available minima on either side
            if peak <= self.troughs[0] or peak >= self.troughs[-1]:
                continue
            # only continue if this maximum is above the water level
            if self.stalta[peak] <= self.config.stalta_waterlevel[peak]:
                continue
            smaller_troughs = self.troughs[self.troughs < peak]
            larger_troughs = self.troughs[self.troughs > peak]

            for left, right in itertools.product(smaller_troughs,
                                                 larger_troughs):
                self.windows.append(Window(
                    left=left, right=right, center=peak,
                    channel_id=self.observed.id,
                    time_of_first_sample=self.synthetic.stats.starttime,
                    dt=self.observed.stats.delta,
                    min_period=self.config.min_period,
                    weight_function=self.config.window_weight_fct))

        logger.info("Initial window selection yielded %i possible windows." %
                    len(self.windows))

    def remove_duplicates(self):
        """
        Filter to remove duplicate windows based on left and right bounds.

        This function will also change the middle to actually be in the
        center of the window. This should result in better results for the
        following stages as lots of thresholds are evaluated at the center
        of a window.
        """
        new_windows = {}
        for window in self.windows:
            tag = (window.left, window.right)
            if tag not in new_windows:
                window.center = \
                    int(window.left + (window.right - window.left) / 2.0)
                new_windows[tag] = window
        self.windows = sorted(new_windows.values(), key=lambda x: x.left)
        logger.info("Removing duplicates retains %i windows." % len(
            self.windows))

    def schedule_weighted_intervals(self):
        """
        Run the weighted interval scheduling.
        """
        self.windows = schedule_weighted_intervals(self.windows)
        logger.info("Weighted interval schedule optimzation retained %i "
                    "windows." % len(self.windows))

    def reject_on_minima_water_level(self):
        """
        Filter function rejecting windows whose internal minima are below the
        water level of the windows peak. This is equivalent to the
        reject_on_water_level() function in flexwin.
        """
        def filter_window_minima(win):
            waterlevel_midpoint = \
                self.config.c_0 * self.config.stalta_waterlevel[win.center]
            internal_minima = win._get_internal_indices(self.troughs)
            return not np.any(self.stalta[internal_minima] <=
                              waterlevel_midpoint)

        self.windows = list(filter(filter_window_minima, self.windows))
        logger.info("Water level rejection retained %i windows" %
                    len(self.windows))

    def reject_on_prominence_of_central_peak(self):
        """
        Equivalent to reject_on_prominence() in the original flexwin code.
        """
        # The fine tuning constant is often set to 0. Nothing to do in this
        # case as all windows will then pass the criteria by definition.
        if not self.config.c_2:
            return self.windows

        def filter_windows_maximum_prominence(win):
            smaller_troughs = self.troughs[self.troughs < win.center]
            larger_troughs = self.troughs[self.troughs > win.center]

            if not len(smaller_troughs) or not len(larger_troughs):
                return False

            left = self.stalta[smaller_troughs[-1]]
            right = self.stalta[larger_troughs[0]]
            center = self.stalta[win.center]
            delta_left = center - left
            delta_right = center - right

            if (delta_left < self.config.c_2 * center) or \
                    (delta_right < self.config.c_2 * center):
                return False
            return True

        self.windows = list(filter(filter_windows_maximum_prominence,
                                   self.windows))
        logger.info("Prominence of central peak rejection retained "
                    "%i windows." % len(self.windows))

    def reject_on_phase_separation(self):
        """
        Reject windows based on phase seperation. Equivalent to
        reject_on_phase_separation() in the original flexwin code.
        """
        def filter_phase_rejection(win):
            # Find the lowest minimum within the window.
            internal_minima = self.troughs[
                (self.troughs >= win.left) & (self.troughs <= win.right)]
            stalta_min = self.stalta[internal_minima].min()
            # find the height of the central maximum above this minimum value
            d_stalta_center = self.stalta[win.center] - stalta_min
            # Find all internal maxima.
            internal_maxima = self.peaks[
                (self.peaks >= win.left) & (self.peaks <= win.right) &
                (self.peaks != win.center)]

            for max_index in internal_maxima:
                # find height of current maximum above lowest minimum
                d_stalta = self.stalta[max_index] - stalta_min
                # find scaled time between current maximum and central maximum
                d_time = abs(win.center - max_index) * \
                    self.observed.stats.delta / self.config.min_period
                # find value of time decay function.
                # The paper has a square root in the numinator of the
                # exponent as well. Not the case here as it is not the case
                # in the original flexwin code.
                if (d_time >= self.config.c_3b):
                    f_time = np.exp(-((d_time - self.config.c_3b) /
                                      self.config.c_3b) ** 2)
                else:
                    f_time = 1.0
                # check condition
                if d_stalta > (self.config.c_3a * d_stalta_center * f_time):
                    break
            else:
                return True
            return False

        self.windows = list(filter(
            filter_phase_rejection, self.windows))
        logger.info("Single phase group rejection retained %i windows" %
                    len(self.windows))

    def curtail_length_of_windows(self):
        """
        Curtail the window length. Equivalent to a call to
        curtail_window_length() in the original flexwin code.
        """
        dt = self.observed.stats.delta

        def curtail_window_length(win):
            time_decay_left = self.config.min_period * self.config.c_4a / dt
            time_decay_right = self.config.min_period * self.config.c_4b / dt
            # Find all internal maxima.
            internal_maxima = self.peaks[
                (self.peaks >= win.left) & (self.peaks <= win.right) &
                (self.peaks != win.center)]
            if len(internal_maxima) < 2:
                return win
            i_left = internal_maxima[0]
            i_right = internal_maxima[-1]

            delta_left = i_left - win.left
            delta_right = win.right - i_right

            # check condition
            if delta_left > time_decay_left:
                logger.info("Curtailing left")
                win.left = int(i_left - time_decay_left)
            if delta_right > time_decay_right:
                logger.info("Curtailing right")
                win.right = int(i_right + time_decay_right)
            return win

        self.windows = [curtail_window_length(i) for i in self.windows]

    @property
    def minimum_window_length(self):
        """
        Minimum acceptable window length.
        """
        return self.config.c_1 * self.config.min_period / \
            self.observed.stats.delta

    def reject_windows_based_on_minimum_length(self):
        """
        Reject windows smaller than the minimal window length.
        """
        self.windows = list(filter(
            lambda x: (x.right - x.left) >= self.minimum_window_length,
            self.windows))
        logger.info("Rejection based on minimum window length retained %i "
                    "windows." % len(self.windows))

    def reject_based_on_data_fit_criteria(self):
        """
        Rejects windows based on similarity between data and synthetics.
        """
        # First calculate the criteria for all remaining windows.
        for win in self.windows:
            win._calc_criteria(self.observed.data, self.synthetic.data)

        def reject_based_on_criteria(win):
            tshift_min = self.config.tshift_reference - \
                self.config.tshift_acceptance_level[win.center]
            tshift_max = self.config.tshift_reference + \
                self.config.tshift_acceptance_level[win.center]
            dlnA_min = self.config.dlna_reference - \
                self.config.dlna_acceptance_level[win.center]
            dlnA_max = self.config.dlna_reference + \
                self.config.dlna_acceptance_level[win.center]

            if not (tshift_min < win.cc_shift *
                    self.observed.stats.delta < tshift_max):
                logger.debug("Window rejected due to time shift: %f" %
                             win.cc_shift)
                return False
            if not (dlnA_min < win.dlnA < dlnA_max):
                logger.debug("Window rejected due to amplitude fit: %f" %
                             win.dlnA)
                return False
            if win.max_cc_value < self.config.cc_acceptance_level[win.center]:
                logger.debug("Window rejected due to CC value: %f" %
                             win.max_cc_value)
                return False
            return True

        self.windows = list(filter(reject_based_on_criteria, self.windows))
        logger.info("Rejection based on data fit criteria retained %i windows."
                    % len(self.windows))

    def _sanity_checks(self):
        """
        Perform a number of basic sanity checks to assure the data is valid
        in a certain sense.

        It checks the types of both, the starttime, sampling rate, number of
        samples, ...
        """
        if not isinstance(self.observed, obspy.Trace):
            # Also accept Stream objects.
            if isinstance(self.observed, obspy.Stream) and \
                    len(self.observed) == 1:
                self.observed = self.observed[0]
            else:
                raise PyflexError(
                    "Observed data must be an ObsPy Trace object.")
        if not isinstance(self.synthetic, obspy.Trace):
            if isinstance(self.synthetic, obspy.Stream) and \
                    len(self.synthetic) == 1:
                self.synthetic = self.synthetic[0]
            else:
                raise PyflexError(
                    "Synthetic data must be an ObsPy Trace object.")

        if self.observed.stats.npts != self.synthetic.stats.npts:
            raise PyflexError("Observed and synthetic data must have the same "
                              "number of samples.")

        sr1 = self.observed.stats.sampling_rate
        sr2 = self.synthetic.stats.sampling_rate

        if abs(sr1 - sr2) / sr1 >= 1E-5:
            raise PyflexError("Observed and synthetic data must have the same "
                              "sampling rate.")

        # Make sure data and synthetics start within half a sample interval.
        if abs(self.observed.stats.starttime -
               self.synthetic.stats.starttime) > self.observed.stats.delta * \
                0.5:
            raise PyflexError("Observed and synthetic data must have the same "
                              "starttime.")

        ptp = sorted([self.observed.data.ptp(), self.synthetic.data.ptp()])
        if ptp[1] / ptp[0] >= 5:
            warnings.warn("The amplitude difference between data and "
                          "synthetic is fairly large.")

        # Also check the components of the data to avoid silly mistakes of
        # users.
        if len(set([self.observed.stats.channel[-1].upper(),
                    self.synthetic.stats.channel[-1].upper()])) != 1:
            warnings.warn("The orientation code of synthetic and observed "
                          "data is not equal.")

    def plot(self, filename=None):
        """
        Plot the current state of the windows.

        :param filename: If given, the plot will be written to this file,
            otherwise the plot will be shown.
        :type filename: str
        """
        # Lazy imports to not import matplotlib all the time.
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        # Use an offset to have the seconds since the event as the time axis.
        if self.event:
            offset = self.event.origin_time - self.observed.stats.starttime
        else:
            offset = 0

        plt.figure(figsize=(15, 5))

        plt.axes([0.025, 0.92, 0.95, 0.07])

        times = self.observed.times() - offset

        # Plot theoretical arrivals.
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(times[0], times[-1])

        for tt in self.ttimes:
            if tt["phase_name"].lower().startswith("p"):
                color = "#008c28"
            else:
                color = "#950000"
            # Don't need an offset as the time axis corresponds to time
            # since event.
            plt.vlines(tt["time"], plt.ylim()[0], plt.ylim()[1], color=color)

        plt.text(0.01, 0.92, 'Phase Arrivals', horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes)

        plt.axes([0.025, 0.51, 0.95, 0.4])
        plt.plot(times, self.observed.data, color="black")
        plt.plot(times, self.synthetic.data, color="red")
        plt.xlim(times[0], times[-1])

        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.set_xticks([])
        ax.set_yticks([])

        plt.text(0.01, 0.99, 'Seismograms', horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes)

        buf = 0.003 * (plt.xlim()[1] - plt.xlim()[0])
        for win in self.windows:
            l = win.relative_starttime - offset
            r = win.relative_endtime - offset
            re = Rectangle((l, plt.ylim()[0]), r - l,
                           plt.ylim()[1] - plt.ylim()[0], color="blue",
                           alpha=(win.max_cc_value ** 2) * 0.25)
            plt.gca().add_patch(re)
            plt.text(l + buf, plt.ylim()[1],
                     "CC=%.2f\ndT=%.2f\ndA=%.2f" %
                     (win.max_cc_value,
                      win.cc_shift * self.observed.stats.delta,
                      win.dlnA),
                     horizontalalignment="left",
                     verticalalignment="top", rotation="vertical",
                     size="small", multialignment="right")

        plt.axes([0.025, 0.1, 0.95, 0.4])
        plt.plot(times, self.stalta, color="blue")
        plt.plot(times, self.config.stalta_waterlevel, linestyle="dashed",
                 color="blue")
        plt.xlim(times[0], times[-1])
        if self.event:
            plt.xlabel("Time [s] since event")
        else:
            plt.xlabel("Time [s]")

        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_yticks([])
        ax.xaxis.set_ticks_position('bottom')

        plt.text(0.01, 0.99, 'STA/LTA', horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes)

        for win in self.windows:
            l = win.relative_starttime - offset
            r = win.relative_endtime - offset
            re = Rectangle((l, plt.ylim()[0]), r - l,
                           plt.ylim()[1] - plt.ylim()[0], color="blue",
                           alpha=(win.max_cc_value ** 2) * 0.25)
            plt.gca().add_patch(re)

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
