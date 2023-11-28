#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class managing the actual window selection process.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
    adjTomo Dev Team (adjtomo@gmail.com), 2022
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import copy
import itertools
import json
import os
import warnings

import numpy as np
import obspy
import obspy.geodetics
from obspy.signal.filter import envelope
from obspy.taup import TauPyModel

from pyflex import PyflexError, PyflexWarning, utils, logger, Event, Station
from pyflex.stalta import sta_lta
from pyflex.window import Window
from pyflex.utils import get_surface_wave_arrivals
from pyflex.interval_scheduling import schedule_weighted_intervals


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
            :class:`~obspy.core.inventory.Inventory` object
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

        self.config = copy.deepcopy(config)
        self.config._convert_to_array(npts=self.observed.stats.npts)

        self.ttimes = []
        self.windows = []
        self.rejects = {}

        self.taupy_model = TauPyModel(model=self.config.earth_model)

        # event and station distance
        self.dist_in_deg = None
        self.dist_in_km = None

        # used by selection mode to specify the time range could be used
        self.selection_timebox = None

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

    def write_text(self, filename):
        """
        Write windows to a plain text file. For example, after
        you selecting windows, you want to write them out and
        save as text files, you can call:
        ws.write_text("window.txt")

        :param filename: Name to write to.
        :type filename: str
        """
        if not os.path.exists(os.path.dirname(filename)):
            raise ValueError("Output file directory not exist: %s" % filename)
        with open(filename, 'w') as f:
            f.write("%s\n" % self.observed.id)
            f.write("%s\n" % self.synthetic.id)
            f.write("%d\n" % len(self.windows))
            for win in self.windows:
                f.write("%10.2f %10.2f\n" % (win.relative_starttime,
                                             win.relative_endtime))

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
        if isinstance(self.station, obspy.core.inventory.Inventory):
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
            if hasattr(self.synthetic.stats, "sac"):
                tr = self.synthetic
                ftype = "synthetic"
            elif hasattr(self.observed.stats, "sac"):
                tr = self.observed
                ftype = "observed"
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
                    origin_time=tr.stats.starttime - values[5])
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

        logger.debug("Found %i peaks and %i troughs." % (
            len(self.peaks), len(self.troughs)))

        if not len(self.peaks) and len(self.troughs):
            return

    def reject_peaks_and_troughs_not_in_signal_region(self):
        """
        Reject all the peaks and troughs not in the signal regions.
        This will save us huge amount of time by rejecting a large
        number of non-sense peaks and troughs.
        """

        if self.ttimes:

            if self.config.selection_mode \
                    and "surface" in self.config.selection_mode:
                # For the surface wave the selection of troughs will be
                # taken care of by the surface wave selection box
                min_time = 0
                max_time = self.observed.stats.endtime - \
                    self.observed.stats.starttime
                min_idx = 0
                max_idx = self.observed.stats.npts


            else:

                offset = self.event.origin_time - self.observed.stats.starttime
                min_time = self.ttimes[0]["time"] - \
                    self.config.max_time_before_first_arrival + offset
                min_idx = int(min_time / self.observed.stats.delta)

                dist_in_km = obspy.geodetics.calc_vincenty_inverse(
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


    def __print_remaining_windows(self):
        logger.debug("Remaining windows: %d" % (len(self.windows)))
        logger.debug("idx:    left(s)  center(s)   right(s)")
        for idx, win in enumerate(self.windows):
            left = win.relative_starttime
            right = win.relative_endtime
            center = win.relative_centertime
            logger.debug("%3d: %10.2f %10.2f %10.2f"
                         % (idx+1, left, center, right))

    def select_windows(self):
        """
        Launch the window selection.
        """
        # Fill self.ttimes.
        logger.debug(f"======================================================")
        logger.debug(f"Start selection for {self.observed.id}")
        logger.debug(f"======================================================")
        if self.event and self.station:
            self.calculate_distance()
            self.calculate_ttimes()

        # Calculate some preliminaries measurements
        self.calculate_preliminiaries()

        # Reject peaks and troughs not in the signal region
        self.reject_peaks_and_troughs_not_in_signal_region()

        # Perform all window selection steps.
        self.initial_window_selection()


        # Reject windows based on traveltime if event and station
        # information is given. This will also fill self.ttimes.
        if self.event and self.station:
            if self.config.selection_mode is not None:
                # Base on specific selection mode
                self.reject_on_selection_mode()
                pass
            else:
                # Based on basic traveltimes
                self.reject_on_traveltimes()
        else:
            msg = "No rejection based on traveltime possible. Event and/or " \
                "station information is not available."
            logger.warning(msg)
            warnings.warn(msg, PyflexWarning)

        # Determine SNR indeces
        self.determine_signal_and_noise_indices()
        if self.config.check_global_data_quality:
            if not self.check_data_quality():
                return []

        # Reject windows in the noise region
        self.reject_on_noise_region()

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
        self.__print_remaining_windows()

        if self.ttimes:
            self.attach_phase_arrivals_to_windows()

        if self.event and self.station:
            self.attach_distance_to_windows()

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

    def attach_distance_to_windows(self):
        for win in self.windows:
            win.distance_in_deg = self.dist_in_deg

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

    def calculate_noise_end_index(self):
        """
        If self.config.noise_end_index is not given, calculate the noise
        end index based the first arrival(event and station information
        required).
        """
        offset = self.event.origin_time - self.observed.stats.starttime
        noise_end_index = int(
            (self.ttimes[0]["time"] + offset -
             self.config.max_time_before_first_arrival) *
            self.observed.stats.sampling_rate)
        noise_end_index = max(noise_end_index, 1)
        return noise_end_index

    def calculate_signal_end_index(self):
        """
        If self.config.noise_end_index is not given, calculate the noise
        end index based the first arrival(event and station information
        required).
        """

        offset = self.event.origin_time - self.observed.stats.starttime
        # signal end index
        surface_wave_arrival = \
            self.dist_in_km / self.config.min_surface_wave_velocity
        # max of last arrival and surface wave arrival
        logger.debug("last ttimess: {}".format(self.ttimes[-1]))
        logger.debug("surface wave arrival: {}".format(surface_wave_arrival))
        logger.debug("sampling rate: {}".format(
            self.observed.stats.sampling_rate))
        last_arrival = max(self.ttimes[-1]["time"], surface_wave_arrival)
        signal_end_index = int(
            (last_arrival + offset +
             self.config.max_time_after_last_arrival) *
            self.observed.stats.sampling_rate)

        npts = self.observed.stats.npts

        if self.config.selection_mode:
            if "surface" in self.config.selection_mode:
                # This is ok, because a timebox will be computed anyways
                # to exclude anything outside specific regions!
                signal_end_index = npts
        else:
            signal_end_index = min(signal_end_index, npts)

        return signal_end_index

    def determine_signal_and_noise_indices(self):
        """
        Calculate the time range of the noise and the signal respectively if
        not yet specified by the user.
        """
        logger.debug("obsd and synt trace npts: {}, {}".format(
            self.observed.stats.npts, self.synthetic.stats.npts))

        if self.config.noise_start_index is None:
            self.config.noise_start_index = 0

        if self.config.noise_end_index is None:
            if not self.ttimes:
                logger.warning("Cannot calculate the end of the noise as "
                               "event and/or station information is not given "
                               "and thus the theoretical arrival times cannot "
                               "be calculated")
            else:
                self.config.noise_end_index = self.calculate_noise_end_index()

        if self.config.signal_start_index is None and \
                self.config.noise_end_index is not None:
            self.config.signal_start_index = self.config.noise_end_index

        if self.config.signal_end_index is None:
            if not self.ttimes:
                logger.warning("Cannot calculate the end of the signal as "
                               "event and/or station information is not given "
                               "and thus the theoretical arrival times cannot "
                               "be calculated")
            else:
                self.config.signal_end_index = \
                    self.calculate_signal_end_index()

        self.config._convert_negative_index(npts=self.observed.stats.npts)

        logger.debug("Noise index [%s, %s]; signal index [%s, %s]" % (
                    self.config.noise_start_index,
                    self.config.noise_end_index,
                    self.config.signal_start_index,
                    self.config.signal_end_index))

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

        # safe guard noise_end_index when event and station are very close
        noise = self.observed.data[self.config.noise_start_index:
                                   max(self.config.noise_end_index, 10)]
        noise_amp = np.abs(noise).max()
        noise_energy = np.sum(noise ** 2) / len(noise)

        def filter_window_noise_amplitude(win):
            win_signal = self.observed.data[win.left:win.right]
            win_noise_amp = np.abs(win_signal).max() / noise_amp
            # attach snr information to window
            win.snr_amplitude = win_noise_amp
            win.snr_amplitude_threshold = self.config.s2n_limit[win.center]
            if win_noise_amp < self.config.s2n_limit[win.center]:
                left = win.relative_starttime
                right = win.relative_endtime
                logger.debug("Win rejected due to S2N ratio (Amplitude):"
                             "%3.1f %5.1f %5.1f"
                             % (win_noise_amp, left, right))
                return False
            return True

        def filter_window_noise_energy(win):
            data = self.observed.data[win.left:win.right]
            win_energy = np.sum(data ** 2) / len(data)
            win_noise_energy = win_energy / noise_energy
            # attach snr information to window
            win.snr_energy = win_noise_energy
            win.snr_energy_threshold = self.config.s2n_limit_energy[win.center]
            if win_noise_energy < self.config.s2n_limit_energy[win.center]:
                left = win.relative_starttime
                right = win.relative_endtime
                logger.debug("Win rejected due to S2N ratio (Energy):"
                             "%3.1f %5.1f %5.1f"
                             % (win_noise_energy, left, right))
                return False
            return True

        # Make sure variable shorter
        window_snr_type = self.config.window_signal_to_noise_type

        # Copy list
        n0 = len(self.windows)
        windows = self.windows[:]

        # Very short source-receiver distances can sometimes produce 0 length
        # noise signals
        if not noise.any():
            logger.warning("pre-arrival noise could not be determined, "
                           "skipping rejection based on signal-to-noise ratio")
            return

        # Filter windows for Amplitude SNR
        if window_snr_type in ("amplitude", "amplitude_and_energy"):
            windows = list(filter(filter_window_noise_amplitude,
                                  windows))
            logger.info("SNR(Amplitude) rejection retained {}/{} "
                        "windows".format(len(windows), n0))

        # Current length of windows
        n1 = len(windows)

        # Filter windows for Energy SNR
        if window_snr_type in ("energy", "amplitude_and_energy"):

            windows = list(filter(filter_window_noise_energy,
                                       windows))
            logger.info("SNR(Energy) rejection retained {}/{} "
                        "windows".format(len(windows), n1))

        # Separate windows and rejects
        self.separate_rejects(windows, "s2n")

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

        # safe guard noise_end_index when event and station are very close
        noise = self.observed.data[self.config.noise_start_index:
                                   max(self.config.noise_end_index, 10)]
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
                   "noise ratio (%f) is above the threshold (%f). No window "
                   "will be selected." %
                   (snr_int, self.config.snr_integrate_base))
            logger.warn(msg)
            return False

        if snr_amp < self.config.snr_max_base:
            msg = ("Whole waveform rejected as the signal to noise amplitude "
                   "ratio (%f) is above the threshold (%f). No window will"
                   "be selected." % (
                       snr_amp, self.config.snr_max_base))
            logger.warn(msg)
            return False

        logger.info("Global SNR checks passed. Integrated SNR: {:.2f}, "
                    "Amplitude SNR {:.2f}".format(snr_int, snr_amp))
        return True

    def calculate_distance(self):
        """
        Calculate the distance between event and station
        """
        self.dist_in_deg = obspy.geodetics.locations2degrees(
            self.station.latitude, self.station.longitude,
            self.event.latitude, self.event.longitude)
        self.dist_in_km = obspy.geodetics.degrees2kilometers(self.dist_in_deg)
        logger.debug(r"dist_in_deg and dist_in_km: {:.2f}, {:.2f} km".format(
            self.dist_in_deg, self.dist_in_km))

    def calculate_ttimes(self):
        """
        Calculate theoretical travel times. Only call if station and event
        information is available!
        """
        logger.debug("event lat, lon: {}, {}".format(
            self.event.latitude, self.event.longitude))
        logger.debug("station lat, lon: {}, {}".format(
            self.station.latitude, self.station.longitude))

        tts = self.taupy_model.get_travel_times(
            source_depth_in_km=self.event.depth_in_m / 1000.0,
            distance_in_degree=self.dist_in_deg)

        logger.debug("source depth: {:.2f} km".format(
            self.event.depth_in_m / 1000.0))

        self.ttimes = [{"time": _i.time, "name": _i.name} for _i in tts]

        logger.info("Calculated travel times.")


    def reject_on_noise_region(self):
        """
        Reject windows whose center is in the noise region
        (center > noise_end_index).
        We also put another check here, to make sure the left boarder
        is not too far away with the `noise_end_index`.
        """
        if self.config.noise_end_index is None:
            return

        # window.center threshold is the noise_end_index
        center_threshold = self.config.noise_end_index
        # window.left threshold is the (noise_end - 2 * min_period)
        two_min_period_npts = 2 * self.config.min_period * \
            self.observed.stats.sampling_rate
        left_threshold = self.config.noise_end_index - two_min_period_npts

        # reject windows which has overlap with the noise region
        self.windows = \
            [win for win in self.windows
             if (win.center > center_threshold) and
             (win.left > left_threshold)]

    def _prepare_min_max_timebox(self, min_time, max_time, srate):
        logger.debug("selection_timebox min/max time: {:.2f} {:.2f}"
                     "".format(min_time, max_time))

        timebox = np.zeros(self.observed.stats.npts, dtype=np.bool)
        il = int(min_time * srate)
        ir = int(max_time * srate)
        timebox[il:(ir+1)] = 1

        logger.debug("selection_timebox index range: {} - {} / {}".format(
            il, ir, len(timebox)))

        self.selection_timebox = timebox
        return timebox

    def _prepare_exclude_surface_wave_timebox(self, min_time, max_time,
                                              srate, offset):
        logger.debug("selection_timebox exclude surface waves")

        # get the surface windows first
        self._prepare_surface_wave_timebox(srate, offset)

        # flip the sign to reject surface windoes
        timebox = self.selection_timebox
        timebox = ~timebox

        # set the min_time and max_time range to non-selectable
        i1 = int(min_time * srate)
        timebox[:i1] = False

        i2 = int(max_time * srate)
        timebox[i2:] = False

        logger.debug("Selectable region coverage percentage: {}/{}".format(
            timebox.sum(), len(timebox)))
        self.selection_timebox = timebox
        return timebox

    def _prepare_surface_wave_timebox(self, srate, offset):
        logger.debug("selection_timebox using surface waves")

        max_vel = self.config.max_surface_wave_velocity
        min_vel = self.config.min_surface_wave_velocity
        arrivals = get_surface_wave_arrivals(
            self.dist_in_deg, min_vel, max_vel, ncircles=2)
        logger.debug("Number of surface wave arrivals: {}".format(
            len(arrivals)))

        timebox = np.zeros(self.observed.stats.npts, dtype=np.bool)

        for _i, arr in enumerate(arrivals):
            il = int((arr[0] + offset - self.config.min_period) * srate)
            ir = int((arr[1] + offset + 2.5 * self.config.min_period) * srate)
            logger.debug(f"Arrival {_i} (min, max): ({il}, {ir})")
            timebox[il:(ir+1)] = 1
        logger.debug("Selectable region coverage percentage: {}/{}".format(
            timebox.sum(), len(timebox)))
        self.selection_timebox = timebox
        return timebox

    def _prepare_phase_timebox(self, phase_list, srate, buffer_time, offset):
        logger.debug("Selection_timebox using phases: {}".format(phase_list))

        source_depth = self.event.depth_in_m / 1000.0
        logger.debug("event depth: {:.2f} km".format(source_depth))
        logger.debug("distance in deg: {:.2f}".format(self.dist_in_deg))
        tts = self.taupy_model.get_travel_times(
            source_depth_in_km=source_depth,
            distance_in_degree=self.dist_in_deg,
            phase_list=phase_list
        )

        timebox = np.zeros(self.observed.stats.npts, dtype=np.bool)
        for t in tts:
            t_left = t.time - buffer_time + offset
            il = int(t_left * srate)
            t_right = t.time + buffer_time + offset
            ir = int(t_right * srate)
            timebox[il:(ir+1)] = 1

        logger.debug("Selectable region percentage: {}/{}".format(
            timebox.sum(), len(timebox)))
        self.selection_timebox = timebox

        return timebox

    def reject_on_traveltimes(self):
        """
        Reject based on traveltimes. Will reject windows containing only
        data before a minimum period before the first arrival and windows
        only containing data after the minimum allowed surface wave speed.
        Only call if station and event information is available!
        """
        dist_in_km = obspy.geodetics.calc_vincenty_inverse(
            self.station.latitude, self.station.longitude, self.event.latitude,
            self.event.longitude)[0] / 1000.0

        offset = self.event.origin_time - self.observed.stats.starttime

        min_time = self.ttimes[0]["time"] - self.config.min_period + offset
        max_time = dist_in_km / self.config.min_surface_wave_velocity + offset
        windows = [win for win in self.windows
                   if (win.relative_endtime >= min_time) and
                   (win.relative_starttime <= max_time)]

        self.separate_rejects(windows, "traveltimes")
        logger.info("Rejection based on travel times retained %i windows." %
                    len(self.windows))


    def reject_on_selection_mode(self):
        """
        Reject based on selection mode.
        This function will reject windows outside of the
        wave category specified. For example, if config.selection_mode
        == "body_waves", only body wave windows will be selected(after
        first arrival and before surface wave arrival).
        """
        select_mode = self.config.selection_mode
        logger.debug("Selection mode <{}>".format((select_mode)))
        if select_mode in [None, "", "custom"]:
            return

        offset = self.event.origin_time - self.observed.stats.starttime
        buffer_time = 2 * self.config.min_period

        surface_arrival = \
            self.dist_in_km / self.config.max_surface_wave_velocity
        surface_end = \
            self.dist_in_km / self.config.min_surface_wave_velocity

        stats = self.observed.stats
        srate = self.observed.stats.sampling_rate
        tr_timelen = stats.endtime - stats.starttime

        first_arrival = self.ttimes[0]["time"]

        if select_mode == "all_waves":
            min_time = first_arrival - buffer_time + offset
            max_time = tr_timelen
            self._prepare_min_max_timebox(min_time, max_time, srate)
        elif select_mode == "body_and_mantle_waves":
            min_time = first_arrival - buffer_time + offset
            max_time = tr_timelen
            self._prepare_exclude_surface_wave_timebox(
                min_time, max_time, srate, offset)
        elif select_mode == "body_and_surface_waves":
            min_time = first_arrival - buffer_time + offset
            max_time = surface_end + buffer_time + offset
            self._prepare_min_max_timebox(min_time, max_time, srate)
        elif select_mode == "body_waves":
            min_time = first_arrival - buffer_time + offset
            max_time = surface_arrival + buffer_time + offset
            self._prepare_min_max_timebox(min_time, max_time, srate)
        elif select_mode == "surface_waves":
            self._prepare_surface_wave_timebox(srate, offset)
        elif select_mode == "mantle_waves":
            min_time = surface_end + offset
            max_time = tr_timelen
            self._prepare_min_max_timebox(min_time, max_time, srate)
        elif select_mode.startswith("phase_list"):
            phases = select_mode.split(":")[-1].split(",")
            phases = [p.strip() for p in phases]
            if len(phases) == 0:
                raise ValueError("No phase provided for 'phase_list' "
                                 "selection mode")
            self._prepare_phase_timebox(
                phases, srate, buffer_time, offset)
        else:
            raise NotImplementedError

        n0 = len(self.windows)
        self.windows = [win for win in self.windows
                        if self.selection_timebox[win.center]]
        logger.info("Rejection based on selection mode retained {}({})"
                    "windows.".format(len(self.windows), n0))

    def initial_window_selection(self):
        """
        Find all possible windows. This is equivalent to the setup_M_L_R()
        function in flexwin.
        """
        for _i, peak in enumerate(self.peaks):

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
                    channel_id_2=self.synthetic.id,
                    time_of_first_sample=self.synthetic.stats.starttime,
                    dt=self.observed.stats.delta,
                    min_period=self.config.min_period,
                    weight_function=self.config.window_weight_fct))

        logger.info("Initial window selection yielded %i possible windows." %
                    len(self.windows))

    def separate_rejects(self, windows, key):
        """
        Separate a new batch of selected windows from the rejected windows.

        Similar to remove_duplicates(), checks left and right bounds to
        determine which windows have been rejected by a given function,
        reassigns internal windows attribute, and adds rejected windows to
        rejects attribute with a given tag specified by calling function

        :type windows: list
        :param windows: list of Window objects that have been outputted by
            a rejection function
        :type tag: str
        :param tag: tag based on the function that rejected the windows
        """
        # Determine unique tags for the new set of windows
        new_windows = [(win.left, win.right) for win in windows]

        # Scroll through current windows and sort by accept/reject
        accepted_windows, rejected_windows = [], []
        for window in self.windows:
            tag = (window.left, window.right)
            if tag in new_windows:
                accepted_windows.append(window)
            else:
                rejected_windows.append(window)

        self.windows = accepted_windows
        if rejected_windows:
            self.rejects[key] = rejected_windows

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

        logger.info("Weighted interval schedule optimization retained %i "
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

        windows = list(filter(filter_window_minima, self.windows))
        self.separate_rejects(windows, "water_level")

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

        windows = list(filter(filter_windows_maximum_prominence,
                                   self.windows))
        self.separate_rejects(windows, "prominence")

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
                if d_time >= self.config.c_3b:
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

        windows = list(filter(filter_phase_rejection, self.windows))
        self.separate_rejects(windows, "phase_sep")
        logger.info("Single phase group rejection retained %i windows" %
                    len(self.windows))

    def curtail_length_of_windows(self):
        """
        Curtail the window length. Equivalent to a call to
        curtail_window_length() in the original flexwin code.
        """
        dt = self.observed.stats.delta

        def curtail_window_length(win):
            curtail_status = [0, 0]
            time_decay_left = self.config.min_period * self.config.c_4a / dt
            time_decay_right = self.config.min_period * self.config.c_4b / dt
            # Find all internal maxima.
            internal_maxima = self.peaks[
                (self.peaks >= win.left) & (self.peaks <= win.right)]
            i_left = internal_maxima[0]
            i_right = internal_maxima[-1]

            delta_left = i_left - win.left
            delta_right = win.right - i_right

            # check condition
            if delta_left > time_decay_left:
                win.left = int(i_left - time_decay_left)
                curtail_status[0] = 1
            if delta_right > time_decay_right:
                win.right = int(i_right + time_decay_right)
                curtail_status[1] = 1
            return win, curtail_status

        # Check curtailing status
        windows = []
        nleft = 0
        nright = 0
        for win in self.windows:
            new_win, curtail_status = curtail_window_length(win)
            nleft += curtail_status[0]
            nright += curtail_status[1]
            windows.append(new_win)

        self.separate_rejects(windows, "curtail")

        logger.info(
            "Curtailing is applied on %d on total %d: <%d(left), %d(right)>"
            % (nleft + nright, len(self.windows), nleft, nright))

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
        windows = list(filter(
            lambda x: (x.right - x.left) >= self.minimum_window_length,
            self.windows))

        self.separate_rejects(windows, "min_length")

        logger.info("Rejection based on minimum window length retained %i "
                    "windows." % len(self.windows))

    def reject_based_on_data_fit_criteria(self):
        """
        Rejects windows based on similarity between data and synthetics.
        """
        # First calculate the criteria for all remaining windows.
        for win in self.windows:
            win._calc_criteria(self.observed.data, self.synthetic.data)

        def reject_based_on_time_shift(win):
            tshift_min = self.config.tshift_reference - \
                self.config.tshift_acceptance_level[win.center]
            tshift_max = self.config.tshift_reference + \
                self.config.tshift_acceptance_level[win.center]

            if not (tshift_min <= win.cc_shift_in_seconds <= tshift_max):
                logger.debug("Window [%.1f - %.1f] rejected due to time "
                             "shift does not satisfy:  %.1f < %.1f < %.1f"
                             % (win.relative_starttime, win.relative_endtime,
                                tshift_min, win.cc_shift_in_seconds,
                                tshift_max))
                return False
            return True

        def reject_based_on_dlna(win):
            dlnA_min = self.config.dlna_reference - \
                self.config.dlna_acceptance_level[win.center]
            dlnA_max = self.config.dlna_reference + \
                self.config.dlna_acceptance_level[win.center]

            if not (dlnA_min < win.dlnA < dlnA_max):
                logger.debug("Window rejected due to amplitude fit: %f" %
                             win.dlnA)
                return False
            return True

        def reject_based_on_cc_value(win):
            if win.max_cc_value < self.config.cc_acceptance_level[win.center]:
                logger.debug("Window [%.1f - %.1f] rejected due to CC value "
                             "does not satisfy: %.3f < %.3f"
                             % (win.relative_starttime, win.relative_endtime,
                                self.config.cc_acceptance_level[win.center],
                                win.max_cc_value))
                return False

            return True

        for func, tag in zip([reject_based_on_time_shift, reject_based_on_dlna,
                              reject_based_on_cc_value],
                             ["tshift", "dlna", "cc"]):
            windows = list(filter(func, self.windows))
            self.separate_rejects(windows, tag)

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
            warnings.warn("The amplitude difference between data and"
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

        fig = plt.figure(figsize=(15, 5))

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

        ylim = plt.ylim()
        for tt in self.ttimes:
            if tt["name"].lower().startswith("p"):
                color = "#008c28"
            else:
                color = "#950000"
            # Don't need an offset as the time axis corresponds to time
            # since event.
            plt.vlines(tt["time"], ylim[0], ylim[1], color=color)
        plt.ylim(*ylim)

        plt.text(0.01, 0.92, 'Phase Arrivals', horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes)

        plt.axes([0.025, 0.51, 0.95, 0.4])
        plt.plot(times, self.observed.data, color="black", label="Observed")
        plt.plot(times, self.synthetic.data, color="red", label="Synthetic")
        plt.xlim(times[0], times[-1])
        plt.legend(prop={'size': 8})

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
            _l = win.relative_starttime - offset
            _r = win.relative_endtime - offset
            re = Rectangle((_l, plt.ylim()[0]), _r - _l,
                           plt.ylim()[1] - plt.ylim()[0], color="blue",
                           alpha=(win.max_cc_value ** 2) * 0.25)
            plt.gca().add_patch(re)
            plt.text(_l + buf, plt.ylim()[1],
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

        # print more information on the figure
        text = "Station id: %s  " % self.observed.id
        plt.text(0.75, 1.00, text, horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes,
                 fontsize=10)

        if self.event:
            text = "Source depth: %.2f km" % (self.event.depth_in_m/1000.)
            plt.text(0.75, 0.86, text, horizontalalignment='left',
                     verticalalignment='top', transform=ax.transAxes,
                     fontsize=10)

        if self.station and self.event:
            logger.debug("event lat, lon: {}, {}".format(
                self.event.latitude, self.event.longitude))
            logger.debug("station lat, lon: {}, {}".format(
                self.station.latitude, self.station.longitude))
            text = r"Epicenter distance:  {}$^\circ$   ".format(
                self.dist_in_deg)
            plt.text(0.75, 0.79, text, horizontalalignment='left',
                     verticalalignment='top', transform=ax.transAxes,
                     fontsize=10)

        if self.config.noise_end_index is not None:
            noise_start = \
                self.config.noise_start_index * self.observed.stats.delta \
                - offset
            noise_end = \
                self.config.noise_end_index * self.observed.stats.delta \
                - offset
            text = "Noise  Zone(s): [%-6.1f, %6.1f]" % (noise_start, noise_end)
            plt.text(0.75, 0.72, text, horizontalalignment='left',
                     verticalalignment='top', transform=ax.transAxes,
                     fontsize=10)

        if self.config.signal_end_index is not None and \
                self.config.signal_start_index is not None:
            signal_start = \
                self.config.signal_start_index * self.observed.stats.delta \
                - offset
            signal_end = \
                self.config.signal_end_index * self.observed.stats.delta \
                - offset
            text = "Signal Zone(s): [%-6.1f, %6.1f]" \
                   % (signal_start, signal_end)
            plt.text(0.75, 0.65, text, horizontalalignment='left',
                     verticalalignment='top', transform=ax.transAxes,
                     fontsize=10)

        for win in self.windows:
            _l = win.relative_starttime - offset
            _r = win.relative_endtime - offset
            re = Rectangle((_l, plt.ylim()[0]), _r - _l,
                           plt.ylim()[1] - plt.ylim()[0], color="blue",
                           alpha=(win.max_cc_value ** 2) * 0.25)
            plt.gca().add_patch(re)

        plt.ylim(0, plt.ylim()[1])

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close(fig)
