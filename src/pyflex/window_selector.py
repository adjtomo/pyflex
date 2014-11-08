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
with standard_library.hooks():
    import itertools

import numpy as np
import obspy
import obspy.station
from obspy.core.util import geodetics
from obspy.signal.filter import envelope
from obspy.taup import getTravelTimes
import warnings

from . import PyflexError, PyflexWarning, utils, logger, Event, Station
from .stalta import sta_lta
from .window import Window
from .interval_scheduling import schedule_weighted_intervals


class WindowSelector(object):
    """
    Low level window selector internally used by Pyflex.
    """
    def __init__(self, observed, synthetic, config, event=None, station=None):
        """
        :param observed: The preprocessed, observed waveform.
        :type observed: :class:`~obspy.core.trace.Trace` or single component
            :class:`~obspy.core.stream.Stream`
        :param observed: The preprocessed, synthetic waveform.
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

    def _parse_event_and_station(self):
        """
        Parse the event and station information.
        """
        # Parse the event.
        if self.event and not isinstance(self.event, Event):
            # It might be an ObsPy event catalog.
            if isinstance(self.event, obspy.core.Catalog):
                if len(self.event) != 1:
                    raise PyflexError("The event catalog must contain "
                                      "exactly one event.")
                self.event = self.event[0]
            # It might be an ObsPy event object.
            if isinstance(self.event, obspy.core.Event):
                if not self.event.origins:
                    raise PyflexError("Event does not contain an origin.")
                origin = self.event.preferred_origin() or self.event.origins[0]
                self.event = Event(latitude=origin.latitude,
                                   longitude=origin.longitude,
                                   depth_in_m=origin.depth)
            else:
                raise PyflexError("Could not parse the event. Unknown type.")

        # Parse the station information if it is an obspy inventory object.
        if isinstance(self.station, obspy.station.Inventory):
            net = self.observed.stats.network
            sta = self.observed.stats.station
            # Workaround for ObsPy 0.9.2 Newer version have a get
            # coordiantes method...
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
            self.station = Station(latitude=station.latitude,
                                   longitude=station.longitude)

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
            values = (sac.evla, sac.evlo, sac.evdp, sac.stla, sac.stlo)
            # Invalid value in sac.
            if -12345.0 in values:
                return
            if not self.station:
                self.station = Station(latitude=values[3], longitude=values[4])
                logger.info("Extracted station information from %s SAC file."
                            % ftype)
            if not self.event:
                self.event = Event(latitude=values[0], longitude=values[1],
                                   depth_in_m=values[2] * 1000.0)
                logger.info("Extracted event information from %s SAC file." %
                            ftype)

    def select_windows(self):
        """
        Launch the window selection.
        """
        logger.info("Calculating envelope of synthetics.")
        self.synthetic_envelope = envelope(self.synthetic.data)
        logger.info("Calculating STA/LTA.")
        self.stalta = sta_lta(self.synthetic_envelope,
                              self.observed.stats.delta,
                              self.config.min_period)
        self.peaks, self.troughs = utils.find_local_extrema(self.stalta)

        # Perform all window selection steps.
        self.initial_window_selection()
        # Reject windows based on traveltime if event and station
        # information is given. This will also fill self.ttimes.
        if self.event and self.station:
            self.calculate_ttimes()
            self.reject_on_traveltimes()
        else:
            logger.info("No rejection based on traveltime possible. Event "
                        "and/or station information is not available.")
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
        self.reject_based_on_data_fit_criteria()
        self.schedule_weighted_intervals()

        return self.windows

    def check_data_quality(self):
        """
        Checks the data quality by estimating signal to noise ratios.
        """
        if self.config.noise_end_index is None:
            if not self.ttimes:
                raise PyflexError(
                    "Cannot check data quality as the noise end index is not "
                    "given and station and/or event information is not "
                    "available so the theoretical arrival times cannot be "
                    "calculated.")
            self.config.noise_end_index = \
                self.ttimes[0]["time"] - self.config.min_period
        if self.config.signal_start_index is None:
            self.config.signal_start_index = self.config.noise_end_index

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

        min_time = self.ttimes[0]["time"] - self.config.min_period
        max_time = dist_in_km / self.config.min_surface_wave_velocity

        self.windows = [win for win in self.windows
                        if (win.right >= min_time) and (win.left <= max_time)]
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
                    time_of_first_sample=self.synthetic.stats.starttime,
                    dt=self.observed.stats.delta,
                    min_period=self.config.min_period))

        logger.info("Initial window selection yielded %i possible windows." %
                    len(self.windows))

    def remove_duplicates(self):
        """
        Filter to remove duplicate windows.
        """
        new_windows = {}
        for window in self.windows:
            tag = (window.left, window.right)
            if tag not in new_windows:
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

        self.windows = list(itertools.ifilter(
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
                return False
            if not (dlnA_min < win.dlnA < dlnA_max):
                return False
            if win.max_cc_value < self.config.cc_acceptance_level[win.center]:
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

        plt.figure(figsize=(15, 5))
        plt.subplot(211)

        plt.plot(self.observed.data, color="black")
        plt.plot(self.synthetic.data, color="red")
        plt.xlim(0, len(self.observed.data))

        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.set_xticks([])
        ax.set_yticks([])

        plt.text(0.01, 0.99, 'seismograms', horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes)

        for win in self.windows:
            re = Rectangle((win.left, plt.ylim()[0]), win.right - win.left,
                           plt.ylim()[1] - plt.ylim()[0], color="blue",
                           alpha=0.3)
            plt.gca().add_patch(re)

        plt.subplot(212)
        plt.plot(self.stalta, color="blue")
        plt.plot(self.config.stalta_waterlevel, linestyle="dashed",
                 color="blue")
        plt.xlim(0, len(self.stalta))

        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_yticks([])
        ax.xaxis.set_ticks_position('bottom')

        plt.text(0.01, 0.99, 'STA/LTA', horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes)

        for win in self.windows:
            re = Rectangle((win.left, plt.ylim()[0]), win.right - win.left,
                           plt.ylim()[1] - plt.ylim()[0], color="blue",
                           alpha=0.3)
            plt.gca().add_patch(re)

        plt.tight_layout()

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
