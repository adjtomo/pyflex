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
from obspy.signal.filter import envelope
import warnings

from . import PyflexError, utils, logger
from .stalta import sta_lta
from .window import Window
from .interval_scheduling import schedule_weighted_intervals


class WindowSelector(object):
    def __init__(self, observed, synthetic, config):
        self.observed = observed
        self.synthetic = synthetic
        self._sanity_checks()

        # Copy to not modify the original data.
        self.observed = self.observed.copy()
        self.synthetic = self.synthetic.copy()
        self.observed.data = np.ascontiguousarray(self.observed.data)
        self.synthetic.data = np.ascontiguousarray(self.synthetic.data)

        self.config = config

        self.windows = []

    def select_windows(self):
        self.synthetic_envelope = envelope(self.synthetic.data)
        self.stalta = sta_lta(self.synthetic_envelope,
                              self.observed.stats.delta,
                              self.config.min_period)
        self.peaks, self.troughs = utils.find_local_extrema(self.stalta)

        # Perform all window selection steps.
        self.initial_window_selection()
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
            if self.stalta[peak] <= self.config.stalta_base:
                continue
            smaller_troughs = self.troughs[self.troughs < peak]
            larger_troughs = self.troughs[self.troughs > peak]

            for left, right in itertools.product(smaller_troughs,
                                                 larger_troughs):
                self.windows.append(Window(
                    left=left, right=right, center=peak,
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
            waterlevel_midpoint = self.config.c_0 * self.config.stalta_base
            internal_minima = win.get_internal_indices(self.troughs)
            return not np.any(self.stalta[internal_minima] <=
                              waterlevel_midpoint)

        self.windows = list(filter(filter_window_minima, self.windows))
        logger.info("Water level rejection retained %i windows" %
                    len(self.windows))

    def reject_on_prominence_of_central_peak(self):
        """
        Equivalent to reject_on_prominence() in the original flexwin code.
        """
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
                # find value of time decay function
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
        return self.config.c_1 * self.config.min_period / \
            self.observed.stats.delta

    def reject_windows_based_on_minimum_length(self):
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
            win.calc_criteria(self.observed.data, self.synthetic.data)

        def reject_based_on_criteria(win):
            tshift_min = self.config.tshift_reference - self.config.tshift_base
            tshift_max = self.config.tshift_reference + self.config.tshift_base
            dlnA_min = self.config.dlna_reference - self.config.dlna_base
            dlnA_max = self.config.dlna_reference + self.config.dlna_base

            if not (tshift_min < win.cc_shift *
                    self.observed.stats.delta < tshift_max):
                return False
            if not (dlnA_min < win.dlnA < dlnA_max):
                return False
            if win.max_cc_value < self.config.cc_base:
                return False
            return True

        self.windows = list(filter(reject_based_on_criteria, self.windows))
        logger.info("Rejection based on data fit criteria retained %i windows."
                    % len(self.windows))

    def _sanity_checks(self):
        """
        Perform a number of basic sanity checks to assure the data is valid in a
        certain sense.

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
            warnings.warn("The amplitude difference between data and synthetic "
                          "is fairly large.")

        # Also check the components of the data to avoid silly mistakes of users.
        if len(set([self.observed.stats.channel[-1].upper(),
                    self.synthetic.stats.channel[-1].upper()])) != 1:
            warnings.warn("The orientation code of synthetic and observed data "
                          "is not equal.")

    def plot(self, filename=None):
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
        plt.hlines(self.config.stalta_base, 0, len(self.observed.data),
                   linestyle="dashed", color="blue")
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
