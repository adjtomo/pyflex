#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple class defining the windows.

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


class Window(object):
    """
    Class representing window candidates and final windows.
    """
    __slots__ = ["_left", "middle", "_right", "max_cc_value",
                 "cc_shift", "dlnA", "dt", "min_win_period"]

    def __init__(self, left, right, center, time_of_first_sample, dt,
                 min_period):
        """
        :param left: The array index of the left bound of the window.
        :type left: int
        :param right: The array index of the right bound of the window.
        :type right: int
        :param center: The array index of the central maximum of the window.
        :type center: int
        :param time_of_first_sample: The absolute time of the first sample
            in the array. Needed for the absolute time calculations.
        :type time_of_first_sample:
            :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param dt: The sample interval in seconds.
        :type dt: float
        :param min_period: The minimum period in seconds.
        :type min_period: float
        """
        self.left = left
        self.right = right
        self.center = center
        self.time_of_first_sample = time_of_first_sample
        self.max_cc_value = None
        self.cc_shift = None
        self.dlnA = None
        self.dt = float(dt)
        self.min_period = float(min_period)

    def _get_internal_indices(self, indices):
        """
        From a list of indices, return the ones inside this window excluding
        the borders..
        """
        return indices[(indices > self.left) & (indices < self.right)]

    @property
    def absolute_starttime(self):
        """
        Absolute time of the left border of this window.
        """
        return self.time_of_first_sample + self.dt * self.left

    @property
    def relative_starttime(self):
        """
        Relative time of the left border in seconds to the first sample in
        the array.
        """
        return self.dt * self.left

    @property
    def absolute_endtime(self):
        """
        Absolute time of the right border of this window.
        """
        return self.time_of_first_sample + self.dt * self.right

    @property
    def relative_endtime(self):
        """
        Relative time of the right border in seconds to the first sample in
        the array.
        """
        return self.dt * self.right

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, value):
        self._left = int(value)

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, value):
        self._right = int(value)

    @property
    def weight(self):
        """
        The weight of the window used for the weighted interval scheduling.

        The weight is window lengths in number of minimum period times the
        cross correlation coefficient.
        """
        return (self.right - self.left) * self.dt / self.min_period * \
            self.max_cc_value

    def __repr__(self):
        return (
            "Window(left={left}, right={right}, center={center}, "
            "max_cc_value={max_cc_value}, cc_shift={cc_shift}, dlnA={dlnA})"
            .format(left=self.left, right=self.right, center=self.center,
                    max_cc_value=self.max_cc_value, cc_shift=self.cc_shift,
                    dlnA=self.dlnA))

    def _xcorr_win(self, d, s):
        cc = np.correlate(d, s, mode="full")
        time_shift = cc.argmax() - len(d) + 1
        # Normalized cross correlation.
        max_cc_value = cc.max() / np.sqrt((s ** 2).sum() * (d ** 2).sum())
        return max_cc_value, time_shift

    def _dlnA_win(self, d, s):
        return 0.5 * np.log(np.sum(d ** 2) / np.sum(s ** 2))

    def _calc_criteria(self, d, s):
        d = d[self.left: self.right + 1]
        s = s[self.left: self.right + 1]
        v, shift = self._xcorr_win(d, s)
        dlnA = self._dlnA_win(d, s)
        self.max_cc_value = v
        self.cc_shift = shift
        self.dlnA = dlnA
