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
import obspy


class Window(object):
    """
    Class representing window candidates and final windows.
    """
    def __init__(self, left, right, center, time_of_first_sample, dt,
                 min_period, channel_id, weight_function=None):
        """
        The optional ``weight_function`` parameter can be used to customize
        the weight of the window. Its single parameter is an instance of the
        window. The following example will create a window function that
        does exactly the same as the default weighting function.

        >>> def weight_function(win):
        ...     return ((win.right - win.left) * win.dt / win.min_period *
        ...         win.max_cc_value)


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
        :param channel_id: The id of the channel of interest. Needed for a
            useful serialization.
        :type channel_id: str
        :param weight_function: Function determining the window weight. The
            only argument of the function is a window instance.
        :type weight_function: function
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
        self.channel_id = channel_id
        self.phase_arrivals = []
        self.weight_function = weight_function

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self == other

    @staticmethod
    def _load_from_json_content(win):
        """
        Load a dictionary coming from a JSON file and parse it to Window
        object.

        :param win: A dictionary containing window information from the JSON
            file.
        :type win: dict
        :returns: A new window object.
        :rtype: :class:`~pyflex.window.Window`
        """
        necessary_keys = set([
            "left_index", "right_index", "center_index", "channel_id",
            "time_of_first_sample", "max_cc_value", "cc_shift_in_samples",
            "cc_shift_in_seconds", "dlnA", "dt", "min_period",
            "phase_arrivals", "absolute_starttime", "absolute_endtime",
            "relative_starttime", "relative_endtime", "window_weight"])
        missing_keys = necessary_keys.difference(set(win.keys()))

        if missing_keys:
            raise ValueError(
                "Window JSON file misses the following keys:\n%s" %
                ", ".join(missing_keys))

        new_win = Window(
            left=win["left_index"], right=win["right_index"],
            center=win["center_index"],
            time_of_first_sample=obspy.UTCDateTime(
                win["time_of_first_sample"]),
            dt=win["dt"], min_period=win["min_period"],
            channel_id=win["channel_id"])
        new_win.max_cc_value = win["max_cc_value"]
        new_win.cc_shift = win["cc_shift_in_samples"]
        new_win.dlnA = win["dlnA"]
        new_win.phase_arrivals = win["phase_arrivals"]

        return new_win

    def _get_json_content(self):
        """
        Returns the window in a representation suitable for inclusion as a
        JSON file.
        """
        info = {
            "left_index": self.left,
            "right_index": self.right,
            "center_index": self.center,
            "channel_id": self.channel_id,
            "time_of_first_sample": self.time_of_first_sample,
            "max_cc_value":  self.max_cc_value,
            "cc_shift_in_samples":  self.cc_shift,
            "cc_shift_in_seconds":  self.cc_shift_in_seconds,
            "dlnA":  self.dlnA,
            "dt": self.dt,
            "min_period": self.min_period,
            "phase_arrivals": self.phase_arrivals,
            "absolute_starttime": self.absolute_starttime,
            "absolute_endtime": self.absolute_endtime,
            "relative_starttime": self.relative_starttime,
            "relative_endtime": self.relative_endtime,
            "window_weight": self.weight}

        return info

    def _get_internal_indices(self, indices):
        """
        From a list of indices, return the ones inside this window excluding
        the borders..
        """
        indices = np.array(indices)
        return indices[(indices > self.left) & (indices < self.right)]

    @property
    def cc_shift_in_seconds(self):
        return self.cc_shift * self.dt if self.cc_shift is not None else None

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
        Either calls a potentially passed window weight function or defaults
        to the window length in number of minimum periods times the cross
        correlation coefficient.
        """
        if self.weight_function:
            return self.weight_function(self)
        if self.max_cc_value is None:
            return None
        return (self.right - self.left) * self.dt / self.min_period * \
            self.max_cc_value

    def __repr__(self):
        return (
            "Window(left={left}, right={right}, center={center}, "
            "channel_id={channel_id}, "
            "max_cc_value={max_cc_value}, cc_shift={cc_shift}, dlnA={dlnA})"
            .format(left=self.left, right=self.right, center=self.center,
                    channel_id=self.channel_id, max_cc_value=self.max_cc_value,
                    cc_shift=self.cc_shift, dlnA=self.dlnA))

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
