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
class Window(object):
    """
    Class representing window candidates and final windows.
    """
    __slots__ = ["_left", "middle", "_right", "max_cc_value",
                 "cc_shift", "dlnA", "dt", "min_win_period"]

    def __init__(self, left, right, center, dt, min_period):
        """
        :param left: The array index of the left bound of the window.
        :param right: The array index of the right bound of the window.
        :param middle: The array index of the central maximum of the window.
        :param dt: The sample interval in seconds.
        :param min_period: The minimum period in seconds.
        """
        self.left = left
        self.right = right
        self.center = center
        self.max_cc_value = None
        self.cc_shift = None
        self.dlnA = None
        self.dt = dt
        self.min_period = min_period

    @property
    def left(self):
        """
        Use getters/setters to avoid type issues.
        """
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
            "Window(left={left}, right={right}, middle={middle}, "
            "max_cc_value={max_cc_value}, cc_shift={cc_shift}, dlnA={dlnA})"
            .format(left=self.left, right=self.right, middle=self.middle,
                    max_cc_value=self.max_cc_value, cc_shift=self.cc_shift,
                    dlnA=self.dlnA))