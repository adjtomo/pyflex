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

from . import utils
from .stalta import sta_lta
from .window import Window


class WindowSelector(object):
    def __init__(self, observed, synthetic, dt, config):
        self.observed = observed
        self.synthetic = synthetic
        self.dt = dt
        self.config = config

        self.windows = []

        self.stalta = sta_lta(np.abs(self.synthetic), self.dt,
                              self.config.min_period)
        self.peaks, self.troughs = utils.find_local_extrema(self.stalta)

        self.launch_selection()

    def launch_selection(self):
        self.initial_window_selection()

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
                self.windows.append(Window(left=left, right=right,
                                           middle=peak))

    def filter_window_minima(self, win):
        """
        Filter function rejecting windows whose internal minima are below the
        water level of the windows peak. This is equivalent to the
        reject_on_water_level() function in flexwin.
        """
        waterlevel_midpoint = self.config.c_0 * self.config.stalta_base
        internal_minima = win.get_internal_indices(self.troughs)
        return not np.any(self.stalta[internal_minima] <= waterlevel_midpoint)

    def reject_windows_based_on_minimum_length(self, min_length):
        self.windows = list(itertools.filter(
            lambda x: (x.right - x.left) > min_length,  self.windows))
