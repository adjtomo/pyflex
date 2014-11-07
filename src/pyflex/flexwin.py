#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convenience functions.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


from .window_selector import WindowSelector


def select_windows(observed, synthetic, config, event=None, station=None,
                   plot=False, plot_filename=None):
    """
    Convenience function for picking windows.

    :param observed: A trace with the observed data.
    :type observed: :class:`~obspy.core.trace.Trace`
    :param synthetic: A preprocessed :class:`~obspy.core.trace.Trace` object
        containing the synthetic data.
    :type synthetic: :class:`~obspy.core.trace.Trace`
    :param config: Configuration object.
    :type config: :class:`~.config.Config`
    """
    ws = WindowSelector(observed=observed, synthetic=synthetic, config=config,
                        event=event, station=station)
    windows = ws.select_windows()

    if plot:
        ws.plot(filename=plot_filename)

    return windows
