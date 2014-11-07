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
    Convenience function for selecting (and plotting) windows.

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
    :param station: The station information. Either a Pyflex Station object,
        or an ObsPy Inventory. If not given this information will be
        extracted from the data traces if either originates from a SAC file.
    :type station: A Pyflex :class:`~pyflex.Station` object or an ObsPy
        :class:`~obspy.station.inventory.Inventory` object
    :param plot: Plot the resulting windows.
    :type plot: bool
    :param plot_filename: If `plot` is True, this gives the possibility to
        specify a filename for the plot. The fileformat will be determines
        from that name. If not given, the plot will be shown with pylab's
        show() function.
    :type plot_filename: str
    """
    ws = WindowSelector(observed=observed, synthetic=synthetic, config=config,
                        event=event, station=station)
    windows = ws.select_windows()

    if plot:
        ws.plot(filename=plot_filename)

    return windows
