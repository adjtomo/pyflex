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


def select_windows(observed, synthetic, config):
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
    ws = WindowSelector(observed=observed, synthetic=synthetic, config=config)
    return ws.select_windows()


def plot_windows(observed, synthetic, config, filename=None):
    """
    Function plotting the selected windows. Will also return the windows
    similar to the select_windows() function.

    :param observed: A trace with the observed data.
    :type observed: :class:`~obspy.core.trace.Trace`
    :param synthetic: A preprocessed :class:`~obspy.core.trace.Trace` object
        containing the synthetic data.
    :type synthetic: :class:`~obspy.core.trace.Trace`
    :param config: Configuration object.
    :type config: :class:`~.config.Config`
    """
    # Lazy imports to not import matplotlib all the time.
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ws = WindowSelector(observed=observed, synthetic=synthetic, config=config)
    windows = ws.select_windows()

    plt.figure(figsize=(15, 5))
    plt.subplot(211)

    plt.plot(ws.observed.data, color="black")
    plt.plot(ws.synthetic.data, color="red")
    plt.xlim(0, len(ws.observed.data))

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.text(0.01, 0.99, 'seismograms', horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes)

    for win in windows:
        re = Rectangle((win.left, plt.ylim()[0]), win.right - win.left,
                       plt.ylim()[1] - plt.ylim()[0], color="blue", alpha=0.3)
        plt.gca().add_patch(re)

    plt.subplot(212)
    plt.plot(ws.stalta, color="blue")
    plt.hlines(config.stalta_base, 0, len(ws.observed.data),
               linestyle="dashed", color="blue")
    plt.xlim(0, len(ws.stalta))

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_yticks([])
    ax.xaxis.set_ticks_position('bottom')

    plt.text(0.01, 0.99, 'STA/LTA', horizontalalignment='left',
             verticalalignment='top', transform=ax.transAxes)

    for win in windows:
        re = Rectangle((win.left, plt.ylim()[0]), win.right - win.left,
                       plt.ylim()[1] - plt.ylim()[0], color="blue", alpha=0.3)
        plt.gca().add_patch(re)

    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
