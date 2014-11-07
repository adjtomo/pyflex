#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pyflex test suite.

Run with pytest.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import inspect
import obspy
import os

import pyflex

# Most generic way to get the data folder path.
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")


def test_window_selection():
    """
    Simple tests assuring it at least runs through.
    """
    # Read data.
    obs_data = obspy.read(os.path.join(
        DATA, "1995.122.05.32.16.0000.II.ABKT.00.LHZ.D.SAC"))
    synth_data = obspy.read(os.path.join(
        DATA, "ABKT.II.LHZ.semd.sac"))

    # Preprocess it.
    obs_data.detrend("linear")
    obs_data.taper(max_percentage=0.05, type="hann")
    obs_data.filter("bandpass", freqmin=1.0 / 150.0, freqmax=1.0 / 50.0,
                    corners=4, zerophase=True)

    synth_data.detrend("linear")
    synth_data.taper(max_percentage=0.05, type="hann")
    synth_data.filter("bandpass", freqmin=1.0 / 150.0, freqmax=1.0 / 50.0,
                      corners=4, zerophase=True)

    config = pyflex.Config(
        min_period=50.0, max_period=150.0,
        stalta_base=0.1, tshift_base=15.0, dlna_base=1.0, cc_base=0.80,
        c_0=0.7, c_1=3.5, c_2=0.0, c_3a=1.0, c_3b=2.0, c_4a=3.0, c_4b=10.0)

    windows = pyflex.select_windows(obs_data, synth_data, config)
    assert len(windows)
