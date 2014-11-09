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
import numpy as np
import obspy
import os

import pyflex

# Most generic way to get the data folder path.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")

# Prepare data to be able to use it in all tests.
OBS_DATA = obspy.read(os.path.join(
    DATA_DIR, "1995.122.05.32.16.0000.II.ABKT.00.LHZ.D.SAC"))
SYNTH_DATA = obspy.read(os.path.join(DATA_DIR, "ABKT.II.LHZ.semd.sac"))

# Preprocess it.
OBS_DATA.detrend("linear")
OBS_DATA.taper(max_percentage=0.05, type="hann")
OBS_DATA.filter("bandpass", freqmin=1.0 / 150.0, freqmax=1.0 / 50.0,
                corners=4, zerophase=True)
SYNTH_DATA.detrend("linear")
SYNTH_DATA.taper(max_percentage=0.05, type="hann")
SYNTH_DATA.filter("bandpass", freqmin=1.0 / 150.0, freqmax=1.0 / 50.0,
                  corners=4, zerophase=True)


def test_window_selection():
    """
    This WILL need to be adjusted if any part of the algorithm changes!

    The settings for this test are more or less the same as for the test
    data example in the original FLEXWIN package.
    """
    config = pyflex.Config(
        min_period=50.0, max_period=150.0,
        stalta_waterlevel=0.08, tshift_acceptance_level=15.0,
        dlna_acceptance_level=1.0, cc_acceptance_level=0.80,
        c_0=0.7, c_1=4.0, c_2=0.0, c_3a=1.0, c_3b=2.0, c_4a=3.0, c_4b=10.0)

    windows = pyflex.select_windows(OBS_DATA, SYNTH_DATA, config)
    assert len(windows) == 7

    assert [_i.left for _i in windows] == [1551, 2221, 2709, 2960, 3353, 3609,
                                           3983]
    assert [_i.right for _i in windows] == [1985, 2709, 2960, 3172, 3609, 3920,
                                            4442]
    np.testing.assert_allclose(
        np.array([_i.max_cc_value for _i in windows]),
        np.array([0.95740629373181685, 0.96646803651993862, 0.9633571597878805,
                  0.98249546895396034, 0.96838753962768898,
                  0.88501979275369003, 0.82529382012185848]))
    assert [_i.cc_shift for _i in windows] == [-3, 0, -5, -5, -6, 4, -9]
    np.testing.assert_allclose(
        np.array([_i.dlnA for _i in windows]),
        np.array([0.074690084388978839, 0.12807961376836777,
                  -0.19276977567364437, 0.18556340842688038,
                  0.093674448597561411, -0.11885913254077075,
                  -0.63865703707265198]))
