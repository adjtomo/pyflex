#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the window object.

Run with pytest.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np
import obspy

import pyflex

EXAMPLE_ID = "BW.FURT.00.BHZ"


def test_windows_time_calculations():
    """
    Tests time calculation for the windows.
    """
    start = obspy.UTCDateTime(2012, 1, 1)
    win = pyflex.window.Window(left=10, right=20, center=15,
                               time_of_first_sample=start, dt=0.5,
                               channel_id=EXAMPLE_ID, min_period=10.0)
    assert win.left == 10
    assert win.right == 20
    assert win.absolute_starttime == start + 5
    assert win.relative_starttime == 5.0
    assert win.absolute_endtime == start + 10
    assert win.relative_endtime == 10.0


def test_calc_criteria():
    """
    Simple sanity check.
    """
    start = obspy.UTCDateTime(2012, 1, 1)
    win = pyflex.window.Window(left=10, right=20, center=15,
                               time_of_first_sample=start, dt=0.5,
                               channel_id=EXAMPLE_ID, min_period=10.0)

    np.random.seed(12345)
    d = np.random.random(100)

    win._calc_criteria(d, d)

    # Same array is passed so the results should be perfect...Never test
    # floating point equality!
    assert abs(win.max_cc_value - 1.0) <= 1E-12
    assert abs(win.cc_shift) <= 1E-12
    assert abs(win.dlnA) <= 1E-12


def test_get_internal_indices():
    """
    Tests the get internal indices function.
    """
    start = obspy.UTCDateTime(2012, 1, 1)
    win = pyflex.window.Window(left=10, right=20, center=15,
                               time_of_first_sample=start, dt=0.5,
                               channel_id=EXAMPLE_ID, min_period=10.0)

    # Excludes the boundaries.
    idxs = win._get_internal_indices([1, 5, 9, 10, 11, 15, 19, 20, 21, 30])
    np.testing.assert_array_equal(idxs, np.array([11, 15, 19]))


def test_window_repr():
    """
    It has failed before...
    """
    start = obspy.UTCDateTime(2012, 1, 1)
    win = pyflex.window.Window(left=10, right=20, center=15,
                               time_of_first_sample=start, dt=0.5,
                               channel_id=EXAMPLE_ID, min_period=10.0)
    assert repr(win) == (
        "Window(left=10, right=20, center=15, channel_id=BW.FURT.00.BHZ, "
        "max_cc_value=None, cc_shift=None, dlnA=None)")


def test_custom_weight_fct():
    np.random.seed(12345)
    d = np.random.random(100)

    start = obspy.UTCDateTime(2012, 1, 1)
    win = pyflex.window.Window(left=10, right=20, center=15,
                               time_of_first_sample=start, dt=0.5,
                               channel_id=EXAMPLE_ID, min_period=1.0)
    win._calc_criteria(d, d)

    # Default is the window length in term of minimum period times the cc
    # value.
    assert (win.weight - 10.0) <= 1E-12

    # Now define a custom function that is simply 5 times the cc value.
    def weight_fct(win):
        return win.max_cc_value * 2.0

    win = pyflex.window.Window(left=10, right=20, center=15,
                               time_of_first_sample=start, dt=0.5,
                               channel_id=EXAMPLE_ID,
                               min_period=1.0, weight_function=weight_fct)
    win._calc_criteria(d, d)
    assert (win.weight - 5.0) <= 1E-12


def test_equality():
    """
    Tests (in)equality for windows.
    """
    np.random.seed(12345)
    d = np.random.random(100)

    start = obspy.UTCDateTime(2012, 1, 1)
    win = pyflex.window.Window(left=10, right=20, center=15,
                               time_of_first_sample=start, dt=0.5,
                               channel_id=EXAMPLE_ID, min_period=1.0)
    win2 = pyflex.window.Window(left=10, right=21, center=15,
                                time_of_first_sample=start, dt=0.5,
                                channel_id=EXAMPLE_ID, min_period=1.0)
    assert win == win
    assert win2 == win2
    assert win != win2
    assert win2 != win

    win._calc_criteria(d, d)
    assert win == win
    assert win2 == win2
    assert win != win2
    assert win2 != win


def test_window_to_dict_and_back(tmpdir):
    """
    Tests exporting and reimporting the window to a dictionary representation
    as a preliminary step towards JSON output.
    """
    np.random.seed(12345)
    d = np.random.random(100)
    start = obspy.UTCDateTime(2012, 1, 1)
    win = pyflex.window.Window(left=10, right=20, center=15,
                               time_of_first_sample=start, dt=0.5,
                               channel_id=EXAMPLE_ID, min_period=1.0)
    win._calc_criteria(d, d)

    output = win._get_json_content()

    expected = {
        "left_index": win.left,
        "right_index": win.right,
        "center_index": win.center,
        "channel_id": win.channel_id,
        "time_of_first_sample": win.time_of_first_sample,
        "max_cc_value": win.max_cc_value,
        "cc_shift_in_samples": win.cc_shift,
        "cc_shift_in_seconds": win.cc_shift_in_seconds,
        "dlnA":  win.dlnA,
        "dt": win.dt,
        "min_period": win.min_period,
        "phase_arrivals": [],
        "absolute_starttime": win.absolute_starttime,
        "absolute_endtime": win.absolute_endtime,
        "relative_starttime": win.relative_starttime,
        "relative_endtime": win.relative_endtime,
        "window_weight": win.weight}

    assert output == expected

    new_win = pyflex.window.Window._load_from_json_content(output)
    assert win == new_win

    win = pyflex.window.Window(left=10, right=20, center=15,
                               time_of_first_sample=start, dt=0.5,
                               channel_id=EXAMPLE_ID, min_period=1.0)

    # Without criteria.
    assert win == \
        pyflex.window.Window._load_from_json_content(win._get_json_content())

    # With criteria.
    win._calc_criteria(d, d)
    assert win == \
        pyflex.window.Window._load_from_json_content(win._get_json_content())

    # With phase arrivals.
    win.phase_arrivals = [
        {
            "d2T/dD2": -0.20390565693378448,
            "dT/dD": -1.909112811088562,
            "dT/dh": -0.12297123670578003,
            "phase_name": "PKKPdf",
            "take-off angle": 8.089553833007812,
            "time": 1744.7362060546875
        }, {
            "d2T/dD2": -0.0033220183104276657,
            "dT/dD": 8.340600967407227,
            "dT/dh": 0.09796148538589478,
            "phase_name": "pSdiff",
            "take-off angle": 142.0636444091797,
            "time": 1759.0133056640625}]
    assert win == \
        pyflex.window.Window._load_from_json_content(win._get_json_content())
