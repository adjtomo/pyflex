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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images as mpl_compare_images
import numpy as np
import obspy
import os

import pyflex

EXAMPLE_ID = "BW.FURT.00.BHZ"

# Most generic way to get the data folder path.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")

# Baseline images for the plotting test.
IMAGE_DIR = os.path.join(os.path.dirname(DATA_DIR), "baseline_images")

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


def reset_matplotlib():
    """
    Reset matplotlib to a common default.
    """
    # Set all default values.
    mpl.rcdefaults()
    # Force agg backend.
    plt.switch_backend('agg')
    # These settings must be hardcoded for running the comparision tests and
    # are not necessarily the default values.
    mpl.rcParams['font.family'] = 'Bitstream Vera Sans'
    mpl.rcParams['text.hinting'] = False
    # Not available for all matplotlib versions.
    try:
        mpl.rcParams['text.hinting_factor'] = 8
    except KeyError:
        pass
    import locale
    locale.setlocale(locale.LC_ALL, str('en_US.UTF-8'))


def images_are_identical(image_name, temp_dir, dpi=None):
    """
    Partially copied from ObsPy. Used to check images for equality.
    """
    image_name += os.path.extsep + "png"
    expected = os.path.join(IMAGE_DIR, image_name)
    actual = os.path.join(temp_dir, image_name)

    if dpi:
        plt.savefig(actual, dpi=dpi)
    else:
        plt.savefig(actual)
    plt.close()

    assert os.path.exists(expected)
    assert os.path.exists(actual)

    # Use a reasonably high tolerance to get around difference with different
    # freetype and possibly agg versions. matplotlib uses a tolerance of 13.
    result = mpl_compare_images(expected, actual, 5, in_decorator=True)
    if result is not None:
        print(result)
    assert result is None


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
    assert len(windows) == 9

    lefties = np.array([_i.left for _i in windows])
    righties = np.array([_i.right for _i in windows])

    np.testing.assert_allclose(
        lefties,
        np.array([1551, 2221, 2709, 2960, 3353, 3609, 3983, 4715, 4962]),
        atol=3)
    np.testing.assert_allclose(
        righties,
        np.array([1985, 2709, 2960, 3172, 3609, 3920, 4442, 4962, 5207]),
        atol=3)

    np.testing.assert_allclose(
        np.array([_i.max_cc_value for _i in windows]),
        np.array([0.95740629, 0.96646804, 0.96335716, 0.98249547, 0.96838754,
                  0.88501979, 0.82529382, 0.92953344, 0.92880873]), rtol=1E-2)

    assert [_i.cc_shift for _i in windows] == [-3, 0, -5, -5, -6, 4, -9, -1, 7]
    np.testing.assert_allclose(
        np.array([_i.dlnA for _i in windows]),
        np.array([0.07469, 0.12808, -0.19277, 0.185563, 0.093674, -0.118859,
                  -0.638657, 0.25942, 0.106571]), rtol=1E-2)

    # Assert the phases of the first window.
    assert sorted([_i["phase_name"] for _i in windows[0].phase_arrivals]) == \
        ['PKIKP', 'PKIKS', 'PKiKP', 'PP', 'SKIKP', 'SKiKP', 'pPKIKP', 'pPKiKP',
         'sPKIKP', 'sPKiKP']


def test_cc_config_setting():
    """
    Make sure setting the CC threshold does something.
    """
    config = pyflex.Config(
        min_period=50.0, max_period=150.0,
        stalta_waterlevel=0.08, tshift_acceptance_level=15.0,
        dlna_acceptance_level=1.0, cc_acceptance_level=0.95,
        c_0=0.7, c_1=4.0, c_2=0.0, c_3a=1.0, c_3b=2.0, c_4a=3.0, c_4b=10.0)

    windows = pyflex.select_windows(OBS_DATA, SYNTH_DATA, config)
    assert np.all(np.array([_i.max_cc_value for _i in windows]) >= 0.95)


def test_custom_weight_function():
    """
    Test the custom weight function. Set the weight of every window with a
    CC of smaller then 95 to 0.0.
    """
    def weight_function(win):
        if win.max_cc_value < 0.95:
            return 0.0
        else:
            return 10.0

    config = pyflex.Config(
        min_period=50.0, max_period=150.0,
        stalta_waterlevel=0.08, tshift_acceptance_level=15.0,
        dlna_acceptance_level=1.0, cc_acceptance_level=0.80,
        c_0=0.7, c_1=4.0, c_2=0.0, c_3a=1.0, c_3b=2.0, c_4a=3.0, c_4b=10.0,
        window_weight_fct=weight_function)

    windows = pyflex.select_windows(OBS_DATA, SYNTH_DATA, config)
    assert np.all(np.array([_i.max_cc_value for _i in windows]) >= 0.95)

    # Not setting it will result in the default value.
    config.window_weight_fct = None
    windows = pyflex.select_windows(OBS_DATA, SYNTH_DATA, config)
    assert bool(np.all(np.array([_i.max_cc_value for _i in windows]) >=
                       0.95)) is False


def test_runs_without_event_information(recwarn):
    """
    Make sure it runs without event information. Some things will not work
    but it will at least not crash.
    """
    config = pyflex.Config(
        min_period=50.0, max_period=150.0,
        stalta_waterlevel=0.08, tshift_acceptance_level=15.0,
        dlna_acceptance_level=1.0, cc_acceptance_level=0.80,
        c_0=0.7, c_1=4.0, c_2=0.0, c_3a=1.0, c_3b=2.0, c_4a=3.0, c_4b=10.0)

    obs = OBS_DATA[0].copy()
    syn = SYNTH_DATA[0].copy()

    # Remove the sac header information.
    del obs.stats.sac
    del syn.stats.sac

    recwarn.clear()
    windows = pyflex.select_windows(obs, syn, config)

    # This will actually result in a bunch more windows as before. So it
    # is always a good idea to specify the event and station information!
    assert len(windows) == 12

    assert len(recwarn.list) == 1
    w = recwarn.list[0]
    assert w.category == pyflex.PyflexWarning
    assert "Event and/or station information is not available".lower() in \
        str(w.message).lower()

    # No phases should be attached as they cannot be calculated.
    phases = []
    for win in windows:
        phases.extend(win.phase_arrivals)

    assert phases == []


def test_event_information_extraction():
    """
    Event information can either be passed or read from sac files.
    """
    config = pyflex.Config(min_period=50.0, max_period=150.0)

    # If not passed, it is read from sac files, if available.
    ws = pyflex.window_selector.WindowSelector(OBS_DATA, SYNTH_DATA, config)
    assert abs(ws.event.latitude - -3.77) <= 1E-5
    assert abs(ws.event.longitude - -77.07) <= 1E-5
    assert abs(ws.event.depth_in_m - 112800.00305) <= 1E-5
    assert ws.event.origin_time == \
        obspy.UTCDateTime(1995, 5, 2, 6, 6, 13, 900000)

    # If it passed, the passed event will be used.
    ev = pyflex.Event(1, 2, 3, obspy.UTCDateTime(2012, 1, 1))
    ws = pyflex.window_selector.WindowSelector(OBS_DATA, SYNTH_DATA, config,
                                               event=ev)
    assert ws.event == ev

    # Alternatively, an ObsPy Catalog or Event object can be passed which
    # opens the gate to more complex workflows.
    cat = obspy.readEvents()
    cat.events = cat.events[:1]
    event = cat[0]

    ev = pyflex.Event(event.origins[0].latitude, event.origins[0].longitude,
                      event.origins[0].depth, event.origins[0].time)

    # Test catalog.
    ws = pyflex.window_selector.WindowSelector(OBS_DATA, SYNTH_DATA, config,
                                               event=cat)
    assert ws.event == ev

    # Test event.
    ws = pyflex.window_selector.WindowSelector(OBS_DATA, SYNTH_DATA, config,
                                               event=cat[0])
    assert ws.event == ev


def test_station_information_extraction():
    """
    Station information can either be passed or read from sac files.
    """
    import obspy.station

    config = pyflex.Config(min_period=50.0, max_period=150.0)

    # If not passed, it is read from sac files, if available.
    ws = pyflex.window_selector.WindowSelector(OBS_DATA, SYNTH_DATA, config)
    assert abs(ws.station.latitude - 37.930401) < 1E-5
    assert abs(ws.station.longitude - 58.1189) < 1E-5

    # The other option is an inventory object. Assemble a dummy one.
    inv = obspy.station.Inventory(networks=[], source="local")
    net = obspy.station.Network(code=OBS_DATA[0].stats.network)
    sta = obspy.station.Station(code=OBS_DATA[0].stats.station, latitude=1.0,
                                longitude=2.0, elevation=3.0)
    inv.networks = [net]
    net.stations = [sta]

    ws = pyflex.window_selector.WindowSelector(OBS_DATA, SYNTH_DATA, config,
                                               station=inv)
    assert ws.station == pyflex.Station(1.0, 2.0)


def test_run_with_data_quality_checks():
    """
    Run with data quality checks.
    """
    config = pyflex.Config(
        min_period=50.0, max_period=150.0,
        stalta_waterlevel=0.08, tshift_acceptance_level=15.0,
        dlna_acceptance_level=1.0, cc_acceptance_level=0.80,
        c_0=0.7, c_1=4.0, c_2=0.0, c_3a=1.0, c_3b=2.0, c_4a=3.0, c_4b=10.0,
        check_global_data_quality=True)

    windows = pyflex.select_windows(OBS_DATA, SYNTH_DATA, config)
    # The data in this case is so good that nothing should have changed.
    assert len(windows) == 9


def test_window_plotting(tmpdir):
    reset_matplotlib()

    config = pyflex.Config(
        min_period=50.0, max_period=150.0,
        stalta_waterlevel=0.08, tshift_acceptance_level=15.0,
        dlna_acceptance_level=1.0, cc_acceptance_level=0.80,
        c_0=0.7, c_1=4.0, c_2=0.0, c_3a=1.0, c_3b=2.0, c_4a=3.0, c_4b=10.0)

    pyflex.select_windows(OBS_DATA, SYNTH_DATA, config, plot=True)
    images_are_identical("picked_windows", str(tmpdir))


def test_window_merging_strategy():
    """
    Pyflex can also merge windows.
    """
    config = pyflex.Config(
        min_period=50.0, max_period=150.0,
        stalta_waterlevel=0.08, tshift_acceptance_level=15.0,
        dlna_acceptance_level=1.0, cc_acceptance_level=0.80,
        c_0=0.7, c_1=4.0, c_2=0.0, c_3a=1.0, c_3b=2.0, c_4a=3.0, c_4b=10.0,
        resolution_strategy="merge")

    windows = pyflex.select_windows(OBS_DATA, SYNTH_DATA, config)
    assert len(windows) == 4


def test_settings_arrays_as_config_values():
    """
    Tests that arrays can be set as config values.
    """
    npts = OBS_DATA[0].stats.npts
    stalta_waterlevel = 0.08 * np.ones(npts)
    tshift_acceptance_level = 15.0 * np.ones(npts)
    dlna_acceptance_level = 1.0 * np.ones(npts)
    cc_acceptance_level = 0.80 * np.ones(npts)
    s2n_limit = 1.5 * np.ones(npts)
    config = pyflex.Config(
        min_period=50.0, max_period=150.0,
        stalta_waterlevel=stalta_waterlevel,
        tshift_acceptance_level=tshift_acceptance_level,
        dlna_acceptance_level=dlna_acceptance_level,
        cc_acceptance_level=cc_acceptance_level, s2n_limit=s2n_limit,
        c_0=0.7, c_1=4.0, c_2=0.0, c_3a=1.0, c_3b=2.0, c_4a=3.0, c_4b=10.0)

    windows = pyflex.select_windows(OBS_DATA, SYNTH_DATA, config)
    assert len(windows) == 9
