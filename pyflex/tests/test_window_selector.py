#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests suite for the window selector class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import inspect
import io
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


def test_reading_and_writing_windows(tmpdir):
    """
    Tests reading and writing of window sets.
    """
    config = pyflex.Config(
        min_period=50.0, max_period=150.0,
        stalta_waterlevel=0.08, tshift_acceptance_level=15.0,
        dlna_acceptance_level=1.0, cc_acceptance_level=0.80,
        c_0=0.7, c_1=4.0, c_2=0.0, c_3a=1.0, c_3b=2.0, c_4a=3.0, c_4b=10.0)

    ws = pyflex.WindowSelector(observed=OBS_DATA, synthetic=SYNTH_DATA,
                               config=config)
    windows = ws.select_windows()
    assert len(windows) > 0

    # Write/read to/from file.
    filename = os.path.join(str(tmpdir), "window.json")
    ws.write(filename)

    ws2 = pyflex.WindowSelector(observed=OBS_DATA, synthetic=SYNTH_DATA,
                                config=config)
    ws2.load(filename)
    assert ws.windows == ws2.windows
    os.remove(filename)

    # Write/read to/from open file.
    with open(filename, "w") as fh:
        ws.write(fh)

        ws2 = pyflex.WindowSelector(observed=OBS_DATA, synthetic=SYNTH_DATA,
                                    config=config)
        fh.seek(0, 0)
        ws2.load(filename)
    assert ws.windows == ws2.windows

    # Write/read to/from StringIO.
    with io.StringIO() as fh:
        ws.write(fh)

        ws2 = pyflex.WindowSelector(observed=OBS_DATA, synthetic=SYNTH_DATA,
                                    config=config)
        fh.seek(0, 0)
        ws2.load(filename)
    assert ws.windows == ws2.windows

    # Write/read to/from BytesIO.
    with io.BytesIO() as fh:
        ws.write(fh)

        ws2 = pyflex.WindowSelector(observed=OBS_DATA, synthetic=SYNTH_DATA,
                                    config=config)
        fh.seek(0, 0)
        ws2.load(filename)
    assert ws.windows == ws2.windows
