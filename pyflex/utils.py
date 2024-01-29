#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functionality for pyflex.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
    adjTomo Dev Team (adjtomo@gmail.com), 2022
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import itertools
import numpy as np
from scipy.signal import argrelextrema
from obspy.geodetics import degrees2kilometers


def find_local_extrema(data):
    """
    Function finding local extrema. It can also deal with flat extrema,
    e.g. a flat top or bottom. In that case the first index of all flat
    values will be returned.

    Returns a tuple of maxima and minima indices.
    """
    diff = np.diff(data)
    flats = np.argwhere(diff == 0)

    # Discard neighbouring flat points.
    new_flats = list(flats[0:1])
    for i, j in zip(flats[:-1], flats[1:]):
        if j - i == 1:
            continue
        new_flats.append(j)
    flats = new_flats

    maxima = []
    minima = []

    # Go over each flats position and check if its a maxima/minima.
    for idx in flats:
        l_type = "left"
        r_type = "right"
        for i in itertools.count():
            this_idx = idx - i - 1
            if diff[this_idx] < 0:
                l_type = "minima"
                break
            elif diff[this_idx] > 0:
                l_type = "maxima"
                break
        for i in itertools.count():
            this_idx = idx + i + 1
            if this_idx >= len(diff):
                break
            if diff[this_idx] < 0:
                r_type = "maxima"
                break
            elif diff[this_idx] > 0:
                r_type = "minima"
                break
        if r_type != l_type:
            continue
        if r_type == "maxima":
            maxima.append(int(idx))
        else:
            minima.append(int(idx))

    maxs = set(list(argrelextrema(data, np.greater)[0]))
    mins = set(list(argrelextrema(data, np.less)[0]))

    return np.array(sorted(list(maxs.union(set(maxima)))), dtype="int32"), \
        np.array(sorted(list(mins.union(set(minima)))), dtype="int32")


def get_surface_wave_arrivals(dist_in_deg, min_vel, max_vel, ncircles=1):
    """
    Calculate the arrival time of surface waves, based on the distance
    and velocity range (min_vel, max_vel).
    This function will calculate both minor-arc and major-arc surface
    waves. It further calcualte the surface orbit multiple times
    if you set the ncircles > 1.

    Returns the list of surface wave arrivals in time order.
    """
    if min_vel > max_vel:
        min_vel, max_vel = max_vel, min_vel

    earth_circle = degrees2kilometers(360.0)
    dt1 = earth_circle / max_vel
    dt2 = earth_circle / min_vel

    # 1st arrival: minor-arc arrival
    minor_dist_km = degrees2kilometers(dist_in_deg)  # major-arc distance
    t_minor = [minor_dist_km / max_vel, minor_dist_km/min_vel]

    # 2nd arrival: major-arc arrival
    major_dist_km = degrees2kilometers(360.0 - dist_in_deg)
    t_major = [major_dist_km / max_vel, major_dist_km / min_vel]

    # prepare the arrival list
    arrivals = []
    for i in range(ncircles):
        ts = [t_minor[0] + i * dt1, t_minor[1] + i * dt2]
        arrivals.append(ts)

        ts = [t_major[0] + i * dt1, t_major[1] + i * dt2]
        arrivals.append(ts)

    return arrivals
