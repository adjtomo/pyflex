#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Weighted interval scheduling implementation.

Adapted from
https://github.com/farazdagi/algorithms/blob/master/
weighted-interval-scheduling.py

:copyright:
    Victor Farazdagi, 2013
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import collections
import bisect


def compute_previous_intervals(I):
    """
    For every interval j, compute the rightmost mutually compatible interval
    i, where i < j I is a sorted list of Interval objects (sorted by finish
    time)
    """
    # extract start and finish times
    start = [i.left for i in I]
    finish = [i.right for i in I]

    p = []
    for j in range(len(I)):
        # rightmost interval f_i <= s_j
        i = bisect.bisect_right(finish, start[j]) - 1
        p.append(i)

    return p


def schedule_weighted_intervals(I):
    """
    Use dynamic algorithm to schedule weighted intervals
       sorting is O(n log n),
       finding p[1..n] is O(n log n),
       finding OPT[1..n] is O(n),
       selecting is O(n)
       whole operation is dominated by O(n log n)
    """
    # f_1 <= f_2 <= .. <= f_n
    I.sort(key=lambda x: x.right)
    p = compute_previous_intervals(I)

    # compute OPTs iteratively in O(n), here we use DP
    OPT = collections.defaultdict(int)
    OPT[-1] = 0
    OPT[0] = 0
    for j in range(1, len(I)):
        OPT[j] = max(I[j].weight + OPT[p[j]], OPT[j - 1])

    # given OPT and p, find actual solution intervals in O(n)
    O = []

    def compute_solution(j):
        if j >= 0:  # will halt on OPT[-1]
            if I[j].weight + OPT[p[j]] > OPT[j - 1]:
                O.append(I[j])
                compute_solution(p[j])
            else:
                compute_solution(j - 1)
    compute_solution(len(I) - 1)

    # resort, as our O is in reverse order (OPTIONAL)
    O.sort(key=lambda x: x.right)

    return O
