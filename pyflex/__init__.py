#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import collections
import logging


class PyflexError(Exception):
    """
    Base class for all Pyflex exceptions. Will probably be used for all
    exceptions to not overcomplicate things as the whole package is pretty
    small.
    """
    pass


class PyflexWarning(UserWarning):
    """
    Base class for all Pyflex warnings.
    """
    pass


Event = collections.namedtuple("Event", ["latitude", "longitude",
                                         "depth_in_m", "origin_time"])

Station = collections.namedtuple("Station", ["latitude", "longitude"])


__version__ = "0.2.1"


# Setup the logger.
logger = logging.getLogger("pyflex")
#logger.setLevel(logging.WARNING)
logger.setLevel(logging.DEBUG)
# Prevent propagating to higher loggers.
logger.propagate = 0
# Console log handler.
ch = logging.StreamHandler()
# Add formatter
FORMAT = "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s"
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
logger.addHandler(ch)


from pyflex.config import Config  # NOQA
from pyflex.flexwin import select_windows  # NOQA
from pyflex.window_selector import WindowSelector  # NOQA
