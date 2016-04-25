# -*- coding: utf-8 -*-
# This file is part of quark-sphinx-theme.
# Copyright (c) 2016 Felix Krull <f_krull@gmx.de>
# Released under the terms of the BSD license; see LICENSE.

import os

from .__version__ import __version_info__, __version__
_package_dir = os.path.dirname(os.path.abspath(__file__))


def get_path():
    """Get theme path."""
    return _package_dir


__all__ = ["__version__", "__version_info__", "get_path"]
