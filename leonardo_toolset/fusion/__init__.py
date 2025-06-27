# -*- coding: utf-8 -*-

"""Top-level package for LSFM FUSE."""

__author__ = "Yu Liu"
__email__ = "liuyu9671@gmail.com"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.0.2"

from .fuse_det import FUSE_det  # noqa: F401
from .fuse_illu import FUSE_illu  # noqa: F401


def get_module_version():
    return __version__
