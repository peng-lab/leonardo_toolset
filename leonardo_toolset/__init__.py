# -*- coding: utf-8 -*-

"""Top-level package for leonardo_toolset."""

__author__ = "Yu Liu"
__email__ = "liuyu9671@gmail.com"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "1.0.0"

from leonardo_toolset.destripe import DeStripe  # noqa: F401
from leonardo_toolset.fusion import FUSE_det  # noqa: F401
from leonardo_toolset.fusion import FUSE_illu  # noqa: F401


def get_module_version():
    return __version__


# from .workflows import workflow_wrapper  # noqa: F401
# from lsfm_destripe import DeStripe  # noqa: F401
# from lsfm_fuse import FUSE_illu  # noqa: F401
# from lsfm_fuse import FUSE_det  # noqa: F401
