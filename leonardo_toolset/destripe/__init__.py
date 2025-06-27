__author__ = "Yu Liu"
__email__ = "liuyu9671@gmail.com"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.2.0"

from .core import DeStripe  # noqa: F401


def get_module_version():
    return __version__
