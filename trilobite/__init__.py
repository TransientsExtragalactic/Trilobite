"""Radio analysis module for supernova shocks, and other synchrotron emission sources."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["data", "inference", "models", "radiation", "utils", "parallel"]

from . import data, inference, models, parallel, radiation, utils

try:
    __version__ = version("trilobite")
except PackageNotFoundError:
    # Package is not installed (e.g., running from source tree)
    __version__ = "0.0.0"
