"""
Multiprocessing toolkit for Triceratops.

This module provides various structures and tools for parallelism throughout the Triceratops codebase. This
includes both MPI parallelism (via the ``mpi4py`` library) and multiprocessing
(via the standard library's ``multiprocessing`` module).

.. note::

    Documentation is currently sparse for this module, but it is expected to be expanded in
    the future as more multiprocessing tools are added to the codebase. For now, users interested
    in using multiprocessing features should refer to the source code and examples
    provided in the Triceratops repository.
"""

__all__ = ["utils", "mpi", "mp", "base", "LikelihoodMPPool", "SerialPool"]

from . import base, mp, mpi, utils
from .base import SerialPool
from .mp import LikelihoodMPPool
