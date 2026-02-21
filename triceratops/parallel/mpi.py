"""
MPI processing pools for Triceratops parallelism.

This module contains :class:`multiprocessing.base.Pool` subclasses for distributing compute over
MPI communicators using the ``mpi4py`` library. These pools can be used to parallelize tasks across multiple
nodes in a cluster or across multiple processes on a single machine, depending on the MPI configuration.

.. note::

    This module does not handle low-level MPI enabled processes. It is instead focused on allowing parallelism
    in the inference layer.
"""
