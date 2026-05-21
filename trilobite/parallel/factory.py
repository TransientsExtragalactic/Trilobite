"""
Factory utilities for constructing parallel pool instances.

This module provides :func:`make_pool`, a convenience function for creating
pool objects without importing backend-specific modules directly. Backend
modules are imported lazily so that optional dependencies (e.g. ``mpi4py``)
are only required when the corresponding backend is actually requested.
"""

from typing import Any, Optional

from .base import Pool, SerialPool


def make_pool(
    backend: str,
    problem: Optional[Any] = None,
    **kwargs: Any,
) -> Pool:
    """
    Construct a parallel pool for the given backend.

    Parameters
    ----------
    backend : str
        Pool backend to construct. One of ``"serial"``, ``"mp"``, ``"mpi"``.
    problem : InferenceProblem, optional
        Inference problem instance. Required for ``"mp"`` and ``"mpi"`` backends.
    **kwargs
        Additional keyword arguments forwarded to the pool constructor.
        For ``"mp"``: accepts ``processes`` and ``wait_timeout``.
        For ``"mpi"``: accepts ``comm``.

    Returns
    -------
    Pool
        A pool instance ready for use with ``emcee`` or other callers.

    Raises
    ------
    ValueError
        If `backend` is not recognised, or if `problem` is required but not supplied.
    RuntimeError
        If ``"mpi"`` is requested but ``mpi4py`` is not installed.

    Examples
    --------
    >>> pool = make_pool("serial")

    >>> pool = make_pool(
    ...     "mp", problem=my_problem, processes=4
    ... )

    >>> pool = make_pool("mpi", problem=my_problem)
    """
    backend = backend.lower()

    if backend == "serial":
        return SerialPool()

    if backend == "mp":
        if problem is None:
            raise ValueError("'problem' must be provided for the 'mp' backend.")
        from .mp import LikelihoodMPPool

        return LikelihoodMPPool(problem=problem, **kwargs)

    if backend == "mpi":
        if problem is None:
            raise ValueError("'problem' must be provided for the 'mpi' backend.")
        from .mpi import LikelihoodMPIPool

        return LikelihoodMPIPool(problem=problem, **kwargs)

    raise ValueError(f"Unknown pool backend {backend!r}. Choose one of: 'serial', 'mp', 'mpi'.")
