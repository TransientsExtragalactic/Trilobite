"""
Multiprocessing backend for likelihood evaluation.

This module provides a specialized multiprocessing pool designed
specifically for evaluating an `InferenceProblem` in parallel
(e.g. for use with `emcee`).

Design Overview
---------------
`emcee` expects a pool object that implements:

    pool.map(func, iterable) -> sequence

However, directly passing a bound method such as
`InferenceProblem._log_free_posterior` can cause pickling issues
under the `spawn` start method (default on macOS and Windows).

To avoid repeated pickling of a complex problem object:

1. The problem is serialized to JSON in the parent process.
2. Each worker reconstructs it exactly once via an initializer.
3. The reconstructed problem is stored in a worker-global.
4. Only lightweight parameter vectors (`theta`) are sent per task.

This dramatically reduces serialization overhead and avoids
common multiprocessing pitfalls.

Interrupt Handling
------------------
Workers ignore SIGINT so that Ctrl+C is handled cleanly by the
parent process. The parent polls with a timeout so KeyboardInterrupt
is not swallowed by blocking waits.

This backend is designed for:
    - MCMC likelihood evaluation
    - Heavy compute per likelihood call
    - Minimal IPC overhead
"""

import multiprocessing as mp
import signal
from collections.abc import Callable, Iterable
from typing import Any, Optional, Union

from .base import Pool

# =====================================================================
# Worker Global States
# =====================================================================
#
# IMPORTANT:
# ----------
# This variable exists *once per worker process*.
#
# It stores the reconstructed InferenceProblem so that:
#   - It is initialized exactly once per process.
#   - It is NOT pickled and sent with every task.
#   - Only parameter vectors are transmitted between processes.
#
# This is the core mechanism that makes this pool efficient.
#
# =====================================================================
_LIKELIHOOD_MP_POOL_WORKER_PROBLEM = None
""" The serialized InferenceProblem reconstructed in each worker process. """

# =====================================================================
# Worker Initialization
# =====================================================================


def _worker_initialize(problem_json: str) -> None:
    """
    Per-process initializer.

    This function is executed exactly once in each worker process.

    Responsibilities
    ----------------
    1. Ignore SIGINT so Ctrl+C is handled by the parent.
    2. Deserialize the InferenceProblem.
    3. Store it in the worker-global namespace.

    Parameters
    ----------
    problem_json : str
        JSON-serialized representation of the InferenceProblem.
    """
    global _LIKELIHOOD_MP_POOL_WORKER_PROBLEM

    from triceratops.inference.problem import InferenceProblem

    # Workers ignore Ctrl+C — parent handles termination.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    _LIKELIHOOD_MP_POOL_WORKER_PROBLEM = InferenceProblem.from_json(problem_json)


def _worker_log_prob(theta: Any) -> float:
    """
    Evaluate the log posterior for a single parameter vector.

    This function is executed inside worker processes.

    Parameters
    ----------
    theta
        Parameter vector.

    Returns
    -------
    float
        Log posterior evaluated at `theta`.

    Notes
    -----
    Assumes `_LIKELIHOOD_MP_POOL_WORKER_PROBLEM` has already been initialized.
    """
    return _LIKELIHOOD_MP_POOL_WORKER_PROBLEM._log_free_posterior(theta)


# =====================================================================
# LikelihoodMPPool
# =====================================================================
class LikelihoodMPPool(Pool[Any, float]):
    """
    Multiprocessing pool specialized for evaluating a single InferenceProblem.

    This class implements the minimal interface required by `emcee`:
        - map(func, iterable)
        - close()
        - context manager support

    Important
    ---------
    The `func` argument to `map()` is intentionally ignored.
    We always evaluate `_worker_log_prob`, since the problem
    is stored inside each worker process.

    Parameters
    ----------
    problem : InferenceProblem
        Problem to evaluate.
    processes : int, optional
        Number of worker processes. Defaults to `cpu_count()`.
    wait_timeout : int, optional
        Timeout (seconds) used when polling async results.
        This allows KeyboardInterrupt to be caught properly.
    """

    def __init__(
        self,
        problem: Any,
        processes: Optional[int] = None,
        wait_timeout: int = 3600,
    ) -> None:
        """
        Construct the multiprocessing likelihood pool.

        The problem is serialized once in the parent and reconstructed
        once per worker process.
        """
        if processes is None:
            processes = mp.cpu_count()

        if processes < 1:
            raise ValueError("processes must be >= 1")

        self._processes = int(processes)
        self._problem_json = problem.to_json()
        self._wait_timeout = int(wait_timeout)

        # Create underlying multiprocessing pool.
        self._pool = mp.Pool(
            processes=self._processes,
            initializer=_worker_initialize,
            initargs=(self._problem_json,),
        )

    # ------------------------------------------------------------------
    # Public Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of worker processes."""
        return self._processes

    # ------------------------------------------------------------------
    # Required emcee interface
    # ------------------------------------------------------------------

    def map(
        self,
        func: Any,
        iterable: Iterable[Any],
        callback: Union[Callable[[float], Any], None] = None,
    ) -> list[float]:
        """
        Map parameter vectors to log posterior values.

        Parameters
        ----------
        func
            Ignored. Present only for compatibility with `emcee`.
        iterable
            Iterable of parameter vectors (`theta`).
        callback : callable, optional
            Called with each result after collection. Fired once per element
            in iteration order.

        Returns
        -------
        list of float
            Log posterior values corresponding to each element.
        """
        async_result = self._pool.map_async(_worker_log_prob, iterable)

        # Poll with timeout to avoid swallowing Ctrl+C
        while True:
            try:
                results: list[float] = async_result.get(self._wait_timeout)
                break

            except mp.TimeoutError:
                continue

            except KeyboardInterrupt:
                # Ensure workers die immediately.
                self.terminate()
                self.join()
                raise

        if callback is not None:
            for r in results:
                callback(r)

        return results

    # ------------------------------------------------------------------
    # Lifecycle Management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Close the pool gracefully.

        Waits for all worker processes to exit cleanly.
        """
        self._pool.close()
        self._pool.join()

    def terminate(self) -> None:
        """
        Terminate worker processes immediately.

        Used for interrupt/error handling.
        """
        self._pool.terminate()

    def join(self) -> None:
        """Wait for worker processes to exit."""
        self._pool.join()

    # ------------------------------------------------------------------
    # Context Manager Support
    # ------------------------------------------------------------------

    def __enter__(self) -> "LikelihoodMPPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Ensure pool is cleaned up on exit.

        If exiting due to an exception, terminate immediately.
        Otherwise close gracefully.
        """
        if exc_type is not None:
            self.terminate()
            self.join()
        else:
            self.close()
