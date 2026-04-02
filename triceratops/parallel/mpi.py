"""
MPI processing pools for Triceratops parallelism.

This module contains :class:`LikelihoodMPIPool`, a :class:`~triceratops.parallel.base.Pool`
subclass for distributing likelihood evaluations across MPI ranks using the ``mpi4py`` library.

.. note::

    ``mpi4py`` is an optional dependency. Importing this module is always safe; however,
    instantiating :class:`LikelihoodMPIPool` will raise :exc:`RuntimeError` if ``mpi4py`` is
    not installed. Install it with ``pip install triceratops[mpi]``.

Design Overview
---------------
The pool follows a classic master-worker pattern:

- **Rank 0** is the *master*. It runs ``emcee`` (or any caller that invokes
  ``pool.map(func, iterable)``). It never enters :meth:`~LikelihoodMPIPool.wait`.
- **Ranks 1 .. N-1** are *workers*. After constructing the pool they must call
  :meth:`~LikelihoodMPIPool.wait`, which blocks until the master sends a stop sentinel
  via :meth:`~LikelihoodMPIPool.close`.

Startup Sequence
~~~~~~~~~~~~~~~~
1. All ranks construct :class:`LikelihoodMPIPool`.
2. The constructor broadcasts the serialized ``InferenceProblem`` JSON from rank 0 so
   every rank reconstructs it exactly once.
3. Worker ranks call ``pool.wait()``; the master proceeds to run emcee.
4. ``emcee`` calls ``pool.map(func, thetas)`` repeatedly.
5. The master dispatches tasks one-by-one to available workers using dynamic dispatch
   and collects results in original task order.
6. When ``emcee`` finishes, the master calls ``pool.close()``, which sends a stop sentinel
   to every worker. Workers unblock from :meth:`~LikelihoodMPIPool.wait` and continue.

MPI Tag Protocol
~~~~~~~~~~~~~~~~
- ``_TAG_TASK = 1``   — master → worker: ``(task_index, theta)``
- ``_TAG_RESULT = 2`` — worker → master: ``(task_index, result)``
- ``_TAG_STOP = 3``   — master → worker: ``None`` (sentinel; worker exits ``wait()``)
"""

from collections.abc import Callable, Iterable
from typing import Any, Optional, Union

from .base import Pool

# ---------------------------------------------------------------------------
# Optional mpi4py import
# ---------------------------------------------------------------------------
try:
    from mpi4py import MPI as _MPI

    _MPI_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _MPI_AVAILABLE = False
    _MPI = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# MPI tag constants
# ---------------------------------------------------------------------------
_TAG_TASK: int = 1
_TAG_RESULT: int = 2
_TAG_STOP: int = 3

# ---------------------------------------------------------------------------
# Worker-global problem store
# ---------------------------------------------------------------------------
_LIKELIHOOD_MPI_POOL_WORKER_PROBLEM = None
"""
The ``InferenceProblem`` instance reconstructed in every MPI rank.

Set once during :meth:`LikelihoodMPIPool.__init__` after the ``comm.bcast``.
Only lightweight theta vectors travel over MPI at evaluation time.
"""


class LikelihoodMPIPool(Pool[Any, float]):
    """
    MPI pool specialized for evaluating a single ``InferenceProblem`` in parallel.

    This class implements the minimal interface required by ``emcee``
    (``pool.map(func, iterable) -> list``) using MPI point-to-point
    communication for dynamic task dispatch.

    Important
    ---------
    The ``func`` argument to :meth:`map` is intentionally ignored.
    Likelihood evaluation is always performed via the ``InferenceProblem`` stored
    inside each rank after :meth:`__init__` broadcasts it.

    Parameters
    ----------
    problem : InferenceProblem
        The inference problem to evaluate in parallel. All ranks must pass
        a valid problem object; rank 0 serializes it and broadcasts the JSON
        to other ranks, which reconstruct it independently.
    comm : mpi4py.MPI.Comm, optional
        MPI communicator. Defaults to ``MPI.COMM_WORLD``.

    Raises
    ------
    RuntimeError
        If ``mpi4py`` is not installed, or if the communicator has fewer than
        2 ranks (at least one master and one worker are required).

    Examples
    --------
    Typical MPI script (run with ``mpirun -n N python script.py``)::

        from triceratops.parallel import LikelihoodMPIPool

        with LikelihoodMPIPool(
            problem=my_problem
        ) as pool:
            if pool.is_master():
                result = sampler.run(
                    n_steps=1000, pool=pool
                )
            else:
                pool.wait()
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        problem: Any,
        comm: Optional[Any] = None,
    ) -> None:
        """
        Construct the MPI likelihood pool and broadcast the problem to all ranks.

        All MPI ranks must call this constructor collectively; it performs a
        ``comm.bcast`` internally.

        Parameters
        ----------
        problem : InferenceProblem
            Inference problem instance. Serialized on rank 0 and broadcast to
            all other ranks, which reconstruct it independently.
        comm : mpi4py.MPI.Comm, optional
            MPI communicator. Defaults to ``MPI.COMM_WORLD``.
        """
        if not _MPI_AVAILABLE:
            raise RuntimeError(
                "mpi4py is required to use LikelihoodMPIPool but is not installed. "
                "Install it with: pip install triceratops[mpi]"
            )

        super().__init__()

        self._comm = comm if comm is not None else _MPI.COMM_WORLD
        self._rank: int = self._comm.Get_rank()
        self._comm_size: int = self._comm.Get_size()

        if self._comm_size < 2:
            raise RuntimeError(
                f"LikelihoodMPIPool requires at least 2 MPI ranks (1 master + 1 worker), "
                f"but the communicator has only {self._comm_size} rank(s). "
                f"Launch your script with: mpirun -n 2 python script.py"
            )

        # Serialize the problem on rank 0 and broadcast the JSON to all ranks.
        if self._rank == 0:
            problem_json: str = problem.to_json()
        else:
            problem_json = None  # type: ignore[assignment]

        problem_json = self._comm.bcast(problem_json, root=0)

        # Reconstruct the problem in every rank and cache in the module-level
        # global so the worker receive loop can access it without referencing self.
        global _LIKELIHOOD_MPI_POOL_WORKER_PROBLEM
        from triceratops.inference.problem import InferenceProblem

        _LIKELIHOOD_MPI_POOL_WORKER_PROBLEM = InferenceProblem.from_json(problem_json)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """
        Number of worker ranks (communicator size minus the master rank).

        Returns
        -------
        int
            ``comm.size - 1``. Always >= 1 after successful construction.
        """
        return self._comm_size - 1

    # ------------------------------------------------------------------
    # Master / worker helpers
    # ------------------------------------------------------------------

    def is_master(self) -> bool:
        """
        Return ``True`` if this process is the master rank (rank 0).

        Returns
        -------
        bool
        """
        return self._rank == 0

    def wait(self) -> None:
        """
        Block worker ranks in a receive loop until a stop sentinel is received.

        Call this on all non-master ranks immediately after constructing the pool.
        The worker receives ``(task_index, theta)`` tuples tagged :data:`_TAG_TASK`,
        evaluates the log posterior, and replies with ``(task_index, result)`` tagged
        :data:`_TAG_RESULT`. On receiving a :data:`_TAG_STOP` message the loop exits.

        Notes
        -----
        - No-op on rank 0 (returns immediately without blocking).
        - Workers do not need to call :meth:`close` themselves.
        """
        if self.is_master():
            return

        status = _MPI.Status()

        while True:
            data = self._comm.recv(source=0, tag=_MPI.ANY_TAG, status=status)
            tag: int = status.Get_tag()

            if tag == _TAG_STOP:
                return

            # tag == _TAG_TASK: data is (task_index, theta)
            task_index, theta = data
            result: float = _LIKELIHOOD_MPI_POOL_WORKER_PROBLEM._log_free_posterior(theta)
            self._comm.send((task_index, result), dest=0, tag=_TAG_RESULT)

    # ------------------------------------------------------------------
    # Core map interface
    # ------------------------------------------------------------------

    def map(
        self,
        func: Any,
        iterable: Iterable[Any],
        callback: Union[Callable[[float], Any], None] = None,
    ) -> list[float]:
        """
        Distribute tasks to worker ranks and collect results in original order.

        Uses dynamic task dispatch: each worker is seeded with one task, and as
        results arrive the newly-free worker is sent the next pending task. This
        keeps all workers busy and handles uneven workloads gracefully.

        Parameters
        ----------
        func : callable
            Ignored. Present for interface compatibility with ``emcee``.
        iterable : Iterable
            Parameter vectors (``theta``) to evaluate.
        callback : callable, optional
            Called with each result as it arrives from a worker. Results are
            fired in arrival order, but the final list is always in task order.

        Returns
        -------
        list of float
            Log-posterior values, one per element of ``iterable``, in the same
            order as the input.

        Raises
        ------
        RuntimeError
            If called on a worker rank. Worker ranks must be blocked in
            :meth:`wait` when the master calls this method.
        """
        if not self.is_master():
            raise RuntimeError(
                "LikelihoodMPIPool.map() must only be called on rank 0 (the master). "
                "Worker ranks should be blocked in pool.wait()."
            )

        tasks: list[Any] = list(iterable)
        n_tasks: int = len(tasks)

        if n_tasks == 0:
            return []

        # results_map stores task_index -> result as replies arrive.
        results_map: dict[int, float] = {}

        # Index of the next task not yet dispatched.
        next_task_index: int = 0

        # --- Seed phase: send one task to each worker --------------------
        for worker_rank in range(1, self._comm_size):
            if next_task_index >= n_tasks:
                break
            self._comm.send(
                (next_task_index, tasks[next_task_index]),
                dest=worker_rank,
                tag=_TAG_TASK,
            )
            next_task_index += 1

        # --- Collection phase: receive results, dispatch remaining tasks --
        status = _MPI.Status()

        while len(results_map) < n_tasks:
            task_index, result = self._comm.recv(
                source=_MPI.ANY_SOURCE,
                tag=_TAG_RESULT,
                status=status,
            )
            worker_rank = status.Get_source()
            results_map[task_index] = result

            if callback is not None:
                callback(result)

            # Dispatch the next pending task to this now-free worker.
            if next_task_index < n_tasks:
                self._comm.send(
                    (next_task_index, tasks[next_task_index]),
                    dest=worker_rank,
                    tag=_TAG_TASK,
                )
                next_task_index += 1

        # Reconstruct in original task order.
        return [results_map[i] for i in range(n_tasks)]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Send a stop sentinel to all worker ranks and finalize the pool.

        Call this on the master after ``emcee`` (or any caller) finishes.
        Workers unblock from :meth:`wait` upon receiving the sentinel.

        Notes
        -----
        - No-op on worker ranks.
        - Do not call :meth:`map` after calling :meth:`close`.
        """
        if not self.is_master():
            return

        for worker_rank in range(1, self._comm_size):
            self._comm.send(None, dest=worker_rank, tag=_TAG_STOP)

    def terminate(self) -> None:
        """
        Abort the MPI communicator immediately.

        Called automatically by the context manager when an exception occurs.
        This is a hard kill of all MPI ranks via ``comm.Abort()``.
        """
        self._comm.Abort()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "LikelihoodMPIPool":
        """Return self to allow use as a context manager."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """
        Clean up the pool on context manager exit.

        On a clean exit, sends the stop sentinel to workers via :meth:`close`.
        On exception, calls :meth:`terminate` (``comm.Abort()``) for an immediate
        hard kill of all ranks.
        """
        if exc_type is not None:
            self.terminate()
        else:
            self.close()
