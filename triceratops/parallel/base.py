"""
Base classes and utilities for multiprocessing pools.

This module defines the abstract base class :class:`Pool` and a concrete implementation
:class:`SerialPool`. The :class:`Pool` class provides a minimal interface for parallel execution, while
:class:`SerialPool` executes tasks sequentially in the current process. These classes are designed to be
backend-agnostic and can be extended to support various parallel execution frameworks (e.g., multiprocessing
or MPI).
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sized
from typing import Any, Generic, TypeVar, Union

from .utils import _callback_wrapper, get_batch_slices

# Type variables for pool input and output types
T = TypeVar("T")  # Input type for pool tasks
R = TypeVar("R")  # Output type for pool results


class Pool(ABC, Generic[T, R]):
    """
    Abstract base class for parallel execution pools.

    Subclasses must implement:
        - size (property)
        - map()
        - close()

    The interface is intentionally minimal and backend-agnostic.
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(self, **_): ...

    # ------------------------------------------------------------------
    # Dunder Methods
    # ------------------------------------------------------------------
    def __enter__(self) -> "Pool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def size(self) -> int:
        """
        Number of worker execution units.

        For:
            - SerialPool → 1
            - MPPool → number of processes
            - MPIPool → communicator size
        """
        ...

    @abstractmethod
    def map(
        self,
        func: Callable[[T], R],
        iterable: Iterable[T],
        callback: Union[Callable[[R], Any], None] = None,
    ) -> Iterable[R]:
        """
        Apply `func` to each element of `iterable`.

        Must return an iterable (preferably lazy).
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up pool resources."""
        ...

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------
    def batched_map(
        self,
        func: Callable[[list[T]], R],
        tasks: Union[Sized, Iterable[T]],
        callback: Union[Callable[[R], Any], None] = None,
    ) -> Iterable[R]:
        """
        Split tasks into `self.size` contiguous batches and map over batches.

        The worker receives a list[T] (a batch).
        """
        if self.size <= 0:
            raise ValueError("Pool size must be >= 1.")

        task_list = list(tasks)
        slices = get_batch_slices(batch_count=self.size, data=task_list)
        batches = [task_list[s] for s in slices]

        results = self.map(func, batches, callback=callback)
        return results

    def _call_callback(
        self,
        callback: Union[Callable[[R], Any], None],
        results: Iterable[R],
    ) -> Iterable[R]:
        """
        Apply a callback to each element of a result stream, if provided.

        If `callback` is not None, returns a lazy iterator that calls
        `callback(result)` for each result before yielding it.
        Otherwise, returns the original iterable unchanged.

        Parameters
        ----------
        callback : Callable[[R], Any] or None
            Function executed on each result for side effects.
        results : Iterable[R]
            Stream of results produced by a pool.

        Returns
        -------
        Iterable[R]
            An iterable yielding the original results.
        """
        return results if callback is None else _callback_wrapper(callback, results)


class SerialPool(Pool[T, R]):
    """
    Serial execution pool.

    This pool executes tasks in the current process using the
    built-in `map()` function. It is primarily useful for:

        - Debugging
        - Testing
        - Environments where multiprocessing is undesirable
        - Fallback when parallel backends are unavailable

    Notes
    -----
    - `size` is always 1.
    - No subprocesses are created.
    - Exceptions propagate immediately.
    - Callback (if provided) is executed inline.
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """
        Initialize the serial pool.

        No resources are allocated.
        """
        super().__init__()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """
        Number of worker execution units.

        Always returns 1 for SerialPool.
        """
        return 1

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def map(
        self,
        func: Callable[[T], R],
        iterable: Iterable[T],
        callback: Union[Callable[[R], Any], None] = None,
    ) -> Iterable[R]:
        """
        Apply `func` to each element of `iterable` sequentially.

        Parameters
        ----------
        func
            Function applied to each input element.
        iterable
            Tasks to process.
        callback
            Optional callback executed after each result is computed.

        Returns
        -------
        Iterable[R]
            Lazy iterator over results.
        """
        results = map(func, iterable)
        return self._call_callback(callback, results)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Close the pool.

        No action required for serial execution.
        """
        pass
