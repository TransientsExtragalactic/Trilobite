"""Multiprocessing utility functions for batching and callbacks."""

from collections.abc import Callable, Iterable, Iterator, Sized
from typing import Any, TypeVar

import numpy as np

# Type variables for input and output types
T = TypeVar("T")  # Input type for pool tasks
R = TypeVar("R")  # Output type for pool results


def _callback_wrapper(
    callback: Callable[[R], Any],
    generator: Iterable[R],
) -> Iterator[R]:
    """
    Wrap an iterable of results so that a callback is executed for each element as it is yielded.

    This function preserves lazy evaluation: elements are not
    computed or consumed until the returned iterator is iterated over.

    Parameters
    ----------
    callback : Callable[[R], Any]
        A function that is called with each element produced by
        `generator`. The return value of the callback is ignored.

        The callback is typically used for side effects such as:
        - Logging
        - Progress tracking
        - Saving results to disk
        - Aggregating statistics

    generator : Iterable[R]
        An iterable (usually a generator) producing results of type `R`.

    Returns
    -------
    Iterator[R]
        A generator that yields the same elements as `generator`,
        but ensures that `callback(element)` is executed before
        each element is yielded.

    Notes
    -----
    - The callback does NOT modify the yielded values.
    - The callback is executed synchronously during iteration.
    - If iteration stops early, the callback will only have been
      applied to the consumed elements.
    """
    for element in generator:
        callback(element)
        yield element


def get_batch_slices(
    batch_count: int,
    task_count: int | None = None,
    data: Sized | None = None,
) -> list[slice]:
    """
    Compute contiguous slices that divide tasks into batches.

    Exactly one of `task_count` or `data` must be provided.

    Parameters
    ----------
    batch_count
        Number of batches to divide the tasks into.
    task_count
        Total number of tasks. Mutually exclusive with `data`.
    data
        Sized iterable (e.g. list, array). If provided, its length is used as
        `task_count`.

    Returns
    -------
    list of slice
        Contiguous slices dividing the tasks into nearly equal-sized batches.

    Raises
    ------
    ValueError
        If inputs are invalid.

    Notes
    -----
    - If `batch_count > task_count`, the number of batches is reduced to `task_count`.
    - Tasks are distributed as evenly as possible; earlier batches receive one extra
      task if necessary.
    """
    # --- Validate inputs ----------------------- #
    if (task_count is None) == (data is None):
        raise ValueError("Exactly one of 'task_count' or 'data' must be provided.")

    if data is not None:
        try:
            task_count = len(data)
        except TypeError as e:
            raise ValueError("data must be a sized iterable.") from e

    if task_count is None or task_count <= 0:
        raise ValueError("task_count must be a positive integer.")

    if batch_count <= 0:
        raise ValueError("batch_count must be a positive integer.")

    # Do not create more batches than tasks
    batch_count = min(batch_count, task_count)

    # --- Compute evenly distributed slices ----- #
    base, remainder = divmod(task_count, batch_count)

    slices: list[slice] = []
    start = 0

    for i in range(batch_count):
        size = base + (1 if i < remainder else 0)
        stop = start + size
        slices.append(slice(start, stop))
        start = stop

    return slices


def split_args_into_batches(args, batch_slices):
    """
    Split positional arguments into per-batch argument tuples.

    For each slice in `batch_slices`, this constructs a tuple of arguments where:
      - list/ndarray arguments are sliced by that batch slice
      - scalar / non-sliceable arguments are passed through unchanged

    Parameters
    ----------
    args
        Positional arguments to split.
    batch_slices
        Batch slices, typically from `get_batch_slices`.

    Returns
    -------
    list of tuple
        One tuple per batch, suitable for mapping over batches.
    """
    batched_args = []
    for batch_slice in batch_slices:
        batch_args = []
        for arg in args:
            if isinstance(arg, (list, np.ndarray)):
                batch_args.append(arg[batch_slice])
            else:
                batch_args.append(arg)
        batched_args.append(tuple(batch_args))
    return batched_args


def split_kwargs_into_batches(kwargs, batch_slices):
    """
    Split keyword arguments into per-batch kwargs dictionaries.

    For each slice in `batch_slices`, this constructs a kwargs dict where:
      - list/ndarray values are sliced by that batch slice
      - scalar / non-sliceable values are passed through unchanged

    Parameters
    ----------
    kwargs
        Keyword arguments to split.
    batch_slices
        Batch slices, typically from `get_batch_slices`.

    Returns
    -------
    list of dict
        One kwargs dict per batch, suitable for mapping over batches.
    """
    batched_kwargs = []
    for batch_slice in batch_slices:
        batch_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (list, np.ndarray)):
                batch_kwargs[key] = value[batch_slice]
            else:
                batch_kwargs[key] = value
        batched_kwargs.append(batch_kwargs)
    return batched_kwargs
