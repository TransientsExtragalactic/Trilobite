"""Tests for :mod:`trilobite.parallel.utils`."""

import numpy as np
import pytest

from trilobite.parallel.utils import (
    _callback_wrapper,
    get_batch_slices,
    split_args_into_batches,
    split_kwargs_into_batches,
)


# ================================================================== #
# get_batch_slices                                                    #
# ================================================================== #


class TestGetBatchSlices:
    def test_even_division(self):
        slices = get_batch_slices(batch_count=4, task_count=8)
        assert len(slices) == 4
        sizes = [s.stop - s.start for s in slices]
        assert all(sz == 2 for sz in sizes)

    def test_uneven_division_front_loaded(self):
        # 10 tasks into 3 batches: [4, 3, 3]
        slices = get_batch_slices(batch_count=3, task_count=10)
        assert len(slices) == 3
        sizes = [s.stop - s.start for s in slices]
        assert sizes[0] == 4
        assert sizes[1] == 3
        assert sizes[2] == 3

    def test_contiguous_and_cover_all(self):
        task_count = 17
        slices = get_batch_slices(batch_count=5, task_count=task_count)
        # Ensure slices are contiguous and cover all tasks
        assert slices[0].start == 0
        for i in range(1, len(slices)):
            assert slices[i].start == slices[i - 1].stop
        assert slices[-1].stop == task_count

    def test_more_batches_than_tasks_clamped(self):
        # batch_count > task_count: number of batches is reduced
        slices = get_batch_slices(batch_count=10, task_count=3)
        assert len(slices) == 3
        assert all((s.stop - s.start) == 1 for s in slices)

    def test_single_batch(self):
        slices = get_batch_slices(batch_count=1, task_count=5)
        assert len(slices) == 1
        assert slices[0] == slice(0, 5)

    def test_single_task(self):
        slices = get_batch_slices(batch_count=4, task_count=1)
        assert len(slices) == 1
        assert slices[0] == slice(0, 1)

    def test_data_kwarg(self):
        data = list(range(6))
        slices = get_batch_slices(batch_count=3, data=data)
        assert len(slices) == 3
        assert all((s.stop - s.start) == 2 for s in slices)

    def test_error_both_none(self):
        with pytest.raises(ValueError, match="Exactly one"):
            get_batch_slices(batch_count=2)

    def test_error_both_provided(self):
        with pytest.raises(ValueError, match="Exactly one"):
            get_batch_slices(batch_count=2, task_count=4, data=[1, 2, 3, 4])

    def test_error_zero_tasks(self):
        with pytest.raises(ValueError):
            get_batch_slices(batch_count=2, task_count=0)

    def test_error_zero_batches(self):
        with pytest.raises(ValueError):
            get_batch_slices(batch_count=0, task_count=5)


# ================================================================== #
# split_args_into_batches                                             #
# ================================================================== #


class TestSplitArgsIntoBatches:
    def test_list_argument_is_sliced(self):
        args = ([0, 1, 2, 3],)
        slices = get_batch_slices(batch_count=2, task_count=4)
        batches = split_args_into_batches(args, slices)
        assert batches[0] == ([0, 1],)
        assert batches[1] == ([2, 3],)

    def test_ndarray_argument_is_sliced(self):
        arr = np.arange(6)
        args = (arr,)
        slices = get_batch_slices(batch_count=3, task_count=6)
        batches = split_args_into_batches(args, slices)
        np.testing.assert_array_equal(batches[0][0], np.array([0, 1]))
        np.testing.assert_array_equal(batches[1][0], np.array([2, 3]))
        np.testing.assert_array_equal(batches[2][0], np.array([4, 5]))

    def test_scalar_argument_is_broadcast(self):
        args = (42,)
        slices = get_batch_slices(batch_count=3, task_count=6)
        batches = split_args_into_batches(args, slices)
        assert all(b[0] == 42 for b in batches)

    def test_mixed_arguments(self):
        args = ([10, 20, 30], "constant", np.array([1.0, 2.0, 3.0]))
        slices = get_batch_slices(batch_count=3, task_count=3)
        batches = split_args_into_batches(args, slices)
        assert len(batches) == 3
        for i, batch in enumerate(batches):
            assert batch[0] == [10 + 10 * i]
            assert batch[1] == "constant"
            assert batch[2] == np.array([1.0 + i])


# ================================================================== #
# split_kwargs_into_batches                                           #
# ================================================================== #


class TestSplitKwargsIntoBatches:
    def test_list_value_is_sliced(self):
        kwargs = {"x": [0, 1, 2, 3]}
        slices = get_batch_slices(batch_count=2, task_count=4)
        batches = split_kwargs_into_batches(kwargs, slices)
        assert batches[0] == {"x": [0, 1]}
        assert batches[1] == {"x": [2, 3]}

    def test_scalar_value_is_broadcast(self):
        kwargs = {"n": 99, "data": np.arange(4)}
        slices = get_batch_slices(batch_count=2, task_count=4)
        batches = split_kwargs_into_batches(kwargs, slices)
        assert all(b["n"] == 99 for b in batches)
        np.testing.assert_array_equal(batches[0]["data"], np.array([0, 1]))
        np.testing.assert_array_equal(batches[1]["data"], np.array([2, 3]))


# ================================================================== #
# _callback_wrapper                                                   #
# ================================================================== #


class TestCallbackWrapper:
    def test_values_pass_through(self):
        results = list(_callback_wrapper(lambda x: None, [1, 2, 3]))
        assert results == [1, 2, 3]

    def test_callback_fires_for_each_element(self):
        seen = []
        list(_callback_wrapper(seen.append, ["a", "b", "c"]))
        assert seen == ["a", "b", "c"]

    def test_lazy_evaluation(self):
        """Callback should not fire until the iterator is consumed."""
        fired = []

        def cb(x):
            fired.append(x)

        gen = _callback_wrapper(cb, [10, 20, 30])
        assert fired == []
        next(gen)
        assert fired == [10]

    def test_callback_return_value_ignored(self):
        """Callback return value must not affect yielded elements."""
        results = list(_callback_wrapper(lambda x: x * 100, [1, 2, 3]))
        assert results == [1, 2, 3]
