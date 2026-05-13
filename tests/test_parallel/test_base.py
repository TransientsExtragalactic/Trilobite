"""Tests for :mod:`triceratops.parallel.base` — ``Pool`` ABC and ``SerialPool``."""

import pytest

from triceratops.parallel.base import Pool, SerialPool


# ================================================================== #
# Pool abstract base class                                            #
# ================================================================== #


class TestPoolABC:
    def test_cannot_instantiate_pool_directly(self):
        with pytest.raises(TypeError):
            Pool()

    def test_concrete_subclass_must_implement_size(self):
        class MissingSize(Pool):
            def map(self, func, iterable, callback=None):
                return map(func, iterable)

            def close(self):
                pass

        with pytest.raises(TypeError):
            MissingSize()

    def test_concrete_subclass_must_implement_map(self):
        class MissingMap(Pool):
            @property
            def size(self):
                return 1

            def close(self):
                pass

        with pytest.raises(TypeError):
            MissingMap()

    def test_concrete_subclass_must_implement_close(self):
        class MissingClose(Pool):
            @property
            def size(self):
                return 1

            def map(self, func, iterable, callback=None):
                return map(func, iterable)

        with pytest.raises(TypeError):
            MissingClose()

    def test_terminate_defaults_to_close(self):
        """Pool.terminate() must call close() by default."""
        closed = []

        class TrackingPool(Pool):
            @property
            def size(self):
                return 1

            def map(self, func, iterable, callback=None):
                return map(func, iterable)

            def close(self):
                closed.append(True)

        pool = TrackingPool()
        pool.terminate()
        assert closed == [True]

    def test_exit_on_exception_calls_terminate(self):
        """__exit__ with an active exception must call terminate(), not close()."""
        record = []

        class TrackingPool(Pool):
            @property
            def size(self):
                return 1

            def map(self, func, iterable, callback=None):
                return map(func, iterable)

            def close(self):
                record.append("close")

            def terminate(self):
                record.append("terminate")

        pool = TrackingPool()
        try:
            with pool:
                raise ValueError("boom")
        except ValueError:
            pass

        assert record == ["terminate"]

    def test_exit_on_success_calls_close(self):
        """__exit__ without an exception must call close()."""
        record = []

        class TrackingPool(Pool):
            @property
            def size(self):
                return 1

            def map(self, func, iterable, callback=None):
                return map(func, iterable)

            def close(self):
                record.append("close")

            def terminate(self):
                record.append("terminate")

        with TrackingPool():
            pass

        assert record == ["close"]


# ================================================================== #
# SerialPool                                                          #
# ================================================================== #


class TestSerialPool:
    def test_size_is_one(self):
        assert SerialPool().size == 1

    def test_map_applies_function(self):
        pool = SerialPool()
        result = list(pool.map(lambda x: x * 2, [1, 2, 3]))
        assert result == [2, 4, 6]

    def test_map_empty_iterable(self):
        pool = SerialPool()
        result = list(pool.map(lambda x: x, []))
        assert result == []

    def test_map_callback_fires(self):
        seen = []
        pool = SerialPool()
        list(pool.map(lambda x: x + 1, [10, 20, 30], callback=seen.append))
        assert seen == [11, 21, 31]

    def test_map_no_callback_is_fine(self):
        pool = SerialPool()
        result = list(pool.map(str, [1, 2, 3]))
        assert result == ["1", "2", "3"]

    def test_close_is_noop(self):
        pool = SerialPool()
        pool.close()  # must not raise

    def test_terminate_is_noop(self):
        pool = SerialPool()
        pool.terminate()  # must not raise

    def test_context_manager_enters_and_exits(self):
        with SerialPool() as pool:
            result = list(pool.map(lambda x: x**2, [2, 3]))
        assert result == [4, 9]

    def test_context_manager_closes_on_exception(self):
        """Verify the pool exits gracefully even if body raises."""
        pool = SerialPool()
        try:
            with pool:
                raise RuntimeError("test error")
        except RuntimeError:
            pass
        # Pool should still be usable (SerialPool has no real state to break)
        result = list(pool.map(lambda x: x, [1]))
        assert result == [1]

    def test_batched_map_basic(self):
        pool = SerialPool()
        # func receives a list (a batch); return sum of batch
        result = list(pool.batched_map(sum, [1, 2, 3, 4]))
        # size=1, so one batch containing all items
        assert result == [10]

    def test_isinstance_pool(self):
        assert isinstance(SerialPool(), Pool)
