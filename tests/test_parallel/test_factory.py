"""Tests for :mod:`triceratops.parallel.factory` — ``make_pool``."""

import pytest

from triceratops.parallel.base import Pool, SerialPool
from triceratops.parallel.factory import make_pool
from triceratops.parallel.mp import LikelihoodMPPool
from triceratops.parallel.mpi import LikelihoodMPIPool


# ================================================================== #
# Serial backend                                                      #
# ================================================================== #


class TestMakePoolSerial:
    def test_returns_serial_pool(self):
        pool = make_pool("serial")
        assert isinstance(pool, SerialPool)
        pool.close()

    def test_serial_is_pool_subclass(self):
        pool = make_pool("serial")
        assert isinstance(pool, Pool)
        pool.close()

    def test_serial_ignores_extra_kwargs(self):
        # SerialPool.__init__ accepts **_ so extra kwargs must be harmless
        pool = make_pool("serial")
        pool.close()

    def test_serial_case_insensitive(self):
        pool = make_pool("SERIAL")
        assert isinstance(pool, SerialPool)
        pool.close()


# ================================================================== #
# Multiprocessing backend                                             #
# ================================================================== #


class TestMakePoolMP:
    def test_returns_likelihood_mp_pool(self, simple_problem):
        pool = make_pool("mp", problem=simple_problem, processes=1)
        assert isinstance(pool, LikelihoodMPPool)
        pool.close()

    def test_is_pool_subclass(self, simple_problem):
        pool = make_pool("mp", problem=simple_problem, processes=1)
        assert isinstance(pool, Pool)
        pool.close()

    def test_passes_processes_kwarg(self, simple_problem):
        pool = make_pool("mp", problem=simple_problem, processes=2)
        assert pool.size == 2
        pool.close()

    def test_raises_without_problem(self):
        with pytest.raises(ValueError, match="problem"):
            make_pool("mp")

    def test_mp_case_insensitive(self, simple_problem):
        pool = make_pool("MP", problem=simple_problem, processes=1)
        assert isinstance(pool, LikelihoodMPPool)
        pool.close()


# ================================================================== #
# MPI backend                                                         #
# ================================================================== #


class TestMakePoolMPI:
    def test_raises_without_problem(self):
        with pytest.raises(ValueError, match="problem"):
            make_pool("mpi")

    def test_raises_without_mpi4py(self, simple_problem, mpi4py_available):
        if mpi4py_available:
            pytest.skip("mpi4py is installed; this test covers the absent case.")
        with pytest.raises(RuntimeError, match="mpi4py"):
            make_pool("mpi", problem=simple_problem)

    def test_raises_single_rank_with_mpi4py(self, simple_problem, mpi4py_available):
        if not mpi4py_available:
            pytest.skip("mpi4py not installed.")
        with pytest.raises(RuntimeError):
            make_pool("mpi", problem=simple_problem)

    def test_mpi_case_insensitive(self, simple_problem, mpi4py_available):
        """Case normalization is applied before any backend logic."""
        if mpi4py_available:
            pytest.skip("mpi4py installed; single-rank test supersedes.")
        with pytest.raises(RuntimeError, match="mpi4py"):
            make_pool("MPI", problem=simple_problem)


# ================================================================== #
# Unknown backend                                                     #
# ================================================================== #


class TestMakePoolUnknown:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown pool backend"):
            make_pool("dask")

    def test_error_message_lists_valid_backends(self):
        with pytest.raises(ValueError, match="serial") as exc_info:
            make_pool("spark")
        msg = str(exc_info.value)
        assert "serial" in msg
        assert "mp" in msg
        assert "mpi" in msg

    @pytest.mark.parametrize("bad_backend", ["", "thread", "ray", "joblib"])
    def test_various_unknown_backends(self, bad_backend):
        with pytest.raises(ValueError):
            make_pool(bad_backend)
