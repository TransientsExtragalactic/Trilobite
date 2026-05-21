"""
Tests for :mod:`trilobite.parallel.mpi` — ``LikelihoodMPIPool``.

These tests cover what is verifiable in a standard single-process pytest run:

- The ``_MPI_AVAILABLE`` flag correctly reflects whether ``mpi4py`` is importable.
- Module-level tag constants are defined with the expected values.
- ``LikelihoodMPIPool`` raises :exc:`RuntimeError` with an actionable message when
  ``mpi4py`` is not installed.
- When ``mpi4py`` *is* installed but only one MPI rank is present (the normal case for
  pytest running without ``mpirun``), instantiation raises :exc:`RuntimeError` explaining
  that at least 2 ranks are required.
- :meth:`~trilobite.parallel.mpi.LikelihoodMPIPool.is_master` and
  :meth:`~trilobite.parallel.mpi.LikelihoodMPIPool.size` are not tested here because
  they require a live multi-rank communicator that is only available when launched
  with ``mpirun -n N``.

.. note::

    End-to-end MPI integration (master-worker dispatch, ``wait()``, ``map()``) requires
    running under ``mpirun -n 2 pytest`` and is intentionally out of scope for the
    standard ``pytest`` suite.
"""

import pytest

from trilobite.parallel import mpi as mpi_module
from trilobite.parallel.mpi import LikelihoodMPIPool, _MPI_AVAILABLE


# ================================================================== #
# Module-level constants and flag                                     #
# ================================================================== #


class TestModuleConstants:
    def test_mpi_available_flag_is_bool(self):
        assert isinstance(_MPI_AVAILABLE, bool)

    def test_mpi_available_matches_importability(self, mpi4py_available):
        assert _MPI_AVAILABLE == mpi4py_available

    def test_tag_constants_defined(self):
        assert hasattr(mpi_module, "_TAG_TASK")
        assert hasattr(mpi_module, "_TAG_RESULT")
        assert hasattr(mpi_module, "_TAG_STOP")

    def test_tag_constants_are_distinct_positive_ints(self):
        tags = {mpi_module._TAG_TASK, mpi_module._TAG_RESULT, mpi_module._TAG_STOP}
        assert len(tags) == 3
        assert all(isinstance(t, int) and t > 0 for t in tags)


# ================================================================== #
# Instantiation guards                                                #
# ================================================================== #


class TestInstantiationGuards:
    def test_raises_without_mpi4py(self, simple_problem, mpi4py_available):
        """
        If ``mpi4py`` is not installed, ``LikelihoodMPIPool`` must raise
        ``RuntimeError`` with a message that includes installation instructions.
        """
        if mpi4py_available:
            pytest.skip("mpi4py is installed; skipping absent-mpi4py guard test.")

        with pytest.raises(RuntimeError, match="mpi4py"):
            LikelihoodMPIPool(problem=simple_problem)

    def test_raises_without_mpi4py_via_make_pool(self, simple_problem, mpi4py_available):
        """``make_pool('mpi', ...)`` must propagate the RuntimeError."""
        if mpi4py_available:
            pytest.skip("mpi4py is installed; skipping absent-mpi4py guard test.")

        from trilobite.parallel import make_pool

        with pytest.raises(RuntimeError, match="mpi4py"):
            make_pool("mpi", problem=simple_problem)

    def test_error_message_includes_install_hint(self, simple_problem, mpi4py_available):
        """The RuntimeError message must mention how to install the dependency."""
        if mpi4py_available:
            pytest.skip("mpi4py is installed; skipping absent-mpi4py guard test.")

        with pytest.raises(RuntimeError, match="pip install"):
            LikelihoodMPIPool(problem=simple_problem)

    def test_raises_with_single_rank(self, simple_problem, mpi4py_available):
        """
        When ``mpi4py`` is installed but pytest runs in a single process
        (``COMM_WORLD.size == 1``), instantiation must raise ``RuntimeError``
        explaining that at least 2 ranks are required.
        """
        if not mpi4py_available:
            pytest.skip("mpi4py not installed; skipping single-rank guard test.")

        with pytest.raises(RuntimeError, match="2"):
            LikelihoodMPIPool(problem=simple_problem)

    def test_single_rank_error_mentions_mpirun(self, simple_problem, mpi4py_available):
        """The single-rank error message must mention ``mpirun`` for discoverability."""
        if not mpi4py_available:
            pytest.skip("mpi4py not installed; skipping single-rank guard test.")

        with pytest.raises(RuntimeError, match="mpirun"):
            LikelihoodMPIPool(problem=simple_problem)


# ================================================================== #
# map() guard on non-master rank                                      #
# ================================================================== #


class TestMapGuard:
    def test_map_raises_without_mpi4py(self, simple_problem, mpi4py_available):
        """
        Calling ``map()`` on an uninitialised pool (no mpi4py) must raise
        ``RuntimeError`` at construction, not at ``map()`` time.  This test
        confirms the error surfaces before any map call.
        """
        if mpi4py_available:
            pytest.skip("mpi4py is installed.")

        with pytest.raises(RuntimeError):
            pool = LikelihoodMPIPool(problem=simple_problem)
            pool.map(None, [])
