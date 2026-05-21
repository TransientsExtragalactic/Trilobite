"""
Tests for :mod:`trilobite.parallel.mp` — ``LikelihoodMPPool``.

These tests spawn real worker processes.  They exercise the multiprocessing
backend end-to-end against a simple inference problem to confirm that:

- likelihood evaluation is correctly distributed across workers,
- results are returned in the same order as the input,
- callbacks fire for each result,
- the pool shuts down cleanly, and
- ``LikelihoodMPPool`` properly inherits from ``Pool``.
"""

import numpy as np
import pytest

from trilobite.parallel.base import Pool
from trilobite.parallel.mp import LikelihoodMPPool


# ================================================================== #
# Helpers                                                             #
# ================================================================== #


def _make_theta_batch(problem, n: int, seed: int = 0):
    """Return ``n`` perturbed copies of ``problem.initial_theta``."""
    rng = np.random.default_rng(seed)
    theta0 = problem.initial_theta
    noise = rng.normal(scale=0.05, size=(n, len(theta0)))
    return [theta0 + noise[i] for i in range(n)]


# ================================================================== #
# LikelihoodMPPool tests                                              #
# ================================================================== #


class TestLikelihoodMPPool:
    def test_isinstance_pool(self, simple_problem):
        pool = LikelihoodMPPool(problem=simple_problem, processes=1)
        pool.close()
        assert isinstance(pool, Pool)

    def test_size_matches_processes(self, simple_problem):
        pool = LikelihoodMPPool(problem=simple_problem, processes=2)
        pool.close()
        assert pool.size == 2

    def test_size_defaults_to_cpu_count(self, simple_problem):
        import multiprocessing as mp

        pool = LikelihoodMPPool(problem=simple_problem)
        pool.close()
        assert pool.size == mp.cpu_count()

    def test_map_returns_floats(self, simple_problem):
        thetas = _make_theta_batch(simple_problem, n=4)
        with LikelihoodMPPool(problem=simple_problem, processes=2) as pool:
            results = pool.map(None, thetas)
        assert len(results) == 4
        assert all(isinstance(r, float) for r in results)

    def test_map_result_order_matches_input(self, simple_problem):
        """Results must correspond to the same-index input theta."""
        thetas = _make_theta_batch(simple_problem, n=6)

        # Evaluate serially as ground truth
        serial = [simple_problem._log_free_posterior(t) for t in thetas]

        with LikelihoodMPPool(problem=simple_problem, processes=2) as pool:
            parallel = pool.map(None, thetas)

        np.testing.assert_allclose(parallel, serial, rtol=1e-10)

    def test_map_single_task(self, simple_problem):
        thetas = _make_theta_batch(simple_problem, n=1)
        with LikelihoodMPPool(problem=simple_problem, processes=1) as pool:
            results = pool.map(None, thetas)
        assert len(results) == 1
        expected = simple_problem._log_free_posterior(thetas[0])
        np.testing.assert_allclose(results[0], expected, rtol=1e-10)

    def test_map_callback_fires_for_each_result(self, simple_problem):
        thetas = _make_theta_batch(simple_problem, n=5)
        received = []
        with LikelihoodMPPool(problem=simple_problem, processes=2) as pool:
            pool.map(None, thetas, callback=received.append)
        assert len(received) == 5
        assert all(isinstance(r, float) for r in received)

    def test_map_callback_values_match_results(self, simple_problem):
        thetas = _make_theta_batch(simple_problem, n=4)
        received = []
        with LikelihoodMPPool(problem=simple_problem, processes=2) as pool:
            results = pool.map(None, thetas, callback=received.append)
        # Callback values should be the same floats as the returned results
        np.testing.assert_allclose(sorted(received), sorted(results), rtol=1e-10)

    def test_map_func_arg_ignored(self, simple_problem):
        """The ``func`` argument must be accepted but ignored."""
        thetas = _make_theta_batch(simple_problem, n=3)
        sentinel = object()
        with LikelihoodMPPool(problem=simple_problem, processes=1) as pool:
            results = pool.map(sentinel, thetas)
        assert len(results) == 3

    def test_context_manager_cleans_up(self, simple_problem):
        thetas = _make_theta_batch(simple_problem, n=2)
        pool = LikelihoodMPPool(problem=simple_problem, processes=1)
        with pool:
            pool.map(None, thetas)
        # Pool is closed; no assertion needed — just must not hang or raise.

    def test_invalid_processes_raises(self, simple_problem):
        with pytest.raises(ValueError, match="processes must be >= 1"):
            LikelihoodMPPool(problem=simple_problem, processes=0)

    def test_map_against_serial_pool(self, simple_problem):
        """Results must match :class:`~trilobite.parallel.base.SerialPool`."""
        from trilobite.parallel.base import SerialPool

        thetas = _make_theta_batch(simple_problem, n=8, seed=7)
        serial_results = list(SerialPool().map(simple_problem._log_free_posterior, thetas))

        with LikelihoodMPPool(problem=simple_problem, processes=2) as pool:
            mp_results = pool.map(None, thetas)

        np.testing.assert_allclose(mp_results, serial_results, rtol=1e-10)
