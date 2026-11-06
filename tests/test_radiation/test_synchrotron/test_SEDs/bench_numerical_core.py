"""
Performance benchmarks for NumericalSynchrotronEngine.

These benchmarks use pytest-benchmark to measure the wall-clock cost of the
most performance-sensitive paths through the numerical synchrotron engine:

  - Full SED (30 frequencies) with synchrotron self-absorption (SSA), spanning
    both optically thick and thin regimes.
  - Emissivity-only and absorption-only evaluations at the same 30 points, to
    separate the radiative transfer overhead from the integration cost.

Run with::

    pytest tests/test_radiation/test_synchrotron/test_SEDs/bench_numerical_core.py --benchmark-only

Add ``--benchmark-save=<name>`` to persist results and compare runs with
``pytest-benchmark compare``.
"""

import numpy as np
import pytest
from astropy import units as u

from trilobite.radiation.synchrotron.SEDs.numerical.core import NumericalSynchrotronEngine
from trilobite.radiation.synchrotron.electron_distributions import PowerLaw

# ================================================== #
# Fixed benchmark parameters                         #
# ================================================== #
_N_0 = 1e4  # cm^{-3}  (elevated to push nu_a into the frequency range below)
_B_CGS = 0.1  # G
_ALPHA = np.pi / 2  # rad
_GAMMA_MIN = 1.0
_GAMMA_MAX = 1e8
_N_GAMMA = 500

# 30 frequencies spanning from well below to well above the SSA turnover.
# With N_0=1e4, B=0.1 G, ell=1e16 cm and p=3, nu_a ~ few×10^11 Hz, so this
# range straddles the optically thick/thin transition.
_NU_30 = np.geomspace(1e9, 1e15, 30) * u.Hz

_ELL = 1e16 * u.cm  # path length through the emitting region


# ================================================== #
# Fixtures                                           #
# ================================================== #


@pytest.fixture(scope="session")
def bench_engine():
    """Session-scoped engine with both kernels loaded at production resolution."""
    eng = NumericalSynchrotronEngine()
    eng.load_first_kernel(x_min=1e-6, x_max=1e3, num_points=1000)
    eng.load_avg_first_kernel(x_min=1e-6, x_max=1e3, num_points=1000)
    return eng


@pytest.fixture(scope="session")
def bench_electron_dist():
    """Session-scoped power-law electron distribution callable."""
    return PowerLaw.as_callable(norm=_N_0, p=3.0, gamma_min=_GAMMA_MIN, gamma_max=_GAMMA_MAX)


# ================================================== #
# Benchmarks                                         #
# ================================================== #


def test_bench_sed_ssa_fixed_alpha(benchmark, bench_engine, bench_electron_dist):
    """Full SED at 30 frequencies with SSA — fixed pitch angle (alpha = pi/2)."""
    benchmark(
        bench_engine.compute_rest_frame_specific_intensity,
        _NU_30,
        _ELL,
        _B_CGS * u.G,
        bench_electron_dist,
        alpha=_ALPHA,
        gamma_min=_GAMMA_MIN,
        gamma_max=_GAMMA_MAX,
        n_gamma=_N_GAMMA,
    )


def test_bench_sed_ssa_pitch_averaged(benchmark, bench_engine, bench_electron_dist):
    """Full SED at 30 frequencies with SSA — pitch-angle-averaged kernel."""
    benchmark(
        bench_engine.compute_rest_frame_specific_intensity,
        _NU_30,
        _ELL,
        _B_CGS * u.G,
        bench_electron_dist,
        alpha=None,
        gamma_min=_GAMMA_MIN,
        gamma_max=_GAMMA_MAX,
        n_gamma=_N_GAMMA,
    )


def test_bench_emissivity_only_fixed_alpha(benchmark, bench_engine, bench_electron_dist):
    """Emissivity at 30 frequencies — fixed pitch angle, no transfer overhead."""
    benchmark(
        bench_engine.compute_emissivity,
        _NU_30,
        _B_CGS * u.G,
        bench_electron_dist,
        alpha=_ALPHA,
        gamma_min=_GAMMA_MIN,
        gamma_max=_GAMMA_MAX,
        n_gamma=_N_GAMMA,
    )


def test_bench_absorption_only_fixed_alpha(benchmark, bench_engine, bench_electron_dist):
    """SSA absorption coefficient at 30 frequencies — fixed pitch angle."""
    benchmark(
        bench_engine.compute_absorption,
        _NU_30,
        _B_CGS * u.G,
        bench_electron_dist,
        alpha=_ALPHA,
        gamma_min=_GAMMA_MIN,
        gamma_max=_GAMMA_MAX,
        n_gamma=_N_GAMMA,
    )
