"""The analytic closure, the distribution classes, and the spectral inversion must share one N0 convention."""

import numpy as np
import pytest

from trilobite.radiation.synchrotron.electron_distributions import BrokenPowerLaw, PowerLaw
from trilobite.radiation.synchrotron.SEDs.one_zone import _closure
from trilobite.radiation.synchrotron.spectral_inversion import _SSAMicrophysics

EPS_E, EPS_B = 0.3, 0.1

# gamma_min = 1 is the non-relativistic regime, where the kinetic and total conventions differ by up to a factor p - 1.
PL_CASES = [(3.0, 1.0, 1e4), (2.5, 1.0, np.inf), (2.2, 1e3, 1e7), (3.0, 2.0, 1e3)]


@pytest.mark.parametrize(("p", "gamma_min", "gamma_max"), PL_CASES)
def test_no_cooling_closure_matches_class(p, gamma_min, gamma_max):
    # The closure returns log(N0 / B^2); the class is evaluated at B = 1.
    from_class = PowerLaw._normalize_from_magnetic_field(
        1.0, EPS_B, EPS_E, p=p, gamma_min=gamma_min, gamma_max=gamma_max
    )
    from_closure = np.exp(_closure.compute_log_N0_no_cooling(gamma_min, p, EPS_E, EPS_B, gamma_max))
    assert from_closure == pytest.approx(from_class, rel=1e-12)


@pytest.mark.parametrize(("p", "gamma_min", "gamma_max"), PL_CASES)
def test_spectral_inversion_matches_class(p, gamma_min, gamma_max):
    microphysics = _SSAMicrophysics(p, EPS_E, EPS_B, 1.0, 1.0, None, True, gamma_min, gamma_max)
    from_class = PowerLaw._normalize_from_magnetic_field(
        1.0, EPS_B, EPS_E, p=p, gamma_min=gamma_min, gamma_max=gamma_max
    )
    assert microphysics.A_N == pytest.approx(from_class, rel=1e-12)


@pytest.mark.parametrize(("p", "gamma_min", "gamma_c", "gamma_max"), [(2.5, 1.0, 30.0, 1e5), (2.2, 1e3, 1e5, 1e7)])
def test_slow_cooling_closure_matches_class(p, gamma_min, gamma_c, gamma_max):
    shape = {"p1": p, "p2": p + 1.0, "gamma_c": gamma_c, "gamma_min": gamma_min, "gamma_max": gamma_max}
    from_class = BrokenPowerLaw._normalize_from_magnetic_field(1.0, EPS_B, EPS_E, **shape)
    from_closure = np.exp(_closure.compute_log_N0_slow_cooling(gamma_min, gamma_c, p, EPS_E, EPS_B, gamma_max))
    assert from_closure == pytest.approx(from_class, rel=1e-12)


@pytest.mark.parametrize(("p", "gamma_m", "gamma_c", "gamma_max"), [(2.5, 30.0, 1.0, 1e5), (2.2, 1e5, 1e3, 1e7)])
def test_fast_cooling_closure_matches_class(p, gamma_m, gamma_c, gamma_max):
    # Fast cooling: slope -2 from gamma_c to gamma_m, then -(p + 1) beyond.
    shape = {"p1": 2.0, "p2": p + 1.0, "gamma_c": gamma_m, "gamma_min": gamma_c, "gamma_max": gamma_max}
    from_class = BrokenPowerLaw._normalize_from_magnetic_field(1.0, EPS_B, EPS_E, **shape)
    from_closure = np.exp(_closure.compute_log_N0_fast_cooling(gamma_m, gamma_c, p, EPS_E, EPS_B, gamma_max))
    assert from_closure == pytest.approx(from_class, rel=1e-12)
