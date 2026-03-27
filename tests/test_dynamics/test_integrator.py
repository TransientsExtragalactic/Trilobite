"""
Tests for :mod:`triceratops.dynamics.accretion.one_zone._integrator`.

Coverage
--------
The test suite is split into two groups:

1. **Validation-only** tests (no ready closure required) — run immediately
   once the Cython extension is compiled.
2. **Integration** tests (require a concrete :class:`OneZoneClosure`
   subclass with all three C function pointers installed).

To run this file:

.. code-block:: bash

    pytest tests/test_dynamics/test_integrator.py -v

The whole file is skipped if the extension module has not been compiled yet.

Result array layout
-------------------
``run_one_zone_model`` returns ``(n_result_fields, n_steps)``: rows are
physical fields, columns are time steps.  For
``gPClosure`` the field indices are:
    0: step_index, 1: t,      2: M,      3: J,       4: R,      5: Sigma,
    6: Omega,      7: T_eff,  8: T_c,    9: tau,     10: cs,    11: nu,
    12: q_visc,   13: dM_dt, 14: dJ_dt, 15: dt,     16: t_visc,
    17: H,        18: H/R,   19: rho
"""

import numpy as np
import pytest

_integrator = pytest.importorskip(
    "triceratops.dynamics.accretion.one_zone.integrator",
    reason="integrator.pyx not compiled — run `pip install -e '.[dev]'` first.",
)

from triceratops.dynamics.accretion.one_zone.closure import OneZoneClosure  # noqa: E402
from triceratops.dynamics.accretion.one_zone.integrator import run_one_zone_model  # noqa: E402

# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

# Physical constants (CGS)
_M_SUN_G = 1.989e33  # g
_M_BH = 3.0 * _M_SUN_G  # g
_M_DISK = 0.1 * _M_SUN_G  # g
_R_IN = 3.0e6  # cm
_ALPHA = 0.1  # dimensionless
_MU = 0.62  # mean molecular weight

# Rough J_D for a disk with R_D = 3e13 cm, using J = xi*M*sqrt(G*M_BH*R_D)
_G_CGS = 6.67430e-8  # cm^3 g^-1 s^-2
_R_D_INIT = 3.0e13  # cm
_XI = 1.33 / 1.62  # B/A
_J_DISK = _XI * _M_DISK * np.sqrt(_G_CGS * _M_BH * _R_D_INIT)  # g cm^2 s^-1

_INITIAL_STATE = np.array([_M_DISK, _J_DISK], dtype=np.float64)
_PARAMETERS = np.array([_M_BH, _R_IN, _ALPHA, _MU], dtype=np.float64)


def _make_closure():
    """Return a ready gPClosure."""
    from triceratops.dynamics.accretion.one_zone.models._gP import (
        gPClosure,
    )

    return gPClosure()


# ------------------------------------------------------------------ #
# Group 1 — Validation / attribute access (no ready closure needed)  #
# ------------------------------------------------------------------ #


class TestOneZoneClosureAttributes:
    """Basic attribute and readiness checks for the base extension type."""

    def test_n_result_fields_readable(self):
        """``n_result_fields`` is accessible and matches the constructor arg."""
        closure = OneZoneClosure(n_result_fields=7)
        assert closure.n_result_fields == 7

    def test_n_result_fields_different_values(self):
        """Different values of ``n_result_fields`` are stored correctly."""
        for n in (1, 4, 10, 100):
            assert OneZoneClosure(n_result_fields=n).n_result_fields == n

    def test_unready_closure_is_not_ready(self):
        """Default ``OneZoneClosure`` has NULL function pointers → not ready."""
        closure = OneZoneClosure(n_result_fields=4)
        assert not closure.is_ready()

    def test_gas_pressure_closure_is_ready(self):
        """gPClosure has all pointers set."""
        assert _make_closure().is_ready()


class TestRunOneZoneModelValidation:
    """Input-validation checks for :func:`run_one_zone_model`."""

    def test_unready_closure_raises_value_error(self):
        """Uninitialised closure raises :exc:`ValueError` immediately."""
        closure = OneZoneClosure(n_result_fields=4)
        with pytest.raises(ValueError, match="uninitialised function pointers"):
            run_one_zone_model(
                _INITIAL_STATE.copy(),
                _PARAMETERS.copy(),
                t_start=0.0,
                t_end=1.0,
                max_steps=10,
                closure=closure,
            )

    def test_wrong_initial_state_shape_raises(self):
        """``initial_state`` with length != 2 raises :exc:`ValueError`."""
        closure = _make_closure()
        bad_state = np.array([_M_DISK, _J_DISK, 0.0], dtype=np.float64)
        with pytest.raises(ValueError, match="initial_state must have shape"):
            run_one_zone_model(
                bad_state,
                _PARAMETERS.copy(),
                t_start=0.0,
                t_end=1.0,
                max_steps=10,
                closure=closure,
            )

    def test_wrong_parameters_shape_raises(self):
        """``parameters`` with length != 5 raises :exc:`ValueError`."""
        closure = _make_closure()
        bad_params = np.array([_M_BH, _R_IN], dtype=np.float64)
        with pytest.raises(ValueError, match="parameters must have shape"):
            run_one_zone_model(
                _INITIAL_STATE.copy(),
                bad_params,
                t_start=0.0,
                t_end=1.0,
                max_steps=10,
                closure=closure,
            )

    def test_t_end_not_after_t_start_raises(self):
        """``t_end <= t_start`` raises :exc:`ValueError`."""
        closure = _make_closure()
        with pytest.raises(ValueError, match="t_end"):
            run_one_zone_model(
                _INITIAL_STATE.copy(),
                _PARAMETERS.copy(),
                t_start=1.0,
                t_end=0.0,
                max_steps=10,
                closure=closure,
            )

    def test_t_end_equal_t_start_raises(self):
        """``t_end == t_start`` also raises :exc:`ValueError`."""
        closure = _make_closure()
        with pytest.raises(ValueError, match="t_end"):
            run_one_zone_model(
                _INITIAL_STATE.copy(),
                _PARAMETERS.copy(),
                t_start=0.0,
                t_end=0.0,
                max_steps=10,
                closure=closure,
            )


# ------------------------------------------------------------------ #
# Group 2 — Integration tests                                        #
# ------------------------------------------------------------------ #


class TestRunOneZoneModelIntegration:
    """Physics and shape tests using gPClosure."""

    def test_smoke_returns_float64_array(self):
        """Wrapper returns a C-contiguous float64 ndarray."""
        closure = _make_closure()
        result = run_one_zone_model(
            _INITIAL_STATE.copy(),
            _PARAMETERS.copy(),
            t_start=1e6,
            t_end=1e9,
            max_steps=100,
            closure=closure,
        )
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        print(result.flags)
        assert result.flags["C_CONTIGUOUS"]

    def test_result_shape(self):
        """Result has shape (n_result_fields, n_steps)."""
        closure = _make_closure()
        result = run_one_zone_model(
            _INITIAL_STATE.copy(),
            _PARAMETERS.copy(),
            t_start=1e6,
            t_end=1e9,
            max_steps=100,
            closure=closure,
        )
        assert result.ndim == 2
        assert result.shape[0] == closure.n_result_fields

    def test_initial_condition_in_column_zero(self):
        """Column 0 encodes the initial state (t=t_start, M=M_disk, J=J_disk)."""
        closure = _make_closure()
        t_start = 1e6
        result = run_one_zone_model(
            _INITIAL_STATE.copy(),
            _PARAMETERS.copy(),
            t_start=t_start,
            t_end=1e9,
            max_steps=100,
            closure=closure,
        )
        # Row 1 = t, row 2 = M, row 3 = J
        assert np.isclose(result[1, 0], t_start)
        assert np.isclose(result[2, 0], _M_DISK)
        assert np.isclose(result[3, 0], _J_DISK)

    def test_early_termination_at_t_end(self):
        """Result has fewer columns than max_steps + 1 when t_end is reached early."""
        closure = _make_closure()
        max_steps = 10_000
        # Short t_end so the integrator stops on the t_end condition.
        result = run_one_zone_model(
            _INITIAL_STATE.copy(),
            _PARAMETERS.copy(),
            t_start=1e6,
            t_end=1e7,
            max_steps=max_steps,
            closure=closure,
        )
        assert result.shape[1] < max_steps + 1, "Integrator did not terminate early — t_end check missing."

    def test_disk_mass_monotonically_decreasing(self):
        """Disk mass M_D (row 2) decreases monotonically for an isolated disk."""
        closure = _make_closure()
        result = run_one_zone_model(
            _INITIAL_STATE.copy(),
            _PARAMETERS.copy(),
            t_start=1e6,
            t_end=1e9,
            max_steps=500,
            closure=closure,
        )
        m_d = result[2, :]  # row 2 = M
        assert np.all(np.diff(m_d) <= 0.0), "Disk mass M_D is not monotonically decreasing — physics regression."
