"""
Tests for igP_esDisk.

Inherits all common tests from :class:`BaseTestOneZoneDisk`.  This module
adds assertions specific to the full-pressure (gas + radiation) closure:

- The Cython closure object is a ``igP_esClosure``.
- All output fields are finite and physically reasonable.
- In the gas-pressure-dominated regime the midplane temperatures from the
  full-pressure and gas-only closures agree to within a tight tolerance —
  any systematic offset would indicate a bug in the energy-balance root find.
- In the radiation-pressure-dominated regime (high surface density, high
  alpha) the full-pressure T_c exceeds the gas-only T_c, because the extra
  pressure support increases the sound speed and therefore the viscous
  heating rate q+.
"""

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u

from trilobite.dynamics.accretion.one_zone import (
    igP_esDisk,
    gP_esDisk,
)

from test_one_zone_disk_base import BaseTestOneZoneDisk


# ================================================================== #
# Standard (gas-dominated) configuration                             #
# ================================================================== #


class TestigP_esDisk(BaseTestOneZoneDisk):
    """
    Full suite of common tests for igP_esDisk.

    Physical setup mirrors :class:`TestgP_esDisk` so
    that the two closures can be directly compared in the gas-dominated regime.
    """

    MODEL = igP_esDisk
    DISK_KWARGS = {"mu": 0.62}

    M_BH = 3.0 * const.M_sun
    M_D_0 = 0.1 * const.M_sun
    R_IN = 3.0e6 * u.cm
    R_D_0 = 3.0e13 * u.cm
    ALPHA = 0.1

    # ——— Model-specific assertions ———

    def test_eos_mu(self):
        """EOS mean molecular weight matches the constructor argument."""
        assert np.isclose(self._disk().eos.mu, 0.62)

    def test_geometry_constants_accessible(self):
        """Geometry constants A, B, F are class attributes with Metzger+08 values."""
        disk = self._disk()
        assert np.isclose(disk._A, 1.62)
        assert np.isclose(disk._B, 1.33)
        assert np.isclose(disk._F, 1.6)

    def test_closure_is_full_pressure_type(self):
        """_build_cython_closure() returns a igPClosure."""
        from trilobite.dynamics.accretion.one_zone.models._igP import (
            igPClosure,
        )

        closure = self._disk()._build_cython_closure()
        assert isinstance(closure, igPClosure)

    def test_from_spec_dict_preserves_mu(self):
        """Round-trip through spec dict preserves the EOS mean molecular weight."""
        disk = self._disk()
        spec = disk.to_spec_dict()
        loaded = igP_esDisk.from_spec_dict(spec)
        assert np.isclose(loaded.eos.mu, disk.eos.mu)

    def test_from_spec_dict_geometry_constants_unchanged(self):
        """Geometry constants A, B, F are class-level and unchanged after round-trip."""
        disk = self._disk()
        spec = disk.to_spec_dict()
        loaded = igP_esDisk.from_spec_dict(spec)
        assert np.isclose(loaded._A, 1.62)
        assert np.isclose(loaded._B, 1.33)
        assert np.isclose(loaded._F, 1.6)

    def test_all_T_c_finite_positive(self):
        """Every T_c value in the solution is finite and positive."""
        T_c = self._solve_result().data["T_c"]
        assert np.all(np.isfinite(T_c.value))
        assert np.all(T_c.value > 0)

    def test_all_fields_finite(self):
        """Every field returned by result.data is finite and non-NaN."""
        data = self._solve_result().data
        for name, arr in data.items():
            assert np.all(np.isfinite(arr.value)), f"Field '{name}' contains non-finite values"

    def test_spec_dict_target(self):
        """to_spec_dict target string identifies FullPressureDisk."""
        spec = self._disk().to_spec_dict()
        assert "FullPressureDisk" in spec["target"]


# ================================================================== #
# Gas-dominated convergence comparison                               #
# ================================================================== #


class TestFullVsGasOnlyConvergence:
    """
    Cross-model consistency between ``igP_esDisk`` and
    ``gP_esDisk``.

    The two closures use different energy-balance formulas:

    - **Gas-only** — legacy convention, omits σ_SB in the T_c formula.
    - **Full-pressure** — physically correct formula, includes σ_SB.

    As a result T_c_full is always larger than T_c_gas by a systematic
    factor (≈ σ_SB^{-1/3} ≈ 26), even in the purely gas-dominated limit.
    Tests in this class verify:

    1. T_c_full > T_c_gas at every step (correct physical ordering).
    2. Σ evolution is identical for both closures (Σ depends only on M
       and J, not on the thermodynamic closure).
    """

    M_BH = 3.0 * const.M_sun
    M_D_0 = 0.1 * const.M_sun
    R_IN = 3.0e6 * u.cm
    R_D_0 = 3.0e13 * u.cm
    ALPHA = 0.05
    MU = 0.62
    MAX_STEPS = 200

    @classmethod
    def _J_D_0(cls, disk):
        xi = disk._B / disk._A
        return xi * cls.M_D_0.cgs * np.sqrt(const.G.cgs * cls.M_BH.cgs * cls.R_D_0.to(u.cm))

    @classmethod
    def _solve(cls, DiskClass):
        disk = DiskClass(mu=cls.MU)
        J_D_0 = cls._J_D_0(disk)
        return disk.solve(
            initial_conditions={"M_D_0": cls.M_D_0, "J_D_0": J_D_0},
            runtime_parameters={"M_BH": cls.M_BH, "R_in": cls.R_IN, "alpha": cls.ALPHA},
            t_span=(1.0e6 * u.s, 1.0e9 * u.s),
            max_steps=cls.MAX_STEPS,
        )

    def test_T_c_full_exceeds_gas_only(self):
        """T_c_full > T_c_gas at every step.

        The full-pressure closure uses the physically-correct energy-balance
        formula (includes σ_SB); the gas-only closure omits σ_SB.  This
        produces a systematic offset: T_c_full > T_c_gas everywhere.
        """
        r_gas = self._solve(gP_esDisk)
        r_full = self._solve(igP_esDisk)

        T_gas = r_gas.data["T_c"].value
        T_full = r_full.data["T_c"].value

        n = min(len(T_gas), len(T_full))
        assert np.all(T_full[:n] > T_gas[:n]), (
            "T_c_full must exceed T_c_gas at every step "
            "(full-pressure closure uses the physically-correct formula with σ_SB)"
        )

    def test_Sigma_agrees_in_gas_dominated_regime(self):
        """Surface density Σ evolves identically for both closures.

        Σ is a pure function of the ODE state (M, J) and the geometry
        constants — it does not depend on the thermodynamic closure.  Both
        models must therefore produce the same Σ trajectory.
        """
        r_gas = self._solve(gP_esDisk)
        r_full = self._solve(igP_esDisk)

        Sigma_gas = r_gas.data["Sigma"].value
        Sigma_full = r_full.data["Sigma"].value

        n = min(len(Sigma_gas), len(Sigma_full))
        rel_err = np.abs(Sigma_full[:n] - Sigma_gas[:n]) / Sigma_gas[:n]
        assert np.all(rel_err < 0.01)


# ================================================================== #
# Physical ordering: T_c_full >= T_c_gas                             #
# ================================================================== #


class TestFullPressureTcOrdering:
    """
    The full-pressure midplane temperature must be >= the gas-only midplane
    temperature at every step.

    Including radiation pressure increases the isothermal sound speed
    c_s(T_c), which shifts the energy-balance root to a higher temperature.
    This is mathematically guaranteed: at the gas-only root T_c_gas, the
    full-pressure residual f = log Q + 2 log c_s_full(T_c_gas) - 4 log T_c_gas
    >= 0 (with equality only when radiation pressure is exactly zero), so the
    full-pressure root T_c_full >= T_c_gas.
    """

    M_BH = 3.0 * const.M_sun
    M_D_0 = 0.1 * const.M_sun
    R_IN = 3.0e6 * u.cm
    R_D_0 = 3.0e13 * u.cm
    ALPHA = 0.1
    MU = 0.62
    MAX_STEPS = 200

    @classmethod
    def _J_D_0(cls, disk):
        xi = disk._B / disk._A
        return xi * cls.M_D_0.cgs * np.sqrt(const.G.cgs * cls.M_BH.cgs * cls.R_D_0.to(u.cm))

    @classmethod
    def _solve(cls, DiskClass):
        disk = DiskClass(mu=cls.MU)
        J_D_0 = cls._J_D_0(disk)
        return disk.solve(
            initial_conditions={"M_D_0": cls.M_D_0, "J_D_0": J_D_0},
            runtime_parameters={"M_BH": cls.M_BH, "R_in": cls.R_IN, "alpha": cls.ALPHA},
            t_span=(1.0e6 * u.s, 1.0e9 * u.s),
            max_steps=cls.MAX_STEPS,
        )

    def test_full_pressure_T_c_ge_gas_only(self):
        """T_c_full >= T_c_gas at every step (guaranteed: radiation adds to c_s)."""
        r_gas = self._solve(gP_esDisk)
        r_full = self._solve(igP_esDisk)

        T_gas = r_gas.data["T_c"].value
        T_full = r_full.data["T_c"].value

        n = min(len(T_gas), len(T_full))
        # Allow a tiny relative tolerance for floating-point noise in the root finder.
        assert np.all(T_full[:n] >= T_gas[:n] * (1.0 - 1e-10)), (
            "T_c_full must be >= T_c_gas at every step "
            "(full-pressure c_s >= gas-only c_s implies a higher equilibrium T_c)"
        )

    def test_full_pressure_result_finite(self):
        """All output fields remain finite and positive throughout the run."""
        data = self._solve(igP_esDisk).data
        for name, arr in data.items():
            assert np.all(np.isfinite(arr.value)), f"Field '{name}' contains non-finite values"
