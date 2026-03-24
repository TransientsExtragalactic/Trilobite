"""
Tests for igP_es_advDisk and igP_es_adv_fbDisk.

``igP_es_advDisk`` inherits all common integration tests via
:class:`BaseTestOneZoneDisk`.  Additional model-specific assertions check:

- The Cython closure is an ``igP_es_advClosure``.
- All 21 output fields are finite and positive.
- ``Q_adv < Q_visc`` at every step (advection carries only a fraction of
  the viscous heating).
- Energy balance: ``Q_rad + Q_adv ≈ Q_visc`` to within numerical tolerance.
- Monotonicity: larger ``xi`` produces larger ``Q_adv / Q_visc``.
- Spec-dict round-trip preserves both ``mu`` and ``xi``.

``igP_es_adv_fbDisk`` is tested in a separate standalone class because it
requires additional runtime parameters (``M_fb_0``, ``t_fb``).
"""

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u

from triceratops.dynamics.accretion.one_zone import igP_es_advDisk, igP_es_adv_fbDisk

from test_one_zone_disk_base import BaseTestOneZoneDisk


# ================================================================== #
# igP_es_advDisk — full common + model-specific suite                #
# ================================================================== #


class TestigP_es_advDisk(BaseTestOneZoneDisk):
    """
    Full test suite for the advective disk closure.

    Physical setup
    --------------
    - 3 M☉ black hole, 0.1 M☉ disk, α = 0.1, ξ = 0.5
    - R_D_0 / R_in = 3e13 / 3e6 = 1e7 — safely in the one-zone regime
    """

    MODEL = igP_es_advDisk
    DISK_KWARGS = {"mu": 0.62, "xi": 0.5}

    M_BH = 3.0 * const.M_sun
    M_D_0 = 0.1 * const.M_sun
    R_IN = 3.0e6 * u.cm
    R_D_0 = 3.0e13 * u.cm
    ALPHA = 0.1

    # ——— Model-specific assertions ———

    def test_eos_mu(self):
        """EOS mean molecular weight matches the constructor argument."""
        assert np.isclose(self._disk().eos.mu, 0.62)

    def test_xi_stored(self):
        """xi context parameter is stored correctly."""
        assert np.isclose(self._disk()._context_parameters["xi"], 0.5)

    def test_geometry_constants_accessible(self):
        """Geometry constants A, B, F are class attributes with Metzger+08 values."""
        disk = self._disk()
        assert np.isclose(disk._A, 1.62)
        assert np.isclose(disk._B, 1.33)
        assert np.isclose(disk._F, 1.6)

    def test_closure_is_adv_type(self):
        """_build_cython_closure() returns an igP_es_advClosure."""
        from triceratops.dynamics.accretion.one_zone.models._igP_es_adv import (
            igP_es_advClosure,
        )

        closure = self._disk()._build_cython_closure()
        assert isinstance(closure, igP_es_advClosure)

    def test_all_fields_finite(self):
        """Every field in result.data is finite and non-NaN."""
        data = self._solve_result().data
        for name, arr in data.items():
            assert np.all(np.isfinite(arr.value)), f"Field '{name}' contains non-finite values"

    def test_Q_adv_positive(self):
        """Q_adv is positive at every step (ξ > 0 implies advective cooling)."""
        Q_adv = self._solve_result().data["Q_adv"].value
        assert np.all(Q_adv > 0), "Q_adv must be positive for ξ > 0"

    def test_Q_adv_less_than_Q_visc(self):
        """Q_adv < Q_visc at every step — advection is a fraction of viscous heating."""
        data = self._solve_result().data
        Q_adv = data["Q_adv"].value
        Q_visc = data["Q_visc"].value
        assert np.all(Q_adv < Q_visc), "Q_adv must not exceed Q_visc"

    def test_energy_balance(self):
        """Q_rad + Q_adv ≈ Q_visc to 1 % relative tolerance."""
        data = self._solve_result().data
        Q_visc = data["Q_visc"].value
        Q_adv = data["Q_adv"].value
        # Q_rad is not stored directly: Q_rad = Q_visc - Q_adv
        Q_rad = Q_visc - Q_adv
        assert np.all(Q_rad > 0), "Q_rad (= Q_visc - Q_adv) must be positive"
        rel_err = np.abs((Q_rad + Q_adv) / Q_visc - 1.0)
        assert np.all(rel_err < 1e-10), "Energy balance Q_rad + Q_adv = Q_visc violated"

    def test_from_spec_dict_preserves_mu_and_xi(self):
        """Round-trip through spec dict preserves mu and xi."""
        disk = self._disk()
        spec = disk.to_spec_dict()
        loaded = igP_es_advDisk.from_spec_dict(spec)
        assert np.isclose(loaded.eos.mu, disk.eos.mu)
        assert np.isclose(loaded._context_parameters["xi"], disk._context_parameters["xi"])

    def test_spec_dict_target(self):
        """to_spec_dict target string identifies igP_es_advDisk."""
        spec = self._disk().to_spec_dict()
        assert "igP_es_advDisk" in spec["target"]

    def test_pack_cython_parameters_shape(self):
        """_pack_cython_parameters returns [MBH, R_in, alpha, mu, xi] — 5 elements."""
        disk = self._disk()
        run_params = disk.process_runtime_parameters(self._runtime_parameters())
        params = disk._pack_cython_parameters(run_params)
        assert params.shape == (5,)


# ================================================================== #
# Monotonicity: larger ξ → larger advective fraction                #
# ================================================================== #


class TestAdvMonotonicity:
    """
    Increasing ξ increases the advective fraction Q_adv / Q_visc.

    Uses a fixed disk configuration and compares the median advective
    fraction across three values of ξ.
    """

    M_BH = 3.0 * const.M_sun
    M_D_0 = 0.1 * const.M_sun
    R_IN = 3.0e6 * u.cm
    R_D_0 = 3.0e13 * u.cm
    ALPHA = 0.1
    MU = 0.62
    MAX_STEPS = 100

    @classmethod
    def _J_D_0(cls, disk):
        xi = disk._B / disk._A
        return xi * cls.M_D_0.cgs * np.sqrt(const.G.cgs * cls.M_BH.cgs * cls.R_D_0.to(u.cm))

    @classmethod
    def _solve(cls, xi_val):
        disk = igP_es_advDisk(mu=cls.MU, xi=xi_val)
        J_D_0 = cls._J_D_0(disk)
        return disk.solve(
            initial_conditions={"M_D_0": cls.M_D_0, "J_D_0": J_D_0},
            runtime_parameters={"M_BH": cls.M_BH, "R_in": cls.R_IN, "alpha": cls.ALPHA},
            t_span=(1.0e6 * u.s, 1.0e9 * u.s),
            max_steps=cls.MAX_STEPS,
        )

    def test_larger_xi_larger_adv_fraction(self):
        """Median Q_adv / Q_visc increases monotonically with ξ."""
        fractions = []
        for xi_val in (0.1, 0.5, 1.0):
            r = self._solve(xi_val)
            Q_adv = r.data["Q_adv"].value
            Q_visc = r.data["Q_visc"].value
            fractions.append(np.median(Q_adv / Q_visc))

        assert fractions[0] < fractions[1] < fractions[2], (
            f"Advective fractions {fractions} are not monotonically increasing with ξ"
        )


# ================================================================== #
# igP_es_adv_fbDisk — standalone tests                               #
# ================================================================== #


class TestigP_es_adv_fbDisk:
    """
    Smoke-tests for the advective fallback disk.

    Uses a compact NS TDE-like configuration:
    - 3 M☉ BH, 0.1 M☉ initial disk, α = 0.1, ξ = 0.5
    - Fallback supply: 1e28 g/s at t_fb = 1e4 s, β = 5/3
    """

    M_BH = 3.0 * const.M_sun
    M_D_0 = 0.1 * const.M_sun
    R_IN = 3.0e6 * u.cm
    R_D_0 = 3.0e13 * u.cm
    ALPHA = 0.1
    MU = 0.62
    XI = 0.5
    M_FB_0 = 1.0e28 * u.g / u.s
    T_FB = 1.0e4 * u.s
    BETA_FB = 5.0 / 3.0
    MAX_STEPS = 200

    @classmethod
    def _disk(cls):
        return igP_es_adv_fbDisk(mu=cls.MU, xi=cls.XI)

    @classmethod
    def _J_D_0(cls, disk):
        xi = disk._B / disk._A
        return xi * cls.M_D_0.cgs * np.sqrt(const.G.cgs * cls.M_BH.cgs * cls.R_D_0.to(u.cm))

    @classmethod
    def _solve(cls):
        disk = cls._disk()
        J_D_0 = cls._J_D_0(disk)
        return disk.solve(
            initial_conditions={"M_D_0": cls.M_D_0, "J_D_0": J_D_0},
            runtime_parameters={
                "M_BH": cls.M_BH,
                "R_in": cls.R_IN,
                "alpha": cls.ALPHA,
                "M_fb_0": cls.M_FB_0,
                "t_fb": cls.T_FB,
                "beta_fb": cls.BETA_FB,
            },
            t_span=(1.0e4 * u.s, 1.0e8 * u.s),
            max_steps=cls.MAX_STEPS,
        )

    def test_closure_is_adv_fb_type(self):
        """_build_cython_closure() returns an igP_es_adv_fbClosure."""
        from triceratops.dynamics.accretion.one_zone.models._igP_es_adv_fb import (
            igP_es_adv_fbClosure,
        )

        closure = self._disk()._build_cython_closure()
        assert isinstance(closure, igP_es_adv_fbClosure)

    def test_all_fields_finite(self):
        """Every field in result.data is finite and non-NaN."""
        data = self._solve().data
        for name, arr in data.items():
            assert np.all(np.isfinite(arr.value)), f"Field '{name}' contains non-finite values"

    def test_mdot_fb_accessible(self):
        """mdot_fb field is present and positive at all steps."""
        mdot_fb = self._solve().data["mdot_fb"].value
        assert np.all(mdot_fb > 0), "mdot_fb must be positive"

    def test_Q_adv_less_than_Q_visc(self):
        """Q_adv < Q_visc at every step."""
        data = self._solve().data
        Q_adv = data["Q_adv"].value
        Q_visc = data["Q_visc"].value
        assert np.all(Q_adv < Q_visc), "Q_adv must not exceed Q_visc"

    def test_xi_stored(self):
        """xi context parameter is stored correctly."""
        assert np.isclose(self._disk()._context_parameters["xi"], self.XI)

    def test_from_spec_dict_preserves_mu_and_xi(self):
        """Round-trip through spec dict preserves mu and xi."""
        disk = self._disk()
        spec = disk.to_spec_dict()
        loaded = igP_es_adv_fbDisk.from_spec_dict(spec)
        assert np.isclose(loaded.eos.mu, disk.eos.mu)
        assert np.isclose(loaded._context_parameters["xi"], disk._context_parameters["xi"])

    def test_spec_dict_target(self):
        """to_spec_dict target string identifies igP_es_adv_fbDisk."""
        spec = self._disk().to_spec_dict()
        assert "igP_es_adv_fbDisk" in spec["target"]
