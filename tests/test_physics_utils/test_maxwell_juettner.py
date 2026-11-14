"""Tests for :mod:`trilobite.physics_utils.maxwell_juettner`."""

import astropy.units as u
import numpy as np
import pytest
from astropy import constants as const
from scipy.integrate import quad

from trilobite.physics_utils import maxwell_juettner as M
from trilobite.radiation.synchrotron.electron_distributions import (
    BrokenPowerLaw,
    MaxwellJuettner,
    MaxwellJuettnerBrokenPowerLaw,
    MaxwellJuettnerPowerLaw,
    PowerLaw,
)
from trilobite.physics_utils.maxwell_juettner import (
    compute_dimensionless_temperature,
    compute_mj_cdf,
    compute_mj_mean_kinetic_energy,
    compute_mj_mean_lorentz_factor,
    compute_mj_moment,
    compute_mj_number_density_from_internal_energy_density,
    compute_mj_number_density_from_total_energy_density,
    compute_mj_pdf,
    compute_mj_theta_from_kinetic_energy,
    compute_temperature_from_dimensionless,
)

THETAS = [0.05, 0.3, 1.0, 5.0]


class TestTemperatureConversion:
    def test_electron_rest_temperature(self):
        T_e = (const.m_e * const.c**2 / const.k_B).to(u.K)
        assert compute_dimensionless_temperature(T_e) == pytest.approx(1.0)

    def test_round_trip(self):
        Theta = np.array([0.01, 1.0, 20.0])
        T = compute_temperature_from_dimensionless(Theta)
        assert T.unit == u.K
        np.testing.assert_allclose(compute_dimensionless_temperature(T), Theta)

    def test_bare_float_is_kelvin(self):
        assert compute_dimensionless_temperature(1e9) == pytest.approx(compute_dimensionless_temperature(1e9 * u.K))

    def test_ion_mass(self):
        # Same temperature, heavier particle: Theta scales as 1/m.
        T = 1e12 * u.K
        ratio = compute_dimensionless_temperature(T) / compute_dimensionless_temperature(T, mass=const.m_p)
        assert ratio == pytest.approx((const.m_p / const.m_e).value)

    def test_incompatible_units_raise(self):
        with pytest.raises(ValueError):
            compute_dimensionless_temperature(1.0 * u.cm)


class TestDistribution:
    @pytest.mark.parametrize("Theta", THETAS)
    def test_pdf_is_normalized(self, Theta):
        total = quad(lambda g: compute_mj_pdf(g, Theta), 1.0, np.inf)[0]
        assert total == pytest.approx(1.0, rel=1e-6)

    def test_pdf_zero_below_support(self):
        assert compute_mj_pdf(0.5, 1.0) == 0.0

    def test_pdf_scales_with_norm(self):
        assert compute_mj_pdf(2.0, 1.0, norm=7.0) == pytest.approx(7.0 * compute_mj_pdf(2.0, 1.0))

    def test_pdf_small_theta_is_finite(self):
        # K_2(1/Theta) underflows here; the scaled form must not.
        assert np.isfinite(compute_mj_pdf(1.0001, 1e-3))

    @pytest.mark.parametrize("Theta", [1e-4, 1e-2, *THETAS, 100.0])
    def test_cdf_matches_quadrature(self, Theta):
        # Reference: adaptive quadrature of the pdf in x = gamma - 1, which sidesteps rounding of 1 + x. epsabs=0 is
        # needed: quad's default absolute tolerance (1.5e-8) is looser than the accuracy being tested.
        x = Theta * np.array([0.05, 0.3, 1.0, 3.0, 8.0, 20.0, 45.0])

        def reference(xi):
            def dF_ds(s):
                return Theta * M._opt_mj_pdf_excess(Theta * s, Theta)

            return quad(dF_ds, 0.0, xi / Theta, epsabs=0.0, epsrel=1e-13, limit=500)[0]

        expected = [reference(xi) for xi in x]
        np.testing.assert_allclose(compute_mj_cdf(1.0 + x, Theta), expected, rtol=0, atol=1e-9)

    def test_cdf_limits(self):
        assert compute_mj_cdf(1.0, 1.0) == 0.0
        assert compute_mj_cdf(0.5, 1.0) == 0.0
        assert compute_mj_cdf(1e4, 1.0) == pytest.approx(1.0, rel=1e-3)

    @pytest.mark.parametrize("Theta", [1e-3, 1.0, 100.0])
    def test_cdf_saturates_to_norm_exactly(self, Theta):
        # Beyond the integration grid, and at infinity, the CDF is exactly norm (not the discretized ~1 - 3e-1 etc.).
        far = np.array([1.0 + 1e3 * Theta, 1e12, 1e300, np.inf])
        np.testing.assert_array_equal(compute_mj_cdf(far, Theta), 1.0)
        np.testing.assert_array_equal(compute_mj_cdf(far, Theta, norm=7.0), 7.0)

    def test_cdf_is_independent_of_other_array_elements(self):
        Theta = 1e-3
        pts = 1.0 + Theta * np.array([0.1, 0.5, 1.0, 4.0, 20.0])
        alone = np.array([compute_mj_cdf(g, Theta) for g in pts])
        poisoned = compute_mj_cdf(np.concatenate([pts, [1e9, np.inf, np.nan]]), Theta)[: len(pts)]
        np.testing.assert_array_equal(compute_mj_cdf(pts, Theta), alone)
        np.testing.assert_array_equal(poisoned, alone)

    def test_cdf_monotone_and_bounded(self):
        Theta = 0.4
        c = compute_mj_cdf(1.0 + Theta * np.logspace(-4, 2, 500), Theta)
        assert np.all(np.diff(c) >= -1e-15)
        assert np.all((c >= 0.0) & (c <= 1.0))

    def test_cdf_nan_empty_and_shapes(self):
        assert np.isnan(compute_mj_cdf(np.nan, 1.0))
        assert compute_mj_cdf(np.array([]), 1.0).shape == (0,)
        assert compute_mj_cdf(np.full((2, 3), 2.0), 1.0).shape == (2, 3)
        assert isinstance(compute_mj_cdf(2.0, 1.0), float)

    def test_cdf_requires_scalar_theta(self):
        with pytest.raises(ValueError, match="scalar"):
            compute_mj_cdf(2.0, np.array([0.5, 1.0]))

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
    def test_invalid_theta_raises(self, bad):
        with pytest.raises(ValueError):
            compute_mj_pdf(2.0, bad)

    @pytest.mark.parametrize("Theta", [0.03, 0.5, 8.0])
    def test_log_pdf_is_the_log_of_the_pdf(self, Theta):
        gamma = np.geomspace(1.0, 50.0 * max(Theta, 1.0), 40)
        got = np.exp(M._opt_mj_log_pdf(gamma, Theta, 2.5))
        np.testing.assert_allclose(got, compute_mj_pdf(gamma, Theta, norm=2.5), rtol=1e-12, atol=0.0)

    def test_log_pdf_is_minus_infinity_where_the_pdf_vanishes_and_broadcasts(self):
        out = M._opt_mj_log_pdf(np.array([1.0, 1.5, 3.0]), 1.0, np.array([[1.0], [2.0]]))
        assert out.shape == (2, 3)
        assert np.all(np.isneginf(out[:, 0]))  # beta = 0 at gamma = 1


class TestMoments:
    @pytest.mark.parametrize("Theta", THETAS)
    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_closed_form_matches_quadrature(self, order, Theta):
        expected = quad(lambda g: g**order * compute_mj_pdf(g, Theta), 1.0, np.inf)[0]
        assert compute_mj_moment(order, Theta) == pytest.approx(expected, rel=1e-6)

    def test_higher_order_uses_quadrature(self):
        Theta = 0.5
        expected = quad(lambda g: g**3 * compute_mj_pdf(g, Theta), 1.0, np.inf)[0]
        assert compute_mj_moment(3, Theta) == pytest.approx(expected, rel=1e-6)

    def test_negative_order_raises(self):
        with pytest.raises(ValueError):
            compute_mj_moment(-1, 1.0)

    def test_array_theta(self):
        out = compute_mj_moment(1, np.array(THETAS))
        assert out.shape == (len(THETAS),)

    @pytest.mark.parametrize("Theta", THETAS)
    def test_approximate_mean_accuracy(self, Theta):
        # The Gammie & McKinney fit is good to ~1.3% over all Theta (worst near Theta ~ 1.7).
        exact = compute_mj_mean_lorentz_factor(Theta)
        approx = compute_mj_mean_lorentz_factor(Theta, approximate=True)
        assert approx == pytest.approx(exact, rel=2e-2)

    @pytest.mark.parametrize("Theta", THETAS)
    def test_kinetic_energy_is_mean_minus_one(self, Theta):
        assert compute_mj_mean_kinetic_energy(Theta) == pytest.approx(compute_mj_mean_lorentz_factor(Theta) - 1.0)
        assert compute_mj_mean_kinetic_energy(Theta, approximate=True) == pytest.approx(
            compute_mj_mean_lorentz_factor(Theta, approximate=True) - 1.0
        )

    def test_limits(self):
        # Non-relativistic: <gamma - 1> -> 3 Theta / 2. Ultra-relativistic: -> 3 Theta.
        assert compute_mj_mean_kinetic_energy(1e-3, approximate=True) == pytest.approx(1.5e-3, rel=1e-2)
        assert compute_mj_mean_kinetic_energy(1e3, approximate=True) == pytest.approx(3e3, rel=1e-2)

    def test_theta_inversion_round_trip(self):
        Theta = np.array([1e-3, 0.1, 1.0, 30.0])
        xi = compute_mj_mean_kinetic_energy(Theta, approximate=True)
        np.testing.assert_allclose(compute_mj_theta_from_kinetic_energy(xi), Theta, rtol=1e-10)

    def test_theta_inversion_known_point(self):
        # Theta = 1 gives xi = 21/9 = 7/3 (the identity noted in the jump-condition docs).
        assert compute_mj_theta_from_kinetic_energy(7.0 / 3.0) == pytest.approx(1.0)

    def test_theta_inversion_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            compute_mj_theta_from_kinetic_energy(0.0)


class TestSmallTheta:
    """The Bessel routes are replaced by series at tiny Theta; scipy's kve returns NaN there."""

    @pytest.mark.parametrize("Theta", [1e-14, 1e-12, 1e-10, 1e-9])
    def test_exact_means_are_finite_and_match_series(self, Theta):
        kinetic = compute_mj_mean_kinetic_energy(Theta)
        assert np.isfinite(kinetic)
        assert kinetic == pytest.approx(1.5 * Theta + 1.875 * Theta**2, rel=1e-8)
        assert compute_mj_mean_lorentz_factor(Theta) == pytest.approx(1.0 + kinetic)

    def test_second_moment_is_finite(self):
        assert compute_mj_moment(2, 1e-12) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("Theta", [1e-6, 0.99e-4, 1.01e-4, 1e-3])
    def test_exact_kinetic_energy_is_continuous_across_switch(self, Theta):
        # Compare to the approximation (itself accurate to O(Theta^3) here), so a wrong series coefficient shows up.
        exact = compute_mj_mean_kinetic_energy(Theta)
        approx = compute_mj_mean_kinetic_energy(Theta, approximate=True)
        assert exact == pytest.approx(approx, rel=1e-2 * Theta**0.5 + 1e-6)


class TestNormalization:
    @pytest.mark.parametrize("approximate", [False, True])
    @pytest.mark.parametrize("Theta", [1e-3, 0.3, 2.0, 30.0])
    def test_total_and_internal_agree(self, Theta, approximate):
        # u_tot = u_int + n m c^2, so both routes must return the same n for the same population.
        n = 3e-2 * u.cm**-3
        rest = (const.m_e * const.c**2).to(u.erg)
        mean_kinetic = compute_mj_mean_kinetic_energy(Theta, approximate=approximate)
        u_int = (n * rest * mean_kinetic).to(u.erg / u.cm**3)
        u_tot = u_int + n * rest

        from_int = compute_mj_number_density_from_internal_energy_density(u_int, Theta, approximate=approximate)
        from_tot = compute_mj_number_density_from_total_energy_density(u_tot, Theta, approximate=approximate)
        assert from_int.unit == from_tot.unit == u.cm**-3
        assert from_int.value == pytest.approx(n.value, rel=1e-10)
        assert from_tot.value == pytest.approx(n.value, rel=1e-10)

    def test_round_trip_total(self):
        Theta, n = 2.0, 3e-2 * u.cm**-3
        u_total = n * const.m_e * const.c**2 * compute_mj_mean_lorentz_factor(Theta)
        out = compute_mj_number_density_from_total_energy_density(u_total.to(u.erg / u.cm**3), Theta)
        assert out.value == pytest.approx(n.value)

    def test_approximate_toggle_changes_result_at_expected_level(self):
        # At Theta ~ 1.7 the total-energy approximation is ~1.2% off; the exact route must differ from it.
        exact = compute_mj_number_density_from_total_energy_density(1e-3, 1.7).value
        approx = compute_mj_number_density_from_total_energy_density(1e-3, 1.7, approximate=True).value
        assert exact != approx
        assert approx == pytest.approx(exact, rel=2e-2)

    def test_total_over_internal_ratio(self):
        # For the same u, n_int / n_tot = <gamma> / <gamma - 1>: ~2/(3 Theta) in the non-relativistic limit.
        Theta = 1e-3
        n_tot = compute_mj_number_density_from_total_energy_density(1.0, Theta).value
        n_int = compute_mj_number_density_from_internal_energy_density(1.0, Theta).value
        assert n_tot / n_int == pytest.approx(
            compute_mj_mean_kinetic_energy(Theta) / compute_mj_mean_lorentz_factor(Theta)
        )
        assert n_int / n_tot == pytest.approx(2.0 / (3.0 * Theta), rel=2e-3)

    def test_bare_float_is_cgs(self):
        with_units = compute_mj_number_density_from_total_energy_density(1e-3 * u.erg / u.cm**3, 1.0)
        bare = compute_mj_number_density_from_total_energy_density(1e-3, 1.0)
        assert bare.value == pytest.approx(with_units.value)

    def test_heavier_particle_needs_fewer_particles(self):
        n_e = compute_mj_number_density_from_total_energy_density(1.0 * u.erg / u.cm**3, 1.0)
        n_p = compute_mj_number_density_from_total_energy_density(1.0 * u.erg / u.cm**3, 1.0, mass=const.m_p)
        assert n_p.value == pytest.approx(n_e.value * (const.m_e / const.m_p).value)

    def test_invalid_theta_raises(self):
        with pytest.raises(ValueError):
            compute_mj_number_density_from_internal_energy_density(1.0, -1.0)


class TestElectronDistributionNormalization:
    """Every distribution fixes N0 from an energy budget through the kinetic energy <gamma - 1> (rest mass excluded)."""

    PL = {"p": 3.0, "gamma_min": 2.0, "gamma_max": 1e3}
    PL_NONREL = {"p": 3.0, "gamma_min": 1.0, "gamma_max": 1e4}
    BPL = {"p1": 2.5, "p2": 3.5, "gamma_c": 20.0, "gamma_min": 2.0, "gamma_max": 1e3}

    @staticmethod
    def _kinetic_energy_integral(cls, params):
        """int N(gamma) (gamma - 1) d gamma for norm = 1, by quadrature of the class's own pdf."""
        Theta = params.get("Theta", 1.0)
        pts = {1.0, 1 + Theta, 1 + 10 * Theta, 1 + 80 * Theta}
        upper = 1 + 80 * Theta
        if "gamma_max" in params:
            pts |= {params["gamma_min"], 10 * params["gamma_min"], 100 * params["gamma_min"], params["gamma_max"]}
            pts |= {params["gamma_c"]} if "gamma_c" in params else set()
            upper = params["gamma_max"]
        edges = sorted(x for x in pts if x <= upper)
        return sum(
            quad(lambda g: (g - 1.0) * float(cls.pdf(g, **params)), a, b, epsabs=0.0, epsrel=1e-11, limit=400)[0]
            for a, b in zip(edges[:-1], edges[1:])
        )

    CASES = [
        (PowerLaw, PL),
        (PowerLaw, PL_NONREL),
        (BrokenPowerLaw, BPL),
        (MaxwellJuettner, {"Theta": 0.05}),
        (MaxwellJuettner, {"Theta": 1.0}),
        (MaxwellJuettner, {"Theta": 20.0}),
        (MaxwellJuettnerPowerLaw, {"Theta": 0.5, "delta": 0.3, **PL}),
        (MaxwellJuettnerPowerLaw, {"Theta": 5.0, "delta": 0.7, **PL_NONREL}),
        (MaxwellJuettnerBrokenPowerLaw, {"Theta": 1.0, "delta": 0.5, **BPL}),
    ]

    @pytest.mark.parametrize(("cls", "params"), CASES)
    def test_energy_budget_is_kinetic(self, cls, params):
        from trilobite.radiation.constants import electron_rest_energy_cgs as mc2

        integral = self._kinetic_energy_integral(cls, params)
        u_th, eps_E, B, eps_B = 2.5e-3, 0.3, 0.7, 0.1

        N0_u = cls.normalize_from_energy_density(u_th, eps_E, **params).value
        N0_B = cls.normalize_from_magnetic_field(B, eps_B, eps_E, **params).value
        assert N0_u * mc2 * integral == pytest.approx(eps_E * u_th, rel=1e-8)
        assert N0_B * mc2 * integral == pytest.approx((eps_E / eps_B) * B**2 / (8 * np.pi), rel=1e-8)

    @pytest.mark.parametrize("Theta", [0.05, 1.0, 20.0])
    @pytest.mark.parametrize(
        ("mixed", "shape"),
        [(MaxwellJuettnerPowerLaw, PL), (MaxwellJuettnerBrokenPowerLaw, BPL)],
    )
    def test_pure_thermal_mixture_matches_maxwell_juettner(self, mixed, shape, Theta):
        # delta = 1 must reproduce the thermal class exactly, for every normalization route.
        kwargs = {"Theta": Theta, "delta": 1.0, **shape}
        for method, args in [
            ("normalize_from_energy_density", (2.5e-3, 0.3)),
            ("normalize_from_magnetic_field", (0.7, 0.1, 0.2)),
        ]:
            thermal = getattr(MaxwellJuettner, method)(*args, Theta=Theta).value
            assert getattr(mixed, method)(*args, **kwargs).value == pytest.approx(thermal, rel=1e-10)

    @pytest.mark.parametrize("Theta", [0.05, 1.0, 20.0])
    @pytest.mark.parametrize("delta", [0.2, 0.7])
    @pytest.mark.parametrize(
        ("mixed", "component", "shape"),
        [(MaxwellJuettnerPowerLaw, PowerLaw, PL), (MaxwellJuettnerBrokenPowerLaw, BrokenPowerLaw, BPL)],
    )
    def test_delta_is_the_thermal_share_of_the_energy_budget(self, mixed, component, shape, Theta, delta):
        # Normalizing each component against its own share of epsilon_E (delta, 1 - delta), as the mixed SED does,
        # gives the same electrons as the mixture with its single N0.
        u_th, eps_E = 2.5e-3, 0.3
        N_thermal = MaxwellJuettner.normalize_from_energy_density(u_th, delta * eps_E, Theta=Theta).value
        N_non_thermal = component.normalize_from_energy_density(u_th, (1.0 - delta) * eps_E, **shape).value
        N0 = mixed.normalize_from_energy_density(u_th, eps_E, Theta=Theta, delta=delta, **shape).value

        gamma = np.geomspace(1.0 + 1e-3, shape["gamma_max"], 60)
        separate = MaxwellJuettner.pdf(gamma, N_thermal, Theta=Theta) + component.pdf(gamma, N_non_thermal, **shape)
        np.testing.assert_allclose(mixed.pdf(gamma, N0, Theta=Theta, delta=delta, **shape), separate, rtol=1e-9)

    @pytest.mark.parametrize("Theta", [1e-3, 0.3, 5.0])
    def test_maxwell_juettner_matches_utility(self, Theta):
        n = compute_mj_number_density_from_internal_energy_density(0.3 * 2.5e-3, Theta)
        assert MaxwellJuettner.normalize_from_energy_density(2.5e-3, 0.3, Theta=Theta).value == pytest.approx(
            n.value, rel=1e-10
        )

    @pytest.mark.parametrize("p", [2.2, 2.5, 3.0, 3.5])
    def test_power_law_down_to_gamma_min_one(self, p):
        # int_1^inf (gamma - 1) gamma^-p d gamma = 1/(p - 2) - 1/(p - 1): finite and positive at gamma_min = 1.
        from trilobite.radiation.constants import electron_rest_energy_cgs as mc2

        N0 = PowerLaw.normalize_from_energy_density(1e-3, 0.1, p=p, gamma_min=1.0, gamma_max=np.inf).value
        assert N0 == pytest.approx(0.1 * 1e-3 / (mc2 * (1 / (p - 2) - 1 / (p - 1))), rel=1e-12)

    def test_energy_normalization_refused_when_ill_conditioned(self):
        # All of the distribution sits within 1e-10 of gamma = 1: <gamma - 1> ~ 1e-21, below the cancellation limit.
        squeezed = {"p": 3.0, "gamma_min": 1.0, "gamma_max": 1.0 + 1e-10}
        with pytest.raises(ValueError, match="almost no kinetic energy"):
            PowerLaw.normalize_from_energy_density(1e-3, 0.1, **squeezed)
        with pytest.raises(ValueError, match="almost no kinetic energy"):
            PowerLaw.normalize_from_magnetic_field(1.0, 0.1, 0.1, **squeezed)
        assert PowerLaw.normalize_from_energy_density(1e-3, 0.1, p=3.0, gamma_min=1.0, gamma_max=1e4).value > 0
