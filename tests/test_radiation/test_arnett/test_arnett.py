"""Tests for the Arnett diffusion engines."""

import astropy.units as u
import numpy as np
import pytest
from numba import njit
from scipy.integrate import quad, solve_ivp

from trilobite.radiation.arnett import (
    FINITE_RADIUS_CLOCK,
    HOMOLOGOUS_CLOCK,
    ArnettEngine,
    Channel,
    MagnetarArnettEngine,
    MagnetarNickelArnettEngine,
    NickelCobaltArnettEngine,
    QuadratureRule,
    build_solver,
)

DAY = 86400.0
MSUN = 1.989e33
C = 2.99792458e10
BETA = 13.8

# Reference 56Ni / 56Co physics, written independently of the engine.
EPS_NI, EPS_CO, TAU_NI, TAU_CO, F_POS = 3.9e10, 6.78e9, 8.8 * DAY, 111.3 * DAY, 0.032


# ---------------------------------------------------------------- #
# Independent reference implementation                             #
# ---------------------------------------------------------------- #
def _ejecta(M_ej, E_ej, kappa):
    v = np.sqrt(10 * E_ej / (3 * M_ej))
    t_d = np.sqrt(2 * kappa * M_ej / (BETA * C * v))
    return v, t_d


def _trap(t, M_ej, E_ej, kappa_gamma):
    A = 9 * kappa_gamma * M_ej**2 / (40 * np.pi * E_ej)
    return -np.expm1(-A / t**2)


def _ni_channels(t, M_Ni):
    """(gamma power, positron power) per the Bateman solution."""
    e_ni, e_co = np.exp(-t / TAU_NI), np.exp(-t / TAU_CO)
    p_ni = EPS_NI * e_ni
    p_co = EPS_CO * TAU_CO / (TAU_CO - TAU_NI) * (e_co - e_ni)
    return M_Ni * (p_ni + (1 - F_POS) * p_co), M_Ni * F_POS * p_co


def _magnetar_power(t, P, B, n, I=1e45, R=1e6):
    om = 2 * np.pi / P
    om_dot = B**2 * R**6 * om**3 / (6 * I * C**3)
    L0 = I * om * om_dot
    t_sd = om / ((n - 1) * om_dot)
    return L0 * (1 + t / t_sd) ** (-(n + 1) / (n - 1)), t_sd


def _reference(tq, power, M_ej, E_ej, kappa, R_0=0.0, E_th_0=0.0, breaks=None):
    """L(tq) from the integral solution, using scipy.quad and the clock written out by hand."""
    v, t_d = _ejecta(M_ej, E_ej, kappa)
    a = R_0 / v
    u_of = lambda t: (t**2 + 2 * a * t) / t_d**2
    dudt = lambda t: 2 * (t + a) / t_d**2
    L_init = E_th_0 * dudt(0.0)
    f = lambda t: power(t) * dudt(t) * np.exp(-(u_of(tq) - u_of(t)))
    return L_init * np.exp(-u_of(tq)) + quad(f, 0, tq, points=breaks, epsabs=0, epsrel=1e-11, limit=500)[0]


def _ni_power(M_ej, E_ej, M_Ni, kappa_gamma, kappa_positron, gamma_trap, positron_trap):
    def power(t):
        g, e = _ni_channels(t, M_Ni)
        if gamma_trap:
            g = g * _trap(t, M_ej, E_ej, kappa_gamma)
        if positron_trap:
            e = e * _trap(t, M_ej, E_ej, kappa_positron)
        return g + e

    return power


PARAMS = dict(M_ej=2 * MSUN, E_ej=1e51, kappa=0.1, M_Ni=0.3 * MSUN)
TIMES = np.array([1, 5, 10, 20, 40, 80, 150, 300, 600]) * DAY


# ---------------------------------------------------------------- #
# Kernel                                                           #
# ---------------------------------------------------------------- #
@njit
def _const_power(t, row):
    """Constant injection; ``row = [F]``."""
    return row[0]


def _run_kernel(clock, clock_row, F, t, l_init=0.0, window_depth=40.0):
    solver = build_solver(clock, _const_power, window_depth=window_depth)
    t = np.asarray(t, float)[:, None]
    out = np.empty_like(t)
    solver(
        t,
        np.array([clock_row]),
        np.array([[F]]),
        np.empty((1, 0)),
        np.empty((1, 0)),
        np.array([l_init]),
        QuadratureRule.gauss_legendre(96),
        out,
    )
    return out[:, 0]


@pytest.mark.parametrize("window_depth", [40.0, 3.0])
def test_constant_injection_homologous(window_depth):
    # Constant injection is the same at every depth, so truncating the window at S clock units gives
    # exactly F (1 - exp(-min(u, S))).
    t_d = 10 * DAY
    t = np.array([0.1, 0.5, 1, 2, 5, 20]) * t_d
    L = _run_kernel(HOMOLOGOUS_CLOCK, [t_d], 1e43, t, window_depth=window_depth)
    exact = 1e43 * (1 - np.exp(-np.minimum((t / t_d) ** 2, window_depth)))
    np.testing.assert_allclose(L, exact, rtol=1e-10)


@pytest.mark.parametrize("window_depth", [40.0, 3.0])
def test_constant_injection_finite_radius(window_depth):
    t_d, a, L_init = 10 * DAY, 4 * DAY, 3e42
    t = np.array([0.05, 0.5, 1, 2, 5, 20]) * t_d
    L = _run_kernel(FINITE_RADIUS_CLOCK, [t_d, a], 1e43, t, l_init=L_init, window_depth=window_depth)
    u_t = (t**2 + 2 * a * t) / t_d**2
    exact = 1e43 * (1 - np.exp(-np.minimum(u_t, window_depth))) + L_init * np.exp(-u_t)
    np.testing.assert_allclose(L, exact, rtol=1e-10)


def test_finite_radius_reduces_to_homologous():
    t_d = 10 * DAY
    t = np.array([0.3, 1, 3, 30]) * t_d
    np.testing.assert_allclose(
        _run_kernel(FINITE_RADIUS_CLOCK, [t_d, 0.0], 1e43, t), _run_kernel(HOMOLOGOUS_CLOCK, [t_d], 1e43, t), rtol=1e-12
    )


def test_nonpositive_time_gives_zero():
    np.testing.assert_array_equal(_run_kernel(HOMOLOGOUS_CLOCK, [DAY], 1e43, [-1.0, 0.0]), [0.0, 0.0])


def test_build_solver_validation():
    with pytest.raises(TypeError):
        build_solver(HOMOLOGOUS_CLOCK, lambda t, p: 1.0)
    with pytest.raises(TypeError):
        build_solver("homologous", _const_power)
    with pytest.raises(ValueError):
        build_solver(HOMOLOGOUS_CLOCK, _const_power, window_depth=0.0)


def test_deposition_slot():
    # A trapped constant source: F (1 - exp(-A/t^2)) on the homologous clock vs. the quadrature reference.
    from trilobite.radiation.arnett import _gamma_trapping

    t_d, A, F = 10 * DAY, (5 * DAY) ** 2, 1e43
    t = np.array([0.5, 1, 3]) * t_d
    solver = build_solver(HOMOLOGOUS_CLOCK, _const_power, deposition_function=_gamma_trapping)
    out = np.empty((3, 1))
    solver(
        t[:, None],
        np.array([[t_d]]),
        np.array([[F]]),
        np.array([[A]]),
        np.empty((1, 0)),
        np.zeros(1),
        QuadratureRule.gauss_legendre(96),
        out,
    )
    expected = [
        quad(
            lambda x: F * -np.expm1(-A / x**2) * 2 * x / t_d**2 * np.exp(-(tt**2 - x**2) / t_d**2),
            0,
            tt,
            epsabs=0,
            epsrel=1e-11,
        )[0]
        for tt in t
    ]
    np.testing.assert_allclose(out[:, 0], expected, rtol=1e-9)


# ---------------------------------------------------------------- #
# Quadrature                                                       #
# ---------------------------------------------------------------- #
@pytest.mark.parametrize(
    "rule",
    [QuadratureRule.gauss_legendre(40), QuadratureRule.gauss_legendre_power(40, 3), QuadratureRule.tanh_sinh(80)],
)
def test_quadrature_integrates_polynomial(rule):
    assert np.sum(rule.weights * rule.nodes**3) == pytest.approx(0.25, rel=1e-6)
    np.testing.assert_allclose(rule.nodes + rule.one_minus_nodes, 1.0, atol=1e-15)
    assert not rule.nodes.flags.writeable


def test_quadrature_validation():
    with pytest.raises(ValueError):
        QuadratureRule.gauss_legendre(0)
    with pytest.raises(ValueError):
        QuadratureRule.gauss_legendre_power(10, power=0)
    with pytest.raises(ValueError):
        QuadratureRule.tanh_sinh(1)


# ---------------------------------------------------------------- #
# Nickel-cobalt engine                                             #
# ---------------------------------------------------------------- #
@pytest.mark.parametrize("gamma_trap", [True, False])
@pytest.mark.parametrize("positron_trap", [True, False])
def test_nickel_matches_quadrature_reference(gamma_trap, positron_trap):
    engine = NickelCobaltArnettEngine(gamma_trapping=gamma_trap, positron_trapping=positron_trap)
    kw = dict(kappa_gamma=0.03, kappa_positron=7.0)
    L = engine._compute_luminosity_cgs(TIMES, **PARAMS, **kw)
    power = _ni_power(
        PARAMS["M_ej"],
        PARAMS["E_ej"],
        PARAMS["M_Ni"],
        kw["kappa_gamma"],
        kw["kappa_positron"],
        gamma_trap,
        positron_trap,
    )
    ref = [_reference(t, power, PARAMS["M_ej"], PARAMS["E_ej"], PARAMS["kappa"]) for t in TIMES]
    np.testing.assert_allclose(L, ref, rtol=1e-8)


def test_gamma_trapping_switch():
    on = NickelCobaltArnettEngine(gamma_trapping=True)
    off = NickelCobaltArnettEngine(gamma_trapping=False)
    assert on.gamma_trapping and not off.gamma_trapping
    L_on, L_off = on._compute_luminosity_cgs(TIMES, **PARAMS), off._compute_luminosity_cgs(TIMES, **PARAMS)
    # Early on the ejecta is opaque to gamma rays (identical to double precision); later it leaks.
    assert np.all(L_on <= L_off * (1 + 1e-14))
    assert np.all(L_on[4:] < L_off[4:])
    # Opaque to gamma rays -> full trapping recovered
    np.testing.assert_allclose(on._compute_luminosity_cgs(TIMES, kappa_gamma=1e8, **PARAMS), L_off, rtol=1e-8)
    # kappa_gamma is irrelevant when trapping is off
    np.testing.assert_array_equal(off._compute_luminosity_cgs(TIMES, kappa_gamma=1.0, **PARAMS), L_off)


def test_positron_trapping_switch():
    base = NickelCobaltArnettEngine(positron_trapping=False)
    leaky = NickelCobaltArnettEngine(positron_trapping=True)
    assert leaky.positron_trapping and not base.positron_trapping
    L0 = base._compute_luminosity_cgs(TIMES, **PARAMS)
    L1 = leaky._compute_luminosity_cgs(TIMES, **PARAMS)
    # Positrons matter only at late times, where they are the dominant heating channel.
    assert np.all(L1 <= L0 * (1 + 1e-14))
    np.testing.assert_allclose(L1[:4], L0[:4], rtol=1e-6)
    assert L1[-1] < L0[-1]
    # With the default opacity positrons are still ~95% trapped at 600 d, so the effect is a few percent...
    assert L1[-1] / L0[-1] == pytest.approx(0.96, abs=0.02)
    # ...and with a smaller effective opacity they leak substantially and positrons stop dominating the tail.
    leakier = leaky._compute_luminosity_cgs(TIMES, kappa_positron=0.3, **PARAMS)
    assert leakier[-1] < 0.8 * L0[-1] and np.all(leakier <= L1 * (1 + 1e-14))
    # Perfectly trapping positrons recovers the default.
    np.testing.assert_allclose(leaky._compute_luminosity_cgs(TIMES, kappa_positron=1e12, **PARAMS), L0, rtol=1e-8)
    # kappa_positron is irrelevant if the switch is off.
    np.testing.assert_array_equal(base._compute_luminosity_cgs(TIMES, kappa_positron=0.1, **PARAMS), L0)


def test_nickel_peak_is_physical():
    t = np.linspace(2, 60, 200) * DAY
    L = NickelCobaltArnettEngine()._compute_luminosity_cgs(t, **PARAMS)
    assert 10 * DAY < t[np.argmax(L)] < 40 * DAY
    assert 5e42 < L.max() < 3e43


def test_nickel_late_time_tail_follows_heating():
    # Far beyond t_d with full trapping, L -> instantaneous heating.
    engine = NickelCobaltArnettEngine(gamma_trapping=False)
    t = np.array([400, 500]) * DAY
    L = engine._compute_luminosity_cgs(t, **PARAMS)
    heating = sum(_ni_channels(t, PARAMS["M_Ni"]))
    np.testing.assert_allclose(L, heating, rtol=0.05)


# ---------------------------------------------------------------- #
# Finite progenitor radius                                         #
# ---------------------------------------------------------------- #
def _ode_reference(t_eval, power, M_ej, E_ej, kappa, R_0, E_th_0):
    """Integrate dL/dt = (du/dt)(P - L), L(0) = E_th_0 du/dt(0), independently of the quadrature solver."""
    v, t_d = _ejecta(M_ej, E_ej, kappa)
    a = R_0 / v
    dudt = lambda t: 2 * (t + a) / t_d**2
    t0 = 1e-6
    u0 = (t0**2 + 2 * a * t0) / t_d**2
    sol = solve_ivp(
        lambda t, L: dudt(t) * (power(t) - L),
        (t0, t_eval[-1]),
        [E_th_0 * dudt(0.0) * np.exp(-u0)],
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-12,
        atol=1e20,
    )
    return sol.y[0]


def test_finite_radius_matches_ode():
    engine = NickelCobaltArnettEngine(clock="finite_radius", positron_trapping=True)
    R_0, E_th_0 = 3e13, 5e48
    L = engine._compute_luminosity_cgs(TIMES, R_0=R_0, E_th_0=E_th_0, **PARAMS)
    power = _ni_power(PARAMS["M_ej"], PARAMS["E_ej"], PARAMS["M_Ni"], 0.03, 7.0, True, True)
    ref = _ode_reference(TIMES, power, PARAMS["M_ej"], PARAMS["E_ej"], PARAMS["kappa"], R_0, E_th_0)
    np.testing.assert_allclose(L, ref, rtol=1e-6)
    # And the quadrature reference with the hand-written clock agrees too.
    quad_ref = [_reference(t, power, PARAMS["M_ej"], PARAMS["E_ej"], PARAMS["kappa"], R_0, E_th_0) for t in TIMES]
    np.testing.assert_allclose(L, quad_ref, rtol=1e-8)


def test_finite_radius_physics():
    homologous = NickelCobaltArnettEngine()
    finite = NickelCobaltArnettEngine(clock="finite_radius")
    t = np.array([2, 5, 10, 30]) * DAY
    base = homologous._compute_luminosity_cgs(t, **PARAMS)
    # R_0 = 0 reproduces the homologous result.
    np.testing.assert_allclose(finite._compute_luminosity_cgs(t, R_0=0.0, E_th_0=0.0, **PARAMS), base, rtol=1e-12)
    # A finite R_0 speeds the clock up: more light early.
    assert np.all(finite._compute_luminosity_cgs(t, R_0=1e14, **PARAMS)[:2] > base[:2])
    # Stored heat adds light early and fades away.
    boosted = finite._compute_luminosity_cgs(t, R_0=1e14, E_th_0=1e49, **PARAMS)
    plain = finite._compute_luminosity_cgs(t, R_0=1e14, **PARAMS)
    assert np.all(boosted > plain)
    assert (boosted[-1] - plain[-1]) / plain[-1] < (boosted[0] - plain[0]) / plain[0]


def test_radius_requires_finite_radius_clock():
    with pytest.raises(ValueError, match="finite_radius"):
        NickelCobaltArnettEngine()._compute_luminosity_cgs(TIMES, R_0=1e13, **PARAMS)
    with pytest.raises(ValueError, match="finite_radius"):
        NickelCobaltArnettEngine()._compute_luminosity_cgs(TIMES, E_th_0=1e48, **PARAMS)
    with pytest.raises(ValueError):
        NickelCobaltArnettEngine(clock="finite_radius")._compute_luminosity_cgs(TIMES, R_0=-1.0, **PARAMS)
    with pytest.raises(ValueError):
        NickelCobaltArnettEngine(clock="bogus")


# ---------------------------------------------------------------- #
# Magnetar engines                                                 #
# ---------------------------------------------------------------- #
MAG = dict(M_ej=5 * MSUN, E_ej=2e51, kappa=0.1)
MAG_TIMES = np.array([2, 10, 30, 80, 200]) * DAY


def _magnetar_ref(t, P, B, n, trap, kappa_magnetar=0.03, **kw):
    def power(tt):
        p, _ = _magnetar_power(tt, P, B, n, **kw)
        return p * _trap(tt, MAG["M_ej"], MAG["E_ej"], kappa_magnetar) if trap else p

    _, t_sd = _magnetar_power(0.0, P, B, n, **kw)
    return _reference(t, power, MAG["M_ej"], MAG["E_ej"], MAG["kappa"], breaks=[t_sd] if t_sd < t else None)


@pytest.mark.parametrize("trapping", [True, False])
@pytest.mark.parametrize("n", [3.0, 2.0, 5.0, 1.5])
@pytest.mark.parametrize("P, B", [(2e-3, 1e14), (5e-3, 3e13), (1e-2, 1e15)])
def test_magnetar_matches_quadrature_reference(trapping, n, P, B):
    engine = MagnetarArnettEngine(gamma_trapping=trapping)
    L = engine._compute_luminosity_cgs(MAG_TIMES, P_spin=P, B_dipole=B, braking_index=n, **MAG)
    ref = [_magnetar_ref(t, P, B, n, trapping) for t in MAG_TIMES]
    np.testing.assert_allclose(L, ref, rtol=1e-6)


def test_braking_index_default_is_dipole():
    engine = MagnetarArnettEngine()
    kw = dict(P_spin=3e-3, B_dipole=2e14, **MAG)
    np.testing.assert_array_equal(
        engine._compute_luminosity_cgs(MAG_TIMES, **kw),
        engine._compute_luminosity_cgs(MAG_TIMES, braking_index=3, **kw),
    )


def test_braking_index_changes_light_curve():
    engine = MagnetarArnettEngine(gamma_trapping=False)
    kw = dict(P_spin=3e-3, B_dipole=2e14, **MAG)
    L3 = engine._compute_luminosity_cgs(MAG_TIMES, braking_index=3.0, **kw)
    L5 = engine._compute_luminosity_cgs(MAG_TIMES, braking_index=5.0, **kw)
    assert not np.allclose(L3, L5, rtol=0.1)
    # Higher braking index: slower decay at late times, since the injected power falls as t^-(n+1)/(n-1).
    # (n = 5 -> t^-1.5 vs n = 3 -> t^-2), but with a longer t_sd and a lower L_0 at fixed Omega_0 only after
    # normalisation; just check both are finite and positive here.
    assert np.all(np.isfinite(L5)) and np.all(L5 > 0)


@pytest.mark.parametrize("n", [2.0, 3.0, 5.0])
def test_magnetar_energy_budget(n):
    from trilobite.radiation import arnett

    L0, t_sd, q = arnett._magnetar_spin_down_cgs(2e-3, 1e14, n, 1e45, 1e6)
    E_rot = 0.5 * 1e45 * (2 * np.pi / 2e-3) ** 2
    assert L0 * t_sd * (n - 1) / 2 == pytest.approx(E_rot, rel=1e-12)
    assert q == pytest.approx((n + 1) / (n - 1))
    # Dipole timescale for n = 3
    if n == 3.0:
        om = 2 * np.pi / 2e-3
        assert t_sd == pytest.approx(3 * 1e45 * C**3 / (1e14**2 * 1e6**6 * om**2), rel=1e-12)


def test_braking_index_validation():
    engine = MagnetarArnettEngine()
    with pytest.raises(ValueError, match="braking_index"):
        engine._compute_luminosity_cgs(MAG_TIMES, P_spin=3e-3, B_dipole=2e14, braking_index=1.0, **MAG)


# ---------------------------------------------------------------- #
# Magnetar + nickel                                                #
# ---------------------------------------------------------------- #
@pytest.mark.parametrize("gamma_trap", [True, False])
@pytest.mark.parametrize("positron_trap", [True, False])
@pytest.mark.parametrize("clock", ["homologous", "finite_radius"])
def test_magnetar_nickel_is_sum_of_engines(gamma_trap, positron_trap, clock):
    # The problem is linear in the deposited power, so the combined engine is the sum of the two single-source
    # engines (each with its own trapping), on the same clock.
    kw = dict(M_ej=5 * MSUN, E_ej=2e51, kappa=0.1)
    extra = dict(R_0=2e13, E_th_0=0.0) if clock == "finite_radius" else {}
    spin = dict(P_spin=3e-3, B_dipole=2e14, braking_index=4.0, kappa_magnetar=0.05)
    nick = dict(M_Ni=0.5 * MSUN, kappa_gamma=0.04, kappa_positron=5.0)
    both = MagnetarNickelArnettEngine(gamma_trapping=gamma_trap, positron_trapping=positron_trap, clock=clock)
    ni = NickelCobaltArnettEngine(gamma_trapping=gamma_trap, positron_trapping=positron_trap, clock=clock)
    mag = MagnetarArnettEngine(gamma_trapping=gamma_trap, clock=clock)
    L = both._compute_luminosity_cgs(MAG_TIMES, **kw, **spin, **nick, **extra)
    expected = ni._compute_luminosity_cgs(MAG_TIMES, **kw, **nick, **extra) + mag._compute_luminosity_cgs(
        MAG_TIMES, **kw, **spin, **extra
    )
    np.testing.assert_allclose(L, expected, rtol=1e-9)


def test_channel_structure():
    both = MagnetarNickelArnettEngine(positron_trapping=True)
    assert [c.name for c in both.channels] == ["nickel_gamma", "nickel_positron", "magnetar"]
    assert [c.opacity for c in both.channels] == ["kappa_gamma", "kappa_positron", "kappa_magnetar"]
    off = MagnetarNickelArnettEngine(gamma_trapping=False)
    assert [c.opacity for c in off.channels] == [None, None, None]


# ---------------------------------------------------------------- #
# Interface                                                        #
# ---------------------------------------------------------------- #
class _Dummy(ArnettEngine):
    def _get_channels(self):
        return [Channel("a", _const_power, 1), Channel("a", _const_power, 1)]

    def _source_parameters(self, **source):
        raise NotImplementedError


def test_duplicate_channel_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        _Dummy()


def test_units_and_shapes():
    engine = NickelCobaltArnettEngine()
    kw = dict(M_ej=2 * u.Msun, E_ej=1e51 * u.erg, kappa=0.1 * u.cm**2 / u.g, M_Ni=0.3 * u.Msun)
    L = engine.compute_luminosity([10, 20, 50] * u.day, **kw)
    assert L.unit == u.erg / u.s and L.shape == (3,)
    np.testing.assert_allclose(
        L.value,
        engine._compute_luminosity_cgs(np.array([10, 20, 50]) * DAY, **{k: v.cgs.value for k, v in kw.items()}),
        rtol=1e-3,  # astropy's Msun differs slightly from the module constant
    )

    scalar = engine(20 * u.day, **kw)
    assert scalar.shape == () and scalar.value == pytest.approx(L[1].value)

    # Batched realizations: (n_times, n_realizations), each column equal to the scalar-parameter result.
    M_Ni = np.array([0.1, 0.2, 0.3]) * u.Msun
    batch = engine.compute_luminosity([10, 20] * u.day, **{**kw, "M_Ni": M_Ni})
    assert batch.shape == (2, 3)
    for j, m in enumerate(M_Ni):
        np.testing.assert_allclose(
            batch[:, j].value, engine.compute_luminosity([10, 20] * u.day, **{**kw, "M_Ni": m}).value
        )

    # Bare floats are CGS
    bare = engine.compute_luminosity(20 * DAY, M_ej=2 * u.Msun.to(u.g), E_ej=1e51, kappa=0.1, M_Ni=0.3 * u.Msun.to(u.g))
    assert bare.value == pytest.approx(scalar.value)


def test_unit_conversion_for_magnetar_engine():
    engine = MagnetarNickelArnettEngine(clock="finite_radius")
    kw = dict(
        M_ej=5 * u.Msun,
        E_ej=2e51 * u.erg,
        kappa=0.1 * u.cm**2 / u.g,
        M_Ni=0.2 * u.Msun,
        P_spin=3 * u.ms,
        B_dipole=2e14 * u.G,
        braking_index=3,
        R_0=100 * u.Rsun,
        E_th_0=1e48 * u.erg,
    )
    L = engine.compute_luminosity([5, 20, 60] * u.day, **kw)
    assert L.unit == u.erg / u.s and np.all(L.value > 0)
    with pytest.raises(ValueError):
        engine.compute_luminosity(10 * u.day, **{**kw, "P_spin": 3 * u.G})


def test_bad_inputs():
    engine = NickelCobaltArnettEngine()
    with pytest.raises(ValueError):
        engine._compute_luminosity_cgs(TIMES, **{**PARAMS, "M_Ni": np.ones(3) * MSUN, "kappa": np.ones(4) * 0.1})
    with pytest.raises(ValueError):
        engine._compute_luminosity_cgs(TIMES, **{**PARAMS, "M_Ni": np.ones((2, 2))})
    with pytest.raises(TypeError, match="unexpected parameter"):
        engine.compute_luminosity(10 * u.day, M_ej=1, E_ej=1, kappa=1, M_Ni=1, bogus=1)
    with pytest.raises(TypeError):
        NickelCobaltArnettEngine(quadrature="gl")
    with pytest.raises(ValueError):
        NickelCobaltArnettEngine(beta=-1)


def test_diffusion_time_value():
    t_d = NickelCobaltArnettEngine().compute_diffusion_time(2 * u.Msun, 1e51 * u.erg, 0.1 * u.cm**2 / u.g)
    assert 10 < t_d.to_value(u.day) < 30


def test_repr_and_shared_solver():
    a, b = NickelCobaltArnettEngine(), NickelCobaltArnettEngine()
    assert all(x is y for x, y in zip(a._solvers, b._solvers))
    assert NickelCobaltArnettEngine(gamma_trapping=False)._solvers[0] is not a._solvers[0]
    assert NickelCobaltArnettEngine(clock="finite_radius")._solvers[0] is not a._solvers[0]
    assert "gamma_trapping=True" in str(a) and "positron_trapping=False" in str(a) and repr(a).startswith("<")
    assert "clock='finite_radius'" in str(NickelCobaltArnettEngine(clock="finite_radius"))


def test_channel_solvers_are_shared_between_engines():
    # Channels are solved one at a time, so engines that share a source share its compiled solver.
    ni = NickelCobaltArnettEngine()
    mag = MagnetarArnettEngine()
    both = MagnetarNickelArnettEngine()
    assert both._solvers[:2] == ni._solvers and both._solvers[2] is mag._solvers[0]


def test_empty_channel_list_rejected():
    class _Empty(_Dummy):
        def _get_channels(self):
            return []

    with pytest.raises(ValueError, match="at least one"):
        _Empty()


def test_source_parameter_plumbing():
    # Missing / unknown source parameters surface as TypeErrors from the private backend too.
    ni = NickelCobaltArnettEngine()
    with pytest.raises(TypeError):
        ni._compute_luminosity_cgs(TIMES, M_ej=PARAMS["M_ej"], E_ej=PARAMS["E_ej"], kappa=0.1)  # no M_Ni
    with pytest.raises(TypeError):
        ni._compute_luminosity_cgs(TIMES, bogus=1.0, **PARAMS)
    both = MagnetarNickelArnettEngine()
    with pytest.raises(TypeError, match="bogus"):
        both._compute_luminosity_cgs(TIMES, bogus=1.0, P_spin=3e-3, B_dipole=2e14, **PARAMS)
    with pytest.raises(TypeError):  # magnetar parameters are required for the combined engine
        both._compute_luminosity_cgs(TIMES, **PARAMS)


def test_combined_engine_defaults_come_from_each_source():
    # Omitted source parameters take each source's own defaults, so spelling them out changes nothing.
    both = MagnetarNickelArnettEngine(positron_trapping=True)
    kw = dict(P_spin=3e-3, B_dipole=2e14, **PARAMS)
    explicit = dict(
        braking_index=3.0,
        kappa_gamma=0.03,
        kappa_positron=7.0,
        kappa_magnetar=0.03,
        I_ns=1e45,
        R_ns=1e6,
        R_0=0.0,
        E_th_0=0.0,
    )
    np.testing.assert_array_equal(
        both._compute_luminosity_cgs(MAG_TIMES, **kw), both._compute_luminosity_cgs(MAG_TIMES, **kw, **explicit)
    )


@pytest.mark.parametrize("engine_cls", [NickelCobaltArnettEngine, MagnetarArnettEngine, MagnetarNickelArnettEngine])
def test_parameter_units_match_source_signatures(engine_cls):
    # _PARAMETER_UNITS (which drives compute_luminosity) must list exactly the common parameters plus the
    # arguments of the source-parameter functions the engine delegates to.
    import inspect

    from trilobite.radiation import arnett

    helpers = {
        NickelCobaltArnettEngine: [arnett._nickel_parameters],
        MagnetarArnettEngine: [arnett._magnetar_parameters],
        MagnetarNickelArnettEngine: [arnett._nickel_parameters, arnett._magnetar_parameters],
    }[engine_cls]
    expected = set(arnett._COMMON_UNITS) | {name for fn in helpers for name in inspect.signature(fn).parameters}
    assert set(engine_cls._PARAMETER_UNITS) == expected


# ---------------------------------------------------------------- #
# Where the trapping is applied: deposition (default) or emergence #
# ---------------------------------------------------------------- #
def _leaky_kw(kappa_magnetar):
    return dict(P_spin=2e-3, B_dipole=1e14, kappa_magnetar=kappa_magnetar, **MAG)


def test_default_trapping_mode_is_deposition():
    assert MagnetarArnettEngine().trapping_mode == "deposition"
    assert "trapping_mode='deposition'" in str(MagnetarArnettEngine())


def test_unknown_trapping_mode_is_rejected():
    with pytest.raises(ValueError, match="trapping_mode"):
        MagnetarArnettEngine(trapping_mode="late")


@pytest.mark.parametrize("kappa_magnetar", [0.01, 0.03, 0.2])
def test_emergence_mode_is_the_leakage_factor_times_the_unleaky_luminosity(kappa_magnetar):
    """Nicholl et al. (2017), Eq. 5: L_out = (1 - exp(-A / t^2)) x the diffused luminosity of the full source."""
    t = MAG_TIMES
    emergence = MagnetarArnettEngine(trapping_mode="emergence")
    unleaky = MagnetarArnettEngine(gamma_trapping=False)
    L = emergence._compute_luminosity_cgs(t, **_leaky_kw(kappa_magnetar))
    L_full = unleaky._compute_luminosity_cgs(t, **_leaky_kw(kappa_magnetar))
    A = 9 * kappa_magnetar * MAG["M_ej"] ** 2 / (40 * np.pi * MAG["E_ej"])
    np.testing.assert_allclose(L, -np.expm1(-A / np.asarray(t) ** 2) * L_full, rtol=1e-12)


def test_the_two_modes_differ_when_the_source_leaks():
    t = MAG_TIMES
    kw = _leaky_kw(0.02)
    deposition = MagnetarArnettEngine(trapping_mode="deposition")._compute_luminosity_cgs(t, **kw)
    emergence = MagnetarArnettEngine(trapping_mode="emergence")._compute_luminosity_cgs(t, **kw)
    assert not np.allclose(deposition, emergence, rtol=1e-2)


def test_the_two_modes_agree_when_the_source_is_fully_trapped():
    t = MAG_TIMES
    kw = _leaky_kw(1.0e4)  # an enormous opacity: nothing leaks
    deposition = MagnetarArnettEngine(trapping_mode="deposition")._compute_luminosity_cgs(t, **kw)
    emergence = MagnetarArnettEngine(trapping_mode="emergence")._compute_luminosity_cgs(t, **kw)
    np.testing.assert_allclose(deposition, emergence, rtol=1e-9)


def test_the_mode_is_irrelevant_without_trapping():
    t = MAG_TIMES
    kw = _leaky_kw(0.02)
    a = MagnetarArnettEngine(gamma_trapping=False, trapping_mode="deposition")._compute_luminosity_cgs(t, **kw)
    b = MagnetarArnettEngine(gamma_trapping=False, trapping_mode="emergence")._compute_luminosity_cgs(t, **kw)
    np.testing.assert_array_equal(a, b)


def test_emergence_mode_works_for_several_channels_and_realizations():
    """Each channel gets its own factor, and parameter arrays still give one realization per entry."""
    t = np.array([20.0, 60.0, 150.0]) * DAY
    M = np.array([2.0, 5.0]) * MSUN
    kw = dict(M_ej=M, E_ej=np.array([1e51, 2e51]), kappa=0.1, M_Ni=np.array([0.2, 0.4]) * MSUN)
    engine = NickelCobaltArnettEngine(trapping_mode="emergence")
    L = engine._compute_luminosity_cgs(t, **kw)
    assert L.shape == (3, 2)
    for j in range(2):
        single = {k: (v[j] if np.ndim(v) else v) for k, v in kw.items()}
        np.testing.assert_allclose(L[:, j], engine._compute_luminosity_cgs(t, **single), rtol=1e-12)


def test_emergence_mode_cannot_be_combined_with_a_custom_emergence_function():
    @njit
    def _unit_emergence(t, params):
        return 1.0

    class CustomEmergence(MagnetarArnettEngine):
        def _get_emergence_function(self):
            return _unit_emergence

    CustomEmergence(trapping_mode="deposition")  # fine
    CustomEmergence(gamma_trapping=False, trapping_mode="emergence")  # nothing leaks, so nothing to combine
    with pytest.raises(ValueError, match="custom emergence"):
        CustomEmergence(trapping_mode="emergence")
