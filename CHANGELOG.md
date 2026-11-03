# Changelog

All notable changes to Trilobite are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `Diggins26_TwoComponentModel` (`trilobite/models/LFBOTs/`) — two-component (equatorial and polar) thin-shell shock and synchrotron model for LFBOTs with an axisymmetric CSM, free-free attenuation (composition-consistent emission measure), full redshift treatment, a paired `forward_model` inference path, a `tabulate`/`compute_dynamics` analysis path, and an explicit NaN-on-infeasible policy with `diagnose()`.
- `LogGaussianCensoredLikelihood` (`trilobite/inference/likelihood/base.py`) — censored Gaussian likelihood in log10 space with an optional systematic floor in dex (`sys_dex`); suited to positive observables spanning several decades.
- `AxisymmetricCSMDensityProfile` (`trilobite/dynamics/profiles/csm.py`) — stationary axisymmetric CSM with a spherical exterior power law plus a disk-confined, radially truncated interior power law; the polar angle is a profile parameter, so a sightline is a frozen `as_optimized_callable`. Includes solid-angle-fraction helpers.
- One-zone accretion disk ODE integrator with Cython backend (`trilobite/dynamics/accretion/one_zone/`)
- Bracket root-finder Cython utility (`trilobite/math_utils/_bracket_root_finder.pyx`)
- TOPS opacity table support (`trilobite/radiation/opacity/grey_opacity/_tops_table.pyx`)
- Additional Rosseland mean opacity laws: electron scattering, Kramers free-free, Kramers bound-free, combined Kramers+ES variants
- Synchrotron SED gallery documentation pages
- A changelog file (`CHANGELOG.md`) to track changes across versions.
- `RadioPhotometryContainer.extract_lightcurve(band, bandwidth)` — extracts all observations within a frequency window and returns a `RadioLightCurveContainer` (`trilobite/data/photometry.py`)
- `OpticalPhotometryContainer.extract_lightcurve(band)` — extracts all observations for a named filter band and returns an `OpticalLightCurveContainer` (`trilobite/data/optical_photometry.py`)
- `trilobite.radiation.synchrotron.electron_distributions` — new module providing class-based relativistic electron distribution interface (power-law, Maxwell-Jüttner, thermal, and mixed populations) with methods for normalization, moments, and single-particle synchrotron emissivity; replaces the function-based `microphysics.py` API.
- Inhomogeneous multi-zone synchrotron engine (`InhomogeneousCylinderEngine`) in `trilobite/radiation/synchrotron/SEDs/numerical/inhomogeneous.py` — evaluates radiative transfer across a grid of zones and sums projected contributions to the observed flux density.
- Aspherical synchrotron geometry models in `trilobite/radiation/synchrotron/SEDs/numerical/aspherical.py` — extends the one-zone slab machinery to non-spherical source geometries.
- `PowerLaw_Cooling_SSA_SynchrotronSED` — analytical one-zone SED class that includes both synchrotron self-absorption and radiative cooling in the broken-power-law regime.
- Pitch-angle averaged synchrotron constants `c_5` and `c_6` in `trilobite/radiation/synchrotron/utils.py`, enabling isotropic pitch-angle distributions throughout the cooling and SED pipelines.
- Comprehensive test suite expansion for the synchrotron subsystem: `test_numerical_core.py` (numerical SED correctness), `test_cooling.py` (cooling engine and pitch-angle averaged rates), `test_electron_distributions.py` (distribution normalization, moments, emissivity), and `test_utilities.py` (low-level synchrotron constants).
- Benchmarking script `bench_numerical_core.py` for profiling the numerical synchrotron core.
- Comparison tests between the analytical one-zone and numerical synchrotron SED pipelines.
- Gallery example `plot_electron_distributions.py` demonstrating the new electron distribution class API.
- Gallery example `plot_inhomogeneous_cylinder.py` demonstrating multi-zone inhomogeneous synchrotron SEDs.
- Reference documentation page `synchrotron_electron_distributions.rst` covering the class-based electron distribution system.
- Overhauled data gallery with dedicated examples for radio photometry, optical photometry, radio light curves, optical light curves, multi-band radio/optical data, and inference data containers.

### Changed
- `trilobite/radiation/synchrotron/SEDs/numerical.py` refactored into a `numerical/` subpackage (`core.py`, `one_zone.py`, `aspherical.py`, `inhomogeneous.py`); existing imports via `trilobite.radiation.synchrotron.SEDs` remain unchanged.
- `trilobite/radiation/synchrotron/SEDs/one_zone.py` refactored into a `one_zone/` subpackage (`seds.py`, `closure.py`, `_closure.py`, `_functions.py`, `_normalization.py`, `_ssa.py`); public API unchanged.
- `trilobite/radiation/synchrotron/microphysics.py` superseded by `electron_distributions.py`; the module is retained for backwards compatibility but its contents now re-export from the new class-based API.
- Pitch-angle averaging applied consistently across the synchrotron cooling pipeline in `cooling.py`.
- Switched readme to Markdown for better PyPI rendering.

### Fixed
- Pitch-angle averaging was incorrectly applied in `core.py`; the isotropic average is now used throughout the numerical synchrotron computations.
- SED normalization error in the analytical one-zone models.
- Default model parameters in `trilobite/models/SEDs/synchrotron.py` now guarantee `gamma_c > 1` to avoid unphysical cooling break positions.
- Spurious warnings and incorrect guard conditions in `cooling.py`.
- Corrected TREX footer attribution name in documentation.

---

## [0.0.1] — Initial Release

### Added
- Core model ABC (`trilobite/models/core/base.py`): `Model`, `ModelParameter`, `ModelVariable`
- Synchrotron radiation: microphysics, SEDs, cooling, one-zone closure (`trilobite/radiation/synchrotron/`)
- Free-free (bremsstrahlung) emission and absorption (`trilobite/radiation/free_free/`)
- Blackbody radiation utilities (`trilobite/radiation/blackbody.py`)
- OPAL Rosseland mean opacity with Cython bilinear interpolator and bundled solar composition table
- Shock dynamics: Rankine-Hugoniot relations, shock engine, Chevalier self-similar SN shock (`trilobite/dynamics/`)
- Bayesian inference pipeline: `InferenceProblem`, prior distributions, parameter transforms, `EmceeSampler` (`trilobite/inference/`)
- Observational data containers: radio/optical photometry, light curves (`trilobite/data/`)
- Optical photometry utilities: `PhotometryFilter`, `FilterBundle`, AB/ST magnitudes, speclite integration
- Cosmology helpers and special relativity utilities (`trilobite/physics_utils/`)
- Runtime configuration via `trilobite/bin/config.yaml` with user override support
- GNU General Public License v3
