# Changelog

All notable changes to Trilobite are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Gallery example `plot_slsn_magnetar_swift.py` (`docs/source/galleries/inference/`) — fits `MagnetarArnettModel` to the Swift/UVOT photometry of SN2020znr (Hinkle et al. 2023) with `emcee` and compares the posterior with the published MOSFiT fit; the photometry is in `docs/source/_data/sn2020znr_swift_uvot.csv`.
- `NickelCobaltArnettModel`, `MagnetarArnettModel` and `MagnetarNickelArnettModel` (`trilobite/models/supernovae/arnett.py`) — optical photometry models built on the Arnett diffusion engines: the engine luminosity heats a blackbody photosphere (ejecta surface with a temperature floor) whose spectrum is convolved with a `FilterBundle`, with observer-frame times and full redshift treatment. They share the `ArnettOpticalModel` base, which also exposes `compute_photosphere` and can be subclassed with any `ArnettEngine`.
- `MCMCSamplingResult.physical_samples` and a `physical` option (default `True`) on `get_flat_samples`, `trace_plot` and `corner_plot` (`trilobite/inference/sampling/result.py`) — map the chain back from sampling space to physical space for parameters with a transform; `physical=False` returns the raw sampling-space chain.
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
- `Likelihood.config` and `get_likelihood_config` (`trilobite/inference/likelihood/`) — the construction options of a likelihood (e.g. `sys_dex`), which `InferenceProblem.to_dict` now stores under a new optional `likelihood_kwargs` key. Specifications written without the key still load.

### Changed
- `MCMCSamplingResult.get_flat_samples`, `trace_plot` and `corner_plot` now return/plot **physical-space** values by default; previously a transformed parameter (e.g. `Log10Transform`) was returned as its raw sampling-space value under the parameter's own name. Pass `physical=False` for the old behaviour. `.samples`, the HDF5 format and the convergence diagnostics (autocorrelation time, ESS, MCSE, R-hat) remain in sampling space. Nothing changes for problems without transforms.
- `InferenceProblem.__call__`, `_log_free_posterior` and `initial_log_posterior` take the free parameters in **sampling space** (identical to physical space without transforms), and `EmceeSampler` starts its walkers at `initial_z` accordingly. `unpack_parameters` is now the exact inverse of `pack_parameters` and no longer applies transforms.
- `Diggins26_TwoComponentModel` evaluates roughly 4x faster (about 85 ms to 20 ms per likelihood evaluation on the AT2018cow photometry): the shock integration defaults to `LSODA` at `rtol=1e-7` (within ~1e-3 in log-likelihood of a tight `Radau` reference), and the electron microphysics and synchrotron spectra are evaluated in batches over epochs instead of one epoch at a time.
- `Diggins26_TwoComponentModel` takes the CSM composition (`hydrogen_mass_fraction`, `helium_mass_fraction`) as a hyperparameter, deriving the mean molecular weights used by the shock and the free-free emission measure; `integrator_kw`, `adiabatic_index` and an optional `seed_photon_luminosity` (inverse-Compton cooling) are also constructor options.
- `BrokenPowerLawEjectaProfile` and `AxisymmetricCSMDensityProfile` evaluate scalar arguments of `as_optimized_callable` with scalar arithmetic (about 20x faster per call, identical to the vectorized path to round-off); array arguments are unchanged.
- The finite-`v_max` normalization of the broken power-law ejecta is evaluated in one vectorized pass (about 60x faster, bit-identical results).
- The numerical synchrotron engine's flux, intensity and emissivity/absorption methods accept `paired=True` to evaluate each zone at its own frequency instead of forming the full frequency-by-zone product.
- `trilobite/radiation/synchrotron/SEDs/numerical.py` refactored into a `numerical/` subpackage (`core.py`, `one_zone.py`, `aspherical.py`, `inhomogeneous.py`); existing imports via `trilobite.radiation.synchrotron.SEDs` remain unchanged.
- `trilobite/radiation/synchrotron/SEDs/one_zone.py` refactored into a `one_zone/` subpackage (`seds.py`, `closure.py`, `_closure.py`, `_functions.py`, `_normalization.py`, `_ssa.py`); public API unchanged.
- `trilobite/radiation/synchrotron/microphysics.py` superseded by `electron_distributions.py`; the module is retained for backwards compatibility but its contents now re-export from the new class-based API.
- Pitch-angle averaging applied consistently across the synchrotron cooling pipeline in `cooling.py`.
- Switched readme to Markdown for better PyPI rendering.

### Fixed
- `to_inference_data` of `OpticalPhotometryContainer`, `OpticalPhotometryEpoch` and `OpticalLightCurveContainer` keyed the band indices as `band_idx`, so it could not feed any `OpticalModel` (which declares `band_index`) and raised a variable-name error. The containers now use whichever name the model declares (`trilobite.data.utils.band_variable_name`).
- `InferenceProblem._log_posterior` applied parameter inverse transforms twice, so the likelihood of a transformed parameter was evaluated at e.g. `10**(10**z)` instead of `10**z` (the prior saw the correct value). Frozen parameters, which are stored in physical space, were also inverted as if they were sampling-space values and contributed a spurious Jacobian term, and `InferenceProblem.__call__` raised a length mismatch whenever a parameter was frozen. Problems without transforms were unaffected.
- `Diggins26_TwoComponentModel` now returns NaN flux (and `diagnose()` reports why) when the ejecta cannot be normalized for a `v_0` that is below `max_initial_beta` but above what the kernel truncated at `c` can represent; previously the `RuntimeError` from the normalization escaped and aborted the run.
- Pitch-angle averaging was incorrectly applied in `core.py`; the isotropic average is now used throughout the numerical synchrotron computations.
- SED normalization error in the analytical one-zone models.
- Default model parameters in `trilobite/models/SEDs/synchrotron.py` now guarantee `gamma_c > 1` to avoid unphysical cooling break positions.
- Spurious warnings and incorrect guard conditions in `cooling.py`.
- Corrected TREX footer attribution name in documentation.
- `InferenceProblem.to_dict` / `to_json` stored the likelihood only as its class path, so `from_dict` / `from_json` rebuilt it with default options (e.g. `LogGaussianCensoredLikelihood(sys_dex=0.1)` came back with `sys_dex=0`). `LikelihoodMPPool` and `LikelihoodMPIPool` rebuild the problem in each worker this way, so pool runs silently sampled a different posterior than serial runs. Likelihood options are now serialized, and an option that cannot be represented in JSON raises `TypeError` instead of being dropped.

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
