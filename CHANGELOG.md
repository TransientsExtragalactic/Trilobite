# Changelog

All notable changes to Trilobite are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- One-zone accretion disk ODE integrator with Cython backend (`trilobite/dynamics/accretion/one_zone/`)
- Bracket root-finder Cython utility (`trilobite/math_utils/_bracket_root_finder.pyx`)
- TOPS opacity table support (`trilobite/radiation/opacity/grey_opacity/_tops_table.pyx`)
- Additional Rosseland mean opacity laws: electron scattering, Kramers free-free, Kramers bound-free, combined Kramers+ES variants
- Synchrotron SED gallery documentation pages
- A changelog file (``CHANGELOG.md``) to track changes across versions.

### Changed
- Switched readme to Markdown for better PyPI rendering

### Fixed
- Corrected TREX footer attribution name in documentation

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
