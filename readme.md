<p align="center">
  <img src="https://raw.githubusercontent.com/TransientsExtragalactic/Trilobite/main/docs/source/images/logo.svg" width="260" alt="Trilobite">
</p>

<h1 align="center">TRILOBITE</h1>

<p align="center"><em>A Python ecosystem for rapid-deployment modeling of astrophysical transients</em></p>

<p align="center">

[![PyPI](https://img.shields.io/pypi/v/trilobite)](https://pypi.org/project/trilobite/)
[![Python](https://img.shields.io/pypi/pyversions/trilobite)](https://pypi.org/project/trilobite/)
[![Tests](https://github.com/TransientsExtragalactic/Trilobite/actions/workflows/run_tests.yml/badge.svg)](https://github.com/TransientsExtragalactic/Trilobite/actions/workflows/run_tests.yml)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://transientsextragalactic.github.io/Trilobite)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://www.conventionalcommits.org/en/v1.0.0/)
[![Powered by Astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)
[![Contributors](https://img.shields.io/github/contributors/TransientsExtragalactic/Trilobite?cacheSeconds=86400)](https://github.com/TransientsExtragalactic/Trilobite/graphs/contributors)
[![Last Commit](https://img.shields.io/github/last-commit/TransientsExtragalactic/Trilobite?cacheSeconds=86400)](https://github.com/TransientsExtragalactic/Trilobite)

</p>

---

The **TR**ansient **I**nference **L**ibrary for **O**bservation, **B**ayesian **I**nference, and **T**ime-domain **E**xploration (**TRILOBITE**) is a powerful, modular computational library for modeling the interaction of transient astrophysical outflows with their surrounding environments.

TRILOBITE bridges the gap between theory and observation by combining shock dynamics, microphysical prescriptions, and radiative transfer into a unified framework — enabling rapid-deployment modeling of **GRBs, TDEs, supernovae**, and other astrophysical transients.

---

## Overview

**What can it do?**

- **Customizable transient models** — Pre-built models for GRBs, TDEs, and supernovae, fully configurable for your science case.
- **Modular physics components** — Mix-and-match dynamics, radiation, and opacity modules to build novel transient models from scratch.
- **Flexible Bayesian inference** — End-to-end parameter estimation pipelines for fitting models to radio and multi-wavelength data.
- **Ecosystem integration** — Native compatibility with `astropy`, `emcee`, and modern radio data formats.

**Who is it for?**

- Researchers modeling transient shock physics and radio emission
- Scientists building custom astrophysical forward models
- Observers looking to constrain explosion parameters from data

---

## Quick Install

Install from PyPI:

```bash
pip install trilobite
```

Install from source:

```bash
git clone https://github.com/TransientsExtragalactic/Trilobite
cd Trilobite
pip install -e .
```

For full setup instructions, see the [Installation Guide](https://transientsextragalactic.github.io/Trilobite/getting_started/installation.html).

---

## Documentation

The full documentation — user guides, API references, worked examples, and theory background — is available at:

https://transientsextragalactic.github.io/Trilobite

---

## Contributing

Development takes place on [GitHub](https://github.com/TransientsExtragalactic/Trilobite).

Bug reports, feature requests, and documentation improvements are welcome via the issue tracker. Contributions should follow the established coding, documentation, and [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) style guidelines.

---

## Citation

If you use Trilobite in your research, please acknowledge it with:

> *We acknowledge the use of the Trilobite software package, written by E. Diggins et al., in this work.*

TRILOBITE is developed and maintained by Eliza Diggins and the **Extragalactic Transients Group (TREX)** in the Department of Astronomy at the **University of California, Berkeley**.
