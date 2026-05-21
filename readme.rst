.. image:: docs/source/images/logo.png
   :width: 260px
   :align: center

|

|Tests| |docs| |docs-stable| |RUFF| |PRE-COMMIT| |ISORT| |NUMPYDOC| |DOCFORMATTER| |COMMITIZEN| |CONVENTIONAL-COMMITS| |ASTROPY| |CONTRIBUTORS| |LAST-COMMIT|

----

The **TR**\ansient **I**\nteraction **C**\ode for **E**\jecta, **R**\emnants, **A**\nd **T**\ime-domain
**O**\bservables from **P**\lasma **S**\hocks (**TRICERATOPS**) is a powerful, modular computational
library for modeling the interaction of transient astrophysical outflows with their surrounding environments.

TRICERATOPS bridges the gap between theory and observation by combining shock dynamics, microphysical
prescriptions, and radiative transfer into a unified framework — enabling rapid-deployment modeling of
**GRBs, TDEs, supernovae**, and other astrophysical transients.


Overview
========

**What can it do?**

- **Customizable transient models** — Pre-built models for GRBs, TDEs, and supernovae, fully configurable for your science case.
- **Modular physics components** — Mix-and-match dynamics, radiation, and opacity modules to build novel transient models from scratch.
- **Flexible Bayesian inference** — End-to-end parameter estimation pipelines for fitting models to radio and multi-wavelength data.
- **Ecosystem integration** — Native compatibility with ``astropy``, ``emcee``, and modern radio data formats.

**Who is it for?**

- Researchers modeling transient shock physics and radio emission
- Scientists building custom astrophysical forward models
- Observers looking to constrain explosion parameters from data

----

Quick Install
=============

.. code-block:: bash

    pip install triceratops-lib

Or install from source:

.. code-block:: bash

    git clone https://github.com/TransientsExtragalactic/Triceratops
    cd Triceratops
    pip install -e .

For full setup instructions, see the
`Installation Guide <https://transientsextragalactic.github.io/Triceratops/getting_started/installation.html>`_.

----

Documentation
=============

The full documentation — user guides, API references, worked examples, and theory background — is available at:

    `https://transientsextragalactic.github.io/Triceratops <https://transientsextragalactic.github.io/Triceratops>`_

----

Contributing
============

Development takes place on
`GitHub <https://github.com/TransientsExtragalactic/Triceratops>`_.

Bug reports, feature requests, and documentation improvements are welcome via the issue tracker.
Contributions should follow the established coding, documentation, and
`Conventional Commits <https://www.conventionalcommits.org/en/v1.0.0/>`_ style guidelines.

----

Citation
========

If you use Triceratops in your research, please acknowledge it with:

    *We acknowledge the use of the Triceratops software package, written by E. Diggins et al., in this work.*

TRICERATOPS is developed and maintained by Eliza Diggins and the
**Extragalactic Transients Group (TREX)** in the Department of Astronomy at the
**University of California, Berkeley**.

----

.. |Tests| image:: https://github.com/TransientsExtragalactic/Triceratops/actions/workflows/run_tests.yml/badge.svg
   :target: https://github.com/TransientsExtragalactic/Triceratops/actions/workflows/run_tests.yml
   :alt: Tests

.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
   :target: https://transientsextragalactic.github.io/Triceratops
   :alt: Latest Docs

.. |docs-stable| image:: https://img.shields.io/badge/docs-stable-brightgreen.svg
   :target: https://transientsextragalactic.github.io/Triceratops/stable
   :alt: Stable Docs

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
   :alt: Ruff

.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit

.. |numpydoc| image:: https://img.shields.io/badge/docstyle-numpydoc-459db9
   :target: https://numpydoc.readthedocs.io/en/latest/
   :alt: Docstring style: numpydoc

.. |commitizen| image:: https://img.shields.io/badge/commitizen-friendly-brightgreen.svg
   :target: https://commitizen-tools.github.io/commitizen/
   :alt: Commit style: Conventional + Gitmoji

.. |conventional-commits| image:: https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white
   :target: https://www.conventionalcommits.org/en/v1.0.0/
   :alt: Commit style: Conventional Commits

.. |contributors| image:: https://img.shields.io/github/contributors/TransientsExtragalactic/Triceratops
   :target: https://github.com/TransientsExtragalactic/Triceratops/graphs/contributors
   :alt: GitHub Contributors

.. |last-commit| image:: https://img.shields.io/github/last-commit/TransientsExtragalactic/Triceratops
   :target: https://github.com/TransientsExtragalactic/Triceratops
   :alt: Last Commit

.. |ASTROPY| image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
   :target: http://www.astropy.org/
   :alt: Powered by Astropy

.. |docformatter| image:: https://img.shields.io/badge/%20formatter-docformatter-fedcba
   :target: https://github.com/PyCQA/docformatter
   :alt: docformatter

.. |isort| image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
   :target: https://pycqa.github.io/isort/
   :alt: isort
