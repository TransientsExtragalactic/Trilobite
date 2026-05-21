.. image:: docs/source/images/logo.png
   :width: 200px
   :align: center

Triceratops
===========

+-------------------+----------------------------------------------------------+
| **Code**          | |RUFF| |PRE-COMMIT| |ASTROPY| |ISORT|                    |
+-------------------+----------------------------------------------------------+
| **Documentation** | |docs| |docs-stable| |numpydoc| |docformatter|           |
+-------------------+----------------------------------------------------------+
| **GitHub**        | |CONTRIBUTORS| |LAST-COMMIT| |Tests|                     |
|                   | |COMMITIZEN| |CONVENTIONAL-COMMITS|                      |
+-------------------+----------------------------------------------------------+
| **PyPi**          | Coming Soon                                              |
+-------------------+----------------------------------------------------------+

Welcome to **Triceratops** — the **T**\ransient **R**\adio **I**\nteraction **C**\ode for **E**\jecta,
**R**\emnants, **A**\nd **T**\ime-domain **O**\bservables from **P**\lasma **S**\hocks.

Triceratops is a computational framework for modeling the dynamical and radiative evolution of
astrophysical transients interacting with their environments. The package is designed to
simulate plasma shocks, ejecta dynamics, and their resulting time-domain observables, with a
particular focus on radio emission from transient outflows.

The software enables forward modeling and parameter inference for radio observations of
explosive and eruptive phenomena, allowing theoretical models to be directly confronted with
data from modern radio facilities. Triceratops emphasizes physical transparency, modular
design, and numerical robustness, making it suitable for both exploratory modeling and
production-level analysis.

Getting Started
---------------

To install the package using ``pip`` (not yet available on PyPI):

.. code-block:: bash

    pip install triceratops

Alternatively, you can install directly from the GitHub repository:

.. code-block:: bash

    git clone https://www.github.com/TransientsExtragalactic/Triceratops
    cd Triceratops
    pip install .

For detailed setup instructions, see the
`Installation Guide <https://transientsextragalactic.github.io/Triceratops/getting_started/installation.html>`_.

Documentation
-------------

The full documentation is available at
`https://transientsextragalactic.github.io/Triceratops <https://transientsextragalactic.github.io/Triceratops>`_.

This includes conceptual overviews, user guides, API references, and worked examples.

Contributing
------------

Development takes place on
`GitHub <https://www.github.com/TransientsExtragalactic/Triceratops>`_.

Bug reports, feature requests, and documentation improvements are welcome via the issue tracker.
Contributions should follow the established coding, documentation, and commit-style guidelines
indicated above.

Attribution
-----------

If you use Triceratops in your research, please cite or acknowledge it with text similar to:

    We acknowledge the use of the Triceratops software package, written by E. Diggins et al., in this work.


.. |Tests| image:: https://github.com/TransientsExtragalactic/Triceratops/actions/workflows/run_tests.yml/badge.svg

.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
   :target: https://transientsextragalactic.github.io/Triceratops
   :alt: Latest Docs

.. |docs-stable| image:: https://img.shields.io/badge/docs-stable-brightgreen.svg
   :target: https://transientsextragalactic.github.io/Triceratops/stable
   :alt: Latest Docs

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

.. |isort| image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
   :target: https://pycqa.github.io/isort/
