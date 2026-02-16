.. image:: ./images/logo.png
   :width: 200px
   :align: center


Triceratops
============

|RUFF| |PRE-COMMIT| |NUMPYDOC| |COMMITIZEN| |CONVENTIONAL-COMMITS| |LAST-COMMIT| |CONTRIBUTORS| |DOCS| |ASTROPY|

The **TR**\ ansient **I**\ nteraction **C**\ ode for **E**\ jecta, **R**\ emnants, **A**\ and
**T**\ ime-domain **O**\ bservables from **P**\ lasma **S**\ hocks (**TRICERATOPS**) is a powerful, modular computational
library for modeling the interaction of transient astrophysical outflows with their surrounding environments.

TRICERATOPS is designed to simulate the coupled dynamics, radiation processes, and observational signatures that arise
when ejecta-driven shocks propagate through ambient media. It enables researchers to model plasma shocks, ejecta
dynamics, and broadband time-domain observables in a physically self-consistent way, with particular emphasis on radio
and millimeter transients.

The library supports both **forward modeling** and **parameter inference**, allowing users to connect physically
motivated shock models directly to observational data from modern radio telescopes. By combining shock dynamics,
microphysical prescriptions, and radiative transfer into a unified framework, TRICERATOPS helps bridge the gap between
theory and observation in time-domain astrophysics.

Built with flexibility and extensibility in mind, TRICERATOPS is suitable for a wide range of users; from newcomers
exploring transient shock physics to experienced researchers developing custom models or extending the codebase.
Its modular architecture makes it straightforward to swap in alternative physical prescriptions, add new radiation
processes, or integrate with external inference and analysis tools.


.. raw:: html

   <hr style="color:black">

Resources
=========

.. grid:: 2
    :padding: 3
    :gutter: 5

    .. grid-item-card::
        :img-top: images/index/stopwatch_icon.png

        Quickstart Guide
        ^^^^^^^^^^^^^^^^
        New to ``Triceratops``? The quickstart guide is the best place to start learning to use all of the
        tools that we have to offer!

        +++

        .. button-ref:: getting_started
            :expand:
            :color: secondary
            :click-parent:

            To The Quickstart Page

    .. grid-item-card::
        :img-top: images/index/lightbulb.png

        Examples
        ^^^^^^^^
        Have some basic experience with ``Triceratops`` but want to see a guide on how to execute a particular task? Need
        to find some code to copy and paste? The examples page contains a wide variety of use case examples and explanations
        for all of the various parts of the ``Triceratops`` library.

        +++

        .. button-ref:: auto_examples/index
            :expand:
            :color: secondary
            :click-parent:

            To the Examples Page

    .. grid-item-card::
        :img-top: images/index/book.png

        User References
        ^^^^^^^^^^^^^^^^
        The user guide contains comprehensive, text-based explanations of the backbone components of the ``Triceratops`` library.
        If you're looking for information on the underlying code or for more details on particular aspects of the API, this is your best resource.

        +++

        .. button-ref:: reference/user_guide
            :expand:
            :color: secondary
            :click-parent:

            To the User Guide

    .. grid-item-card::
        :img-top: images/index/api_icon.png

        API Reference
        ^^^^^^^^^^^^^

        Doing a deep dive into our code? Looking to contribute to development? The API reference is a comprehensive resource
        complete with source code and type hinting so that you can find every detail you might need.

        +++

        .. button-ref:: api
            :expand:
            :color: secondary
            :click-parent:

            API Reference

Contents
========
.. raw:: html

   <hr style="height:10px;background-color:black">

.. toctree::
   :maxdepth: 1

   api
   reference/user_guide
   auto_examples/index
   getting_started

Indices and tables
==================

.. raw:: html

   <hr style="height:10px;background-color:black">


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
   :target: https://eliza-diggins.github.io/Triceratops
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

.. |contributors| image:: https://img.shields.io/github/contributors/eliza-diggins/Triceratops
   :target: https://github.com/eliza-diggins/Triceratops/graphs/contributors
   :alt: GitHub Contributors

.. |last-commit| image:: https://img.shields.io/github/last-commit/eliza-diggins/Triceratops
   :target: https://github.com/eliza-diggins/Triceratops
   :alt: Last Commit

.. |ASTROPY| image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
   :target: http://www.astropy.org/
   :alt: Powered by Astropy
