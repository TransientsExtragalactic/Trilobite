.. raw:: html

   <div class="hero">
       <img src="_images/logo.svg" class="hero-logo">
       <div class="hero-text">
           <h1>TRILOBITE</h1>
           <p> A python ecosystem for rapid-deployment modeling of astrophysical transients</p>
           <div class="badges">

|PYPI| |PYPI-PYTHON| |RUFF| |PRE-COMMIT| |NUMPYDOC| |COMMITIZEN| |CONVENTIONAL-COMMITS| |LAST-COMMIT| |CONTRIBUTORS| |DOCS| |ASTROPY|

.. raw:: html

           </div>
       </div>
   </div>

Overview
=========

The **TR**\ ansient **I**\ nference **L**\ ibrary for **O**\ bservation, **B**\ ayesian **I**\ nference, and **T**\ ime-domain **E**\ xploration (**TRILOBITE**) is a powerful, modular computational
library for modeling the interaction of transient astrophysical outflows with their surrounding environments.

.. grid:: 2
   :gutter: 3

   .. grid-item-card::
      :class-card: sd-shadow-sm sd-border-1

      **What does it do?**

      - Provides **customizable models** of astrophysical transients (GRBs, TDEs, SNe, etc.)
      - **Modular physics** components (dynamics, radiation, opacity, etc.) for generating new transient models.
      - **Flexible inference tools** for fitting models to observational data and constraining physical parameters.
      - Seamless **integration with external analysis tools** and libraries.

   .. grid-item-card::
      :class-card: sd-shadow-sm sd-border-1

      **Core capabilities**

      - Fast, Efficient, and Detailed **radiative processes** and **dynamics engines** for modeling emission from shocks,
        disks, ejecta, and more.
      - Flexible model construction tools for building **custom transient models** from modular physics components.
      - Comprehensive inference pipelines for fitting models to data and **performing parameter estimation**.
      - **Extensive documentation**, examples, and user guides to help you get started and make the most of the library.

.. grid:: 2
   :gutter: 3

   .. grid-item-card::
      :class-card: sd-shadow-sm sd-border-1

      **Why use Trilobite?**

      TRILOBITE bridges the gap between theory and observation by combining shock dynamics,
      microphysical prescriptions, and radiative transfer into a unified framework.

   .. grid-item-card::
      :class-card: sd-shadow-sm sd-border-1

      **Who is it for?**

      - Researchers modeling transient shock physics
      - Users building custom astrophysical models
      - Developers extending simulation frameworks

.. container:: install-block

    .. card::
       :class-card: sd-shadow-md sd-border-primary sd-p-3 install-card

       **Get started in seconds!**

       .. code-block:: bash

          pip install trilobite

       Or install from source:

       .. code-block:: bash

          git clone https://github.com/TransientsExtragalactic/Trilobite
          cd Trilobite && pip install -e .

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
        New to ``Trilobite``? The quickstart guide is the best place to start learning to use all of the
        tools that we have to offer!

        +++

        .. button-ref:: getting_started
            :ref-type: doc
            :expand:
            :color: secondary
            :click-parent:

            To The Quickstart Page

    .. grid-item-card::
        :img-top: images/index/lightbulb.png

        Examples
        ^^^^^^^^
        Have some basic experience with ``Trilobite`` but want to see a guide on how to execute a particular task? Need
        to find some code to copy and paste? The examples page contains a wide variety of use case examples and explanations
        for all of the various parts of the ``Trilobite`` library.

        +++

        .. button-ref:: examples
            :expand:
            :color: secondary
            :click-parent:

            To the Examples Page

    .. grid-item-card::
        :img-top: images/index/book.png

        User References
        ^^^^^^^^^^^^^^^^
        The user guide contains comprehensive, text-based explanations of the backbone components of the ``Trilobite`` library.
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
            :ref-type: doc
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
   examples
   getting_started

Indices and tables
==================

.. raw:: html

   <hr style="height:10px;background-color:black">


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. raw:: html

   <div class="affiliation-footer">
     <p class="affiliation-label">Developed at</p>
     <div class="affiliation-logos">
       <a href="https://www.berkeley.edu" target="_blank" rel="noopener noreferrer">
         <img src="_static/berkeley_logo.svg"
              alt="University of California, Berkeley"
              class="affiliation-logo">
       </a>
       <img src="_static/trex_logo.png"
            alt="Extragalactic Transients Group (TREX)"
            class="affiliation-logo affiliation-logo-trex">
     </div>
     <p class="affiliation-text">
       TRILOBITE is developed and maintained by Eliza Diggins and the
       <strong>Extragalactic Transients Group (TREX)</strong> in the Department
       of Astronomy at the <strong>University of California, Berkeley</strong>.
       Available under the GNU AGPLv3 license.
     </p>
   </div>


.. |PYPI| image:: https://img.shields.io/pypi/v/trilobite
   :target: https://pypi.org/project/trilobite/
   :alt: PyPI version

.. |PYPI-PYTHON| image:: https://img.shields.io/pypi/pyversions/trilobite
   :target: https://pypi.org/project/trilobite/
   :alt: Supported Python versions

.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
   :target: https://transientsextragalactic.github.io/Trilobite
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

.. |contributors| image:: https://img.shields.io/github/contributors/TransientsExtragalactic/Trilobite
   :target: https://github.com/TransientsExtragalactic/Trilobite/graphs/contributors
   :alt: GitHub Contributors

.. |last-commit| image:: https://img.shields.io/github/last-commit/TransientsExtragalactic/Trilobite
   :target: https://github.com/TransientsExtragalactic/Trilobite
   :alt: Last Commit

.. |ASTROPY| image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
   :target: http://www.astropy.org/
   :alt: Powered by Astropy
