.. _getting_started:

.. raw:: html

   <style>
     .bd-article h1 {
         font-size: 2.6rem;
         color: #1f7a8c;
         margin-bottom: 0.5rem;
     }
   </style>

   <div style="
    display: flex;
    align-items: center;
    gap: 2.5rem;
    padding: 2.5rem 0 2.4rem 0;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 1.8rem;
    ">

       <img src="_images/logo.svg" style="width: 25%; height: auto;">

       <div style="max-width: 620px;">

==================
Quickstart Guide
==================

.. raw:: html

           <p style="
               margin: 0.4rem 0 0 0;
               font-size: 1rem;
               color: #666;
           ">
               Get up and running with Trilobite.
           </p>

           <p style="
               margin: 1rem 0 0 0;
               font-size: 1.1rem;
               line-height: 1.6;
               color: #3a4a52;
           ">
               Install the package, explore worked examples, and understand
               how the library is structured — everything you need to start
               modeling astrophysical transients.
           </p>

       </div>
   </div>

Installation
============

.. tab-set::

   .. tab-item:: PyPI (Recommended)
      :sync: pypi

      Install the latest stable release directly from PyPI:

      .. code-block:: bash

         pip install trilobite

      To include optional support for optical photometry (requires ``speclite``):

      .. code-block:: bash

         pip install trilobite[optical]

   .. tab-item:: From Source
      :sync: source

      Clone the repository and install in editable mode to work with the latest
      development code or contribute to the project:

      .. code-block:: bash

         git clone https://github.com/TransientsExtragalactic/Trilobite
         cd Trilobite
         pip install -e .

      For the full development environment (testing, linting, docs):

      .. code-block:: bash

         pip install -e ".[dev]"

----

Where to Go Next
================

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: User Guide
      :link: user_guide
      :link-type: ref

      **The primary reference for using Trilobite.**

      Covers data handling, model construction, inference pipelines, and the
      physics building blocks — from shocks and synchrotron emission to
      accretion disks.

   .. grid-item-card:: Example Gallery
      :link: gallery_examples
      :link-type: ref

      **Runnable, self-contained worked examples.**

      Explore end-to-end workflows: loading data, fitting shock models,
      computing synchrotron SEDs, and running MCMC inference.

   .. grid-item-card:: API Reference
      :link: api
      :link-type: ref

      **Complete call signatures and module documentation.**

      Auto-generated reference for every public class, function, and
      parameter in the library.

   .. grid-item-card:: Developer Guide
      :link: developer_guide
      :link-type: ref

      **Contribute to Trilobite.**

      Coding standards, testing conventions, the documentation workflow,
      and guidance for extending the physics building blocks.

----

Getting Help
============

If you encounter a bug, have a question, or want to suggest an improvement,
open an issue on the
`Trilobite GitHub <https://github.com/TransientsExtragalactic/Trilobite/issues>`__.
Contributions are welcome at all skill levels.
