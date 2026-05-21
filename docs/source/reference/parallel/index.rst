.. _parallel:

========================
Parallel Computing
========================

Trilobite provides a unified framework for parallel execution across several
backends. Parallelism appears at multiple levels of the library:

- **Inference** — evaluating log-posteriors for many walker positions simultaneously
  (e.g., during MCMC sampling with :mod:`emcee`).
- **Grid execution** — sweeping physical models over large parameter grids to build
  interpolation tables or survey parameter space.
- **Simulation batches** — running independent simulations in parallel for ensemble
  analyses (e.g., light-curve grids for TDE or GRB parameter studies).

All parallel execution in Trilobite is organized around a common :class:`~trilobite.parallel.base.Pool`
abstraction. Every backend — whether single-process, multiprocessing, or MPI — exposes
the same interface, making it straightforward to switch backends without modifying
application code.

.. note::

    MPI support requires the optional ``mpi4py`` dependency. Install it with::

        pip install trilobite[mpi]

    All other parallel backends are available without additional dependencies.

----

Choosing a Backend
------------------

The right backend depends on your compute environment and task structure:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Backend
     - Class
     - When to use
   * - ``"serial"``
     - :class:`~trilobite.parallel.base.SerialPool`
     - Debugging, single-core environments, testing.
   * - ``"mp"``
     - :class:`~trilobite.parallel.mp.LikelihoodMPPool`
     - Multi-core shared-memory machines (laptops, workstations). Best for
       MCMC inference on a single node.
   * - ``"mpi"``
     - :class:`~trilobite.parallel.mpi.LikelihoodMPIPool`
     - Multi-node HPC clusters. Requires ``mpi4py`` and launching with
       ``mpirun`` / ``srun``.

The :func:`~trilobite.parallel.factory.make_pool` factory function provides a
single entry point that selects the right backend by name:

.. code-block:: python

    from trilobite.parallel import make_pool

    pool = make_pool("mp", problem=my_problem, processes=8)

    with pool:
        result = sampler.run(n_steps=1000, pool=pool)

----

.. _parallel_pools:

Pool System
-----------

The core abstraction is the abstract base class :class:`~trilobite.parallel.base.Pool`.
All backends inherit from it and expose the same minimal interface:

.. code-block:: python

    pool.map(func, iterable)    # Distribute tasks, collect results
    pool.size                   # Number of worker execution units
    pool.close()                # Graceful shutdown
    pool.terminate()            # Hard kill (used on exception)

Pools are context managers, so resources are always released cleanly:

.. code-block:: python

    with make_pool("mp", problem=my_problem) as pool:
        result = sampler.run(n_steps=500, pool=pool)

See :ref:`parallel_pools_api` for full API documentation on each pool class.

----

.. _parallel_mpi_workflow:

MPI Workflow
------------

MPI execution requires all processes to start together (via ``mpirun`` or a cluster
job scheduler). The master-worker pattern used by :class:`~trilobite.parallel.mpi.LikelihoodMPIPool`
means that **rank 0** drives the computation while **all other ranks** block waiting
for tasks:

.. code-block:: python

    # Run with: mpirun -n 8 python my_script.py

    from trilobite.parallel import LikelihoodMPIPool

    with LikelihoodMPIPool(problem=my_problem) as pool:
        if pool.is_master():
            result = sampler.run(n_steps=1000, pool=pool)
        else:
            pool.wait()   # Workers block here until the master closes the pool

.. important::

    Every rank must construct the pool before the ``if pool.is_master():`` branch.
    The :meth:`~trilobite.parallel.mpi.LikelihoodMPIPool.__init__` broadcasts the
    serialized :class:`~trilobite.inference.problem.InferenceProblem` to all ranks
    using a collective ``MPI.Bcast``.

    Calling :meth:`~trilobite.parallel.mpi.LikelihoodMPIPool.map` from a non-master
    rank raises :exc:`RuntimeError`.

----

Contents
--------

.. toctree::
   :maxdepth: 1

   pools
