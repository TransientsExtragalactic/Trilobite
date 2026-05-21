.. _parallel_pools_api:

Pool API Reference
==================

.. contents:: Contents
   :local:
   :depth: 2

----

Abstract Base Class
-------------------

*Module:* :mod:`trilobite.parallel.base`

All parallel pools in Trilobite inherit from :class:`~trilobite.parallel.base.Pool`.
This class defines the minimal interface required by ``emcee`` and other callers, and
is intentionally backend-agnostic. Subclasses only need to implement :attr:`~trilobite.parallel.base.Pool.size`,
:meth:`~trilobite.parallel.base.Pool.map`, and :meth:`~trilobite.parallel.base.Pool.close`.

The :meth:`~trilobite.parallel.base.Pool.terminate` method defaults to calling
:meth:`~trilobite.parallel.base.Pool.close`, but backends with a hard-kill path
(such as :class:`~trilobite.parallel.mpi.LikelihoodMPIPool`) override it. The context
manager calls :meth:`~trilobite.parallel.base.Pool.terminate` on exception and
:meth:`~trilobite.parallel.base.Pool.close` on clean exit, ensuring resources are
always released correctly.

.. currentmodule:: trilobite.parallel.base

.. autosummary::
   :toctree: ../../_as_gen

   Pool
   SerialPool

----

.. _pool_serial:

Serial Pool
-----------

:class:`~trilobite.parallel.base.SerialPool` executes tasks sequentially in the calling
process using the built-in :func:`map`. It is the default fallback for:

- debugging and testing (exceptions propagate immediately, no subprocess overhead),
- environments where multiprocessing is undesirable or unavailable,
- short runs where parallelism overhead would exceed any benefit.

.. code-block:: python

    from trilobite.parallel import SerialPool

    pool = SerialPool()
    results = list(pool.map(my_func, my_tasks))

:class:`~trilobite.parallel.base.SerialPool` always has ``size == 1``.

----

.. _pool_mp:

Multiprocessing Pool
---------------------

*Module:* :mod:`trilobite.parallel.mp`

:class:`~trilobite.parallel.mp.LikelihoodMPPool` distributes :class:`~trilobite.inference.problem.InferenceProblem`
likelihood evaluations across a pool of worker processes using Python's standard
:mod:`multiprocessing` library.

Design
^^^^^^

Directly pickling a bound method such as
``InferenceProblem._log_free_posterior`` is fragile under the ``spawn`` start
method (the default on macOS and Windows). To avoid this:

1. The :class:`~trilobite.inference.problem.InferenceProblem` is serialized to JSON
   **once** in the parent process.
2. Each worker reconstructs it **exactly once** via a pool initializer and stores it in a
   worker-global variable.
3. Only lightweight parameter vectors (``theta``) travel over the process boundary
   per task.

This eliminates repeated serialization overhead and is robust to the ``spawn``
start method.

Usage
^^^^^

.. code-block:: python

    from trilobite.parallel import LikelihoodMPPool

    with LikelihoodMPPool(problem=my_problem, processes=8) as pool:
        result = sampler.run(n_steps=1000, pool=pool)

.. hint::

    ``processes`` defaults to :func:`multiprocessing.cpu_count` if not specified.
    For MCMC, a good starting point is the number of physical cores on your machine.

Interrupt handling
^^^^^^^^^^^^^^^^^^

Workers ignore ``SIGINT`` so that ``Ctrl+C`` is caught cleanly by the parent.
The parent polls asynchronous results with a timeout (``wait_timeout``, default 3600 s)
so ``KeyboardInterrupt`` is never swallowed by a blocking wait.

.. currentmodule:: trilobite.parallel.mp

.. autosummary::
   :toctree: ../../_as_gen

   LikelihoodMPPool

----

.. _pool_mpi:

MPI Pool
--------

*Module:* :mod:`trilobite.parallel.mpi`

:class:`~trilobite.parallel.mpi.LikelihoodMPIPool` distributes likelihood evaluations
across MPI ranks using the ``mpi4py`` library. It is intended for multi-node HPC clusters
where shared-memory multiprocessing is insufficient.

.. important::

    ``mpi4py`` is an optional dependency. Importing this module is always safe;
    however, instantiating :class:`~trilobite.parallel.mpi.LikelihoodMPIPool`
    raises :exc:`RuntimeError` if ``mpi4py`` is not installed. Install it with::

        pip install trilobite[mpi]

Design
^^^^^^

The pool uses a **master-worker pattern**:

- **Rank 0** (master) runs the sampler and calls :meth:`~trilobite.parallel.mpi.LikelihoodMPIPool.map`.
- **Ranks 1 … N-1** (workers) block inside :meth:`~trilobite.parallel.mpi.LikelihoodMPIPool.wait`,
  processing tasks as they arrive.

During initialization, the problem JSON is broadcast from rank 0 using ``MPI.Bcast``,
so every rank reconstructs the :class:`~trilobite.inference.problem.InferenceProblem`
exactly once. Only theta vectors travel over MPI at evaluation time.

Task dispatch is *dynamic*: each worker is seeded with one task at start-up, and as
results arrive the newly-free worker is sent the next pending task. This keeps all
workers busy even when evaluation times vary.

MPI tag protocol
^^^^^^^^^^^^^^^^

Point-to-point messages use three tags:

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Tag
     - Value
     - Meaning
   * - ``_TAG_TASK``
     - 1
     - Master → worker: ``(task_index, theta)``
   * - ``_TAG_RESULT``
     - 2
     - Worker → master: ``(task_index, result)``
   * - ``_TAG_STOP``
     - 3
     - Master → worker: ``None`` — exit :meth:`~trilobite.parallel.mpi.LikelihoodMPIPool.wait`

Usage
^^^^^

Launch your script with ``mpirun`` (or ``srun`` on SLURM clusters):

.. code-block:: bash

    mpirun -n 8 python my_mcmc_script.py

Inside the script:

.. code-block:: python

    from trilobite.parallel import LikelihoodMPIPool

    with LikelihoodMPIPool(problem=my_problem) as pool:
        if pool.is_master():
            result = sampler.run(n_steps=1000, pool=pool)
        else:
            pool.wait()

.. hint::

    The :func:`~trilobite.parallel.factory.make_pool` factory provides a consistent
    entry point that works across all backends:

    .. code-block:: python

        from trilobite.parallel import make_pool

        with make_pool("mpi", problem=my_problem) as pool:
            if pool.is_master():
                result = sampler.run(n_steps=1000, pool=pool)
            else:
                pool.wait()

.. currentmodule:: trilobite.parallel.mpi

.. autosummary::
   :toctree: ../../_as_gen

   LikelihoodMPIPool

----

Factory
-------

*Module:* :mod:`trilobite.parallel.factory`

:func:`~trilobite.parallel.factory.make_pool` is the recommended entry point for
constructing pools. It accepts a backend name and forwards all keyword arguments to
the appropriate pool constructor, keeping backend-specific imports lazy:

.. code-block:: python

    from trilobite.parallel import make_pool

    # Serial (no extra arguments)
    pool = make_pool("serial")

    # Multiprocessing
    pool = make_pool("mp", problem=my_problem, processes=4)

    # MPI
    pool = make_pool("mpi", problem=my_problem)

.. currentmodule:: trilobite.parallel.factory

.. autosummary::
   :toctree: ../../_as_gen

   make_pool
