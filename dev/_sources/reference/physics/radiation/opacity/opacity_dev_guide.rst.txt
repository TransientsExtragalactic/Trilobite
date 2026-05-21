.. _opacity_dev_guide:

==========================================
Developer Guide: Implementing Opacity Laws
==========================================

This guide walks through adding a new opacity law to Triceratops.  It covers the
full path from the abstract base class to a compiled Cython extension type, including
registration in the string resolver and the test conventions you should follow.

For the physics of existing opacity laws see :ref:`opacity_theory`.
For user-facing usage see :ref:`opacity_user_guide`.

.. contents::
    :local:
    :depth: 2

----

Architecture Overview
----------------------

The opacity system is organized into subpackages by frequency-averaging type, with a two-level
Python class hierarchy:

.. code-block:: none

    triceratops.radiation.opacity/
    ├── base.py              OpacityLaw   (root ABC; mean_type = None by default)
    ├── grey_opacity/        frequency-averaged opacities
    │   ├── base.py          GreyOpacityLaw, ConstantGreyOpacity
    │   ├── rosseland/       Rosseland mean — all current implementations
    │   │   ├── models.py    ElectronScatteringOpacity, Kramers*, OPALOpacity
    │   │   └── _*.pyx       Cython C-level implementations
    │   └── planck/          Planck mean — stub for future work
    └── tables/              bundled opacity data (e.g. asplund_grevesse_05.h5)

**mean_type convention:** Each concrete class sets
:attr:`~triceratops.radiation.opacity.base.OpacityLaw.mean_type` to the string identifying
its frequency-averaging convention.  All current Rosseland mean laws set
``mean_type = "rosseland"``.  New Planck mean implementations would set
``mean_type = "planck"`` and live in :mod:`triceratops.radiation.opacity.grey_opacity.planck`.

**Key design invariants:**

1. All opacity state lives in the Python layer.  Cython objects are constructed once
   in ``__init__`` and are considered read-only after that.
2. The **public** methods (:meth:`~triceratops.radiation.opacity.base.OpacityLaw.opacity`,
   :meth:`~triceratops.radiation.opacity.base.OpacityLaw.dlogkappa_dlogrho`,
   :meth:`~triceratops.radiation.opacity.base.OpacityLaw.dlogkappa_dlogT`) are
   implemented once in :class:`~triceratops.radiation.opacity.base.OpacityLaw` and must
   **not** be overridden.
3. Subclasses override only the **private** log-space methods (``_log_opacity``,
   ``_dlogkappa_dlogrho``, ``_dlogkappa_dlogT``) **or** set ``IS_C_BACKED = True``
   and implement ``_initialize_C_object``.
4. C-backed subclasses must also keep ``_LOG_KAPPA`` in sync for introspection.

----

The Private Log-Space Interface
---------------------------------

Every opacity evaluation eventually calls one of three private methods:

.. code-block:: python

    def _log_opacity(self, log_T, log_rho):
        """Return ln(κ) given ln(T) [K] and ln(ρ) [g cm⁻³]."""

    def _dlogkappa_dlogrho(self, log_T, log_rho):
        """Return ∂ ln κ / ∂ ln ρ (dimensionless)."""

    def _dlogkappa_dlogT(self, log_T, log_rho):
        """Return ∂ ln κ / ∂ ln T (dimensionless)."""

All three receive and return plain ``float`` or ``numpy.ndarray`` values in natural-log
CGS units — **no** :class:`~astropy.units.Quantity` objects.  The public wrappers in
:class:`~triceratops.radiation.opacity.base.OpacityLaw` handle unit conversion before
calling these methods and re-attach units to the result.

.. important::

    Never override the public wrappers.  Any unit conversion mistake introduced in a
    subclass override could silently corrupt disk integrations, and the mistake would be
    difficult to trace.  Override only the private ``_log_*`` methods.

----

Route 1 — Pure-Python Opacity Law
-----------------------------------

A pure-Python opacity law is the simplest approach.  Override the three private
log-space methods directly.  No Cython is required.

Worked Example: Power-Law Opacity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose you want an opacity of the form
:math:`\kappa(\rho, T) = \kappa_0\,\rho^a\,T^b` (a generalized Kramers law with
user-specified exponents).

.. code-block:: python

    # triceratops/radiation/opacity/grey_opacity/rosseland/models.py  (append to file)

    import numpy as np
    from triceratops.radiation.opacity.grey_opacity.base import GreyOpacityLaw


    class PowerLawOpacity(GreyOpacityLaw):
        r"""User-defined power-law opacity :math:`\kappa = \kappa_0\,\rho^a\,T^b`.

        Parameters
        ----------
        kappa0 : float
            Normalisation in CGS units consistent with (a, b).
        a : float
            Power-law index on density (dimensionless).
        b : float
            Power-law index on temperature (dimensionless).

        Notes
        -----
        Setting ``a=1`` and ``b=-3.5`` recovers Kramers scaling.
        Setting ``a=0`` and ``b=0`` with ``kappa0=kappa_es`` gives a constant opacity.
        """

        IS_C_BACKED = False   # pure Python
        mean_type = "rosseland"  # set to "planck" for Planck mean, None for mean-type-agnostic

        def __init__(self, kappa0: float, a: float, b: float):
            self.kappa0 = kappa0
            self.a = a
            self.b = b
            # _LOG_KAPPA is not well-defined for a density/temperature-dependent law,
            # but set it to log(kappa0) for introspection purposes.
            self._LOG_KAPPA = np.log(float(kappa0))
            super().__init__()   # IS_C_BACKED=False, so no Cython object is built

        def _log_opacity(self, log_T, log_rho):
            # ln κ = ln κ₀ + a·ln ρ + b·ln T
            return self._LOG_KAPPA + self.a * log_rho + self.b * log_T

        def _dlogkappa_dlogrho(self, log_T, log_rho):
            return self.a   # constant power-law exponent

        def _dlogkappa_dlogT(self, log_T, log_rho):
            return self.b   # constant power-law exponent

The base class ``__init__`` accepts ``**parameters`` and passes them to
``_initialize_C_object`` if ``IS_C_BACKED`` is ``True``.  For a pure-Python class, call
``super().__init__()`` with no keyword arguments; the C companion will remain ``None``.

.. note::

    :class:`~triceratops.radiation.opacity.grey_opacity.base.GreyOpacityLaw` provides default
    implementations of all three private methods that return ``self._LOG_KAPPA`` (for
    ``_log_opacity``) and ``0.0`` (for the derivatives).  For a density- or
    temperature-dependent law you **must** override all three.  Only subclasses where
    :math:`\kappa` is truly constant (grey) can rely on the ``_LOG_KAPPA`` default.

----

Route 2 — C-Backed Opacity Law
---------------------------------

C-backed opacity laws pair a Python class with a compiled Cython extension type.  This
is the preferred approach for any law that will be called millions of times per integration
(e.g. from inside the one-zone disk explicit-Euler loop).

The C-backed path has four steps: write the Cython extension, write the Python class,
build the extension, and register it.

Step 1 — Write the Cython Extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new ``.pyx`` file in
``triceratops/radiation/opacity/grey_opacity/rosseland/``.  The extension type must inherit from the base
C class declared in ``opacity_base.pxd``:

.. code-block:: cython

    # triceratops/radiation/opacity/grey_opacity/rosseland/_my_opacity.pyx

    from triceratops.radiation.opacity.opacity_base cimport C_GreyOpacityBase

    cdef class C_MyOpacity(C_GreyOpacityBase):
        """Cython extension for a custom opacity law."""

        cdef double _kappa0
        cdef double _a
        cdef double _b

        def __cinit__(self, double kappa0, double a, double b):
            self._kappa0 = kappa0
            self._a = a
            self._b = b

        cpdef double log_opacity(self, double log_T, double log_rho):
            return log(self._kappa0) + self._a * log_rho + self._b * log_T

        cpdef double dlogkappa_dlogrho(self, double log_T, double log_rho):
            return self._a

        cpdef double dlogkappa_dlogT(self, double log_T, double log_rho):
            return self._b

        cpdef log_opacity_array(self, double[::1] log_T, double[::1] log_rho):
            import numpy as np
            cdef int n = log_T.shape[0]
            result = np.empty(n, dtype=np.float64)
            cdef double[::1] out = result
            cdef int i
            for i in range(n):
                out[i] = log(self._kappa0) + self._a * log_rho[i] + self._b * log_T[i]
            return result

        cpdef dlogkappa_dlogrho_array(self, double[::1] log_T, double[::1] log_rho):
            import numpy as np
            return np.full(log_T.shape[0], self._a, dtype=np.float64)

        cpdef dlogkappa_dlogT_array(self, double[::1] log_T, double[::1] log_rho):
            import numpy as np
            return np.full(log_T.shape[0], self._b, dtype=np.float64)

The six methods (scalar + array variants for each of the three quantities) are the
full interface required by :class:`~triceratops.radiation.opacity.grey_opacity.base.GreyOpacityLaw`.

.. important::

    Array methods (``*_array``) receive **typed memoryviews** (``double[::1]``), not
    numpy arrays.  Use ``np.ascontiguousarray(arr, dtype=np.float64)`` before calling
    them if your arrays might not be C-contiguous.  The Python-layer
    :class:`~triceratops.radiation.opacity.grey_opacity.base.GreyOpacityLaw` does this automatically.

Step 2 — Write the Python Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class MyOpacity(GreyOpacityLaw):
        r"""C-backed power-law opacity :math:`\kappa = \kappa_0\,\rho^a\,T^b`.

        Parameters
        ----------
        kappa0 : float
            Normalisation constant (CGS).
        a : float
            Power-law index on density.
        b : float
            Power-law index on temperature.
        """

        IS_C_BACKED = True

        def __init__(self, kappa0: float, a: float, b: float):
            self.kappa0 = kappa0
            self.a = a
            self.b = b
            self._LOG_KAPPA = np.log(float(kappa0))
            # super().__init__(**kwargs) calls _initialize_C_object(**kwargs)
            super().__init__(kappa0=kappa0, a=a, b=b)

        def _initialize_C_object(self, kappa0, a, b):
            from triceratops.radiation.opacity.grey_opacity.rosseland._my_opacity import C_MyOpacity
            return C_MyOpacity(kappa0=kappa0, a=a, b=b)

The keyword arguments passed to ``super().__init__(**kwargs)`` are forwarded verbatim to
``_initialize_C_object(**kwargs)``, so they must match its signature exactly.

Step 3 — Register the Extension with setuptools/Cython
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the new ``.pyx`` file to the list of Cython extension modules in
``setup.py`` (or ``pyproject.toml`` if using scikit-build-core):

.. code-block:: python

    # In setup.py or the equivalent build configuration
    Extension(
        "triceratops.radiation.opacity.grey_opacity.rosseland._my_opacity",
        sources=["triceratops/radiation/opacity/grey_opacity/rosseland/_my_opacity.pyx"],
        include_dirs=[np.get_include()],
    ),

Re-install the package in development mode to compile the extension:

.. code-block:: bash

    pip install -e ".[dev]"

----

Step 4 — Register in ``get_opacity()``
-----------------------------------------

To expose your new law via the string resolver, add an entry to ``_OPACITY_REGISTRY``
in ``triceratops/radiation/opacity/utils.py``:

.. code-block:: python

    # utils.py
    _OPACITY_REGISTRY: dict[str, str] = {
        # ... existing entries ...
        "my_opacity": "MyOpacity",   # ← add this line
    }

The value is the class name as it appears in the appropriate subpackage
(e.g. ``triceratops.radiation.opacity.grey_opacity.rosseland``).  The resolver imports
that class lazily (to avoid Cython import overhead at module load time) and calls it
with no arguments,
so the class **must** have sensible defaults for all constructor parameters if you want
to use the string interface without customization.

After registering, verify that the resolver works:

.. code-block:: python

    from triceratops.radiation.opacity.utils import get_opacity
    kap = get_opacity("my_opacity")
    print(type(kap))   # <class 'MyOpacity'>

----

Step 5 — Update the Public ``__init__.py``
--------------------------------------------

Export the new class through the ``__init__.py`` chain:

.. code-block:: python

    # triceratops/radiation/opacity/grey_opacity/rosseland/__init__.py
    from .models import MyOpacity  # add to the existing import block

    # triceratops/radiation/opacity/grey_opacity/__init__.py
    from .rosseland import MyOpacity  # re-export upward

    # triceratops/radiation/opacity/__init__.py
    from .grey_opacity import MyOpacity  # re-export at top level
    __all__ = [
        # ... existing ...
        "MyOpacity",
    ]

----

Step 6 — Write Tests
-----------------------

All opacity tests live in ``tests/test_radiation/test_opacity/``.  Add a new class (or
extend ``test_models.py``) using the conventions described in the existing test suite.

**Minimum test surface for any new opacity law:**

.. code-block:: python

    # tests/test_radiation/test_opacity/test_models.py  (extend existing file)

    class TestMyOpacity:
        """Tests for MyOpacity."""

        _rho = 1.0e-5 * u.g / u.cm**3
        _T   = 1.0e7  * u.K

        def test_is_grey_opacity_law(self):
            """MyOpacity inherits from GreyOpacityLaw."""
            assert isinstance(MyOpacity(1.0, 1.0, -3.5), GreyOpacityLaw)

        def test_opacity_formula(self):
            """opacity() matches κ₀ · ρ^a · T^b at the reference point."""
            kappa0, a, b = 3.68e22, 1.0, -3.5
            law = MyOpacity(kappa0=kappa0, a=a, b=b)
            expected = kappa0 * 1.0e-5 * 1.0e7 ** (-3.5)
            result = law.opacity(self._rho, self._T)
            assert result.value == pytest.approx(expected, rel=1e-6)

        def test_dlogkappa_dlogrho(self):
            """d ln κ / d ln ρ matches the power-law index a."""
            law = MyOpacity(kappa0=3.68e22, a=1.0, b=-3.5)
            assert law.dlogkappa_dlogrho(self._rho, self._T) == pytest.approx(1.0, rel=1e-6)

        def test_dlogkappa_dlogT(self):
            """d ln κ / d ln T matches the power-law index b."""
            law = MyOpacity(kappa0=3.68e22, a=1.0, b=-3.5)
            assert law.dlogkappa_dlogT(self._rho, self._T) == pytest.approx(-3.5, rel=1e-6)

**C-backed laws should also test:**

.. code-block:: python

        def test_is_c_backed(self):
            """IS_C_BACKED is True."""
            assert MyOpacity.IS_C_BACKED is True

        def test_c_object_initialised(self):
            """_c_object is not None after construction."""
            assert MyOpacity(1.0, 1.0, -3.5)._c_object is not None

**If you register in** ``_OPACITY_REGISTRY`` **also add to** ``test_utils.py``:

.. code-block:: python

    # In TestGetOpacityStringInput
    @pytest.mark.parametrize(
        "name, expected_class",
        [
            # ... existing entries ...
            ("my_opacity", MyOpacity),
        ],
    )
    def test_string_resolves_to_correct_type(self, name, expected_class):
        ...

    # Verify registry completeness
    def test_registry_is_complete(self):
        expected_names = {
            # ... existing ...
            "my_opacity",
        }
        assert set(_OPACITY_REGISTRY.keys()) == expected_names

Run the full opacity test suite to confirm everything passes:

.. code-block:: bash

    pytest tests/test_radiation/test_opacity/ -v

----

Checklist
----------

Use this checklist when adding a new opacity law:

.. list-table::
    :header-rows: 1

    * - Step
      - Action
      - Required
    * - 1
      - Create Python class in the appropriate subpackage:
        ``grey_opacity/rosseland/models.py`` for Rosseland mean,
        ``grey_opacity/planck/models.py`` for Planck mean
      - ✓ Always
    * - 2
      - Set ``mean_type`` class attribute: ``"rosseland"``, ``"planck"``, or ``None``
        (mean-type-agnostic).  See :attr:`~triceratops.radiation.opacity.base.OpacityLaw.mean_type`.
      - ✓ Always
    * - 3
      - Set ``IS_C_BACKED`` correctly
      - ✓ Always
    * - 4
      - Override ``_log_opacity``, ``_dlogkappa_dlogrho``, ``_dlogkappa_dlogT``
        (pure Python) OR implement Cython extension and ``_initialize_C_object``
      - ✓ Always
    * - 5
      - Keep ``_LOG_KAPPA`` in sync (even for C-backed laws)
      - ✓ Always
    * - 6
      - Export from the subpackage ``__init__.py`` chain and ``opacity/__init__.py``
      - ✓ Always
    * - 7
      - Add to ``_OPACITY_REGISTRY`` in ``utils.py``
      - If string-resolved
    * - 8
      - Write ``.pyx`` extension, add to build system (``setup.py``), and update
        Cython import paths in the Python class
      - If C-backed
    * - 9
      - Add tests (formula, derivatives, ``IS_C_BACKED``, ``mean_type``, registry)
      - ✓ Always
    * - 10
      - Document in ``opacity_theory.rst`` and ``opacity_user_guide.rst``
      - ✓ Always
