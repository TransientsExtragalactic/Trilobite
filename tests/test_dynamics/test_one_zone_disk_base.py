"""
Reusable pytest base class for testing :class:`~triceratops.dynamics.accretion.one_zone.OneZoneAccretionDiskBase`
subclasses and :class:`~triceratops.dynamics.accretion.one_zone.OneZoneAccretionResult`.

Architecture
------------
Tests are split into two parts:

1. :class:`TestOneZoneMetaValidation` — standalone unit tests for the
   :class:`~triceratops.dynamics.accretion.one_zone.base._OneZoneMeta`
   metaclass.  These tests verify that malformed parameter declarations are
   caught at class-definition time, before any instance is created.

2. :class:`BaseTestOneZoneDisk` — a reusable, parameterised base class for
   testing concrete :class:`~triceratops.dynamics.accretion.one_zone.OneZoneAccretionDiskBase`
   subclasses.  Concrete test modules inherit from this class, supply the
   required class-level attributes, and may add model-specific assertions.

Usage
-----
::

    from tests.test_dynamics.test_one_zone_disk_base import (
        BaseTestOneZoneDisk,
    )


    class TestMyDisk(BaseTestOneZoneDisk):
        MODEL = MyDisk
        DISK_KWARGS = {"mu": 0.62}
        M_BH = 3.0 * const.M_sun
        M_D_0 = 0.1 * const.M_sun
        R_IN = 3.0e6 * u.cm
        R_D_0 = 3.0e13 * u.cm
        ALPHA = 0.1
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from numpy.testing import assert_allclose

from triceratops.dynamics.accretion.one_zone import (
    OneZoneAccretionDiskBase,
    OneZoneAccretionResult,
)

# ================================================================== #
# Standalone: _OneZoneMeta schema validation                         #
# ================================================================== #


class TestOneZoneMetaValidation:
    """Unit tests for _OneZoneMeta — checked at class-definition time."""

    def test_missing_required_field_raises(self):
        """A RUNTIME_PARAMETERS entry missing a required field raises TypeError."""
        with pytest.raises(TypeError, match="missing required"):

            class _BadDisk(OneZoneAccretionDiskBase):
                RUNTIME_PARAMETERS = {
                    "alpha": {
                        "description": "viscosity",
                        "base_units": None,
                        "default": 0.1,
                        # "log_transform" deliberately omitted
                    }
                }

    def test_extra_field_raises(self):
        """An unrecognised key (e.g. a typo) in a spec dict raises TypeError."""
        with pytest.raises(TypeError, match="unrecognised"):

            class _BadDisk(OneZoneAccretionDiskBase):
                RUNTIME_PARAMETERS = {
                    "alpha": {
                        "description": "viscosity",
                        "base_units": None,
                        "default": 0.1,
                        "log_transform": False,
                        "log_trnasform": False,  # typo — should be caught
                    }
                }

    def test_spec_not_dict_raises(self):
        """A spec value that is not a dict raises TypeError."""
        with pytest.raises(TypeError, match="must be a dict"):

            class _BadDisk(OneZoneAccretionDiskBase):
                RUNTIME_PARAMETERS = {"alpha": "not_a_dict"}

    def test_empty_description_raises(self):
        """An empty or whitespace-only description string raises TypeError."""
        with pytest.raises(TypeError, match="description"):

            class _BadDisk(OneZoneAccretionDiskBase):
                RUNTIME_PARAMETERS = {
                    "alpha": {
                        "description": "   ",  # blank
                        "base_units": None,
                        "default": 0.1,
                        "log_transform": False,
                    }
                }

    def test_log_transform_not_bool_raises(self):
        """log_transform must be a bool; an int (0/1) raises TypeError."""
        with pytest.raises(TypeError, match="log_transform"):

            class _BadDisk(OneZoneAccretionDiskBase):
                RUNTIME_PARAMETERS = {
                    "alpha": {
                        "description": "viscosity",
                        "base_units": None,
                        "default": 0.1,
                        "log_transform": 1,  # int, not bool
                    }
                }

    def test_base_units_not_str_raises(self):
        """base_units must be a str or None; any other type raises TypeError."""
        with pytest.raises(TypeError, match="base_units"):

            class _BadDisk(OneZoneAccretionDiskBase):
                RUNTIME_PARAMETERS = {
                    "alpha": {
                        "description": "viscosity",
                        "base_units": 42,  # not str or None
                        "default": 0.1,
                        "log_transform": False,
                    }
                }

    def test_result_fields_missing_units_raises(self):
        """A RESULT_FIELDS entry missing 'units' raises TypeError."""
        with pytest.raises(TypeError, match="missing required"):

            class _BadDisk(OneZoneAccretionDiskBase):
                RESULT_FIELDS = {
                    "T_c": {
                        "description": "central temperature",
                        # "units" deliberately omitted
                    }
                }

    def test_context_parameters_validated(self):
        """CONTEXT_PARAMETERS entries are also validated."""
        with pytest.raises(TypeError, match="missing required"):

            class _BadDisk(OneZoneAccretionDiskBase):
                CONTEXT_PARAMETERS = {
                    "mu": {
                        "description": "mean molecular weight",
                        # "base_units" and "default" omitted
                    }
                }


class BaseTestOneZoneDisk:
    """Reusable pytest base class for :class:`~triceratops.dynamics.accretion.one_zone.OneZoneAccretionDiskBase`
    subclasses.

    All common tests are inherited automatically.  Concrete test classes only
    need to set the required class-level attributes and add any model-specific
    assertions.

    The test suite covers:

    - Metaclass parameter validation
    - Runtime-parameter and initial-condition processing
    - The three-step closure pipeline
    - ODE solve and basic result properties
    - :class:`~triceratops.dynamics.accretion.one_zone.OneZoneAccretionResult`
      dunder methods, field accessors, plotting utilities, and field
      interaction methods
    - HDF5 round-trip including reconstruction-cache persistence
    - Model serialisation (``to_spec_dict`` / ``from_spec_dict``)

    Required attributes
    -------------------
    MODEL : type
        The disk model class (a ``OneZoneAccretionDiskBase`` subclass).
    M_BH : `~astropy.units.Quantity`
        Black-hole mass.
    M_D_0 : `~astropy.units.Quantity`
        Initial disk mass.
    R_IN : `~astropy.units.Quantity`
        Inner truncation radius.
    R_D_0 : `~astropy.units.Quantity`
        Initial disk outer radius, used by the default :meth:`_compute_J_D_0`
        implementation.  Override that classmethod for models without
        ``disk._A`` / ``disk._B`` attributes.
    ALPHA : float
        Shakura-Sunyaev viscosity parameter.

    Optional attributes
    -------------------
    DISK_KWARGS : dict
        Keyword arguments forwarded to the model constructor.  Default ``{}``.
    N_EVAL : int
        Number of time-evaluation points in the ODE solve.  Default ``20``.
    T_SPAN_FRACTION : tuple of float
        ``(t_start / BOOTSTRAP_T0, t_end / BOOTSTRAP_T0)``.
        Default ``(0.01, 5.0)``.
    BOOTSTRAP_T0 : float
        Rough viscous-timescale estimate [s] used only for scaling the
        integration time span.  Default ``1e8``.
    """

    # ——— Required ———
    MODEL = None
    M_BH = None
    M_D_0 = None
    R_IN = None
    R_D_0 = None
    ALPHA = None

    # ——— Optional ———
    DISK_KWARGS = {}
    N_EVAL = 20
    T_SPAN_FRACTION = (0.01, 5.0)
    BOOTSTRAP_T0 = 1.0e8  # [s] — rough t_visc estimate used only for t_span scaling
    # Expected shape of _pack_cython_parameters() output.
    # Base disks: [MBH, R_in, alpha, mu, M_fb_0, t_fb, beta_fb] = 7
    # Adv disks:  [MBH, R_in, alpha, mu, xi, M_fb_0, t_fb, beta_fb] = 8
    EXPECTED_CYTHON_PARAM_SHAPE = (7,)

    # Per-subclass cache — each concrete class gets its own via __init_subclass__
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._cache = {}

    # ================================================================== #
    # Lazy shared-state accessors                                        #
    # ================================================================== #

    @classmethod
    def _disk(cls):
        """Return (or create) the shared disk model instance."""
        if "disk" not in cls._cache:
            cls._cache["disk"] = cls.MODEL(**cls.DISK_KWARGS)
        return cls._cache["disk"]

    @classmethod
    def _compute_J_D_0(cls, disk):
        """
        Compute the self-consistent initial angular momentum.

        Default implementation uses the one-zone scaling relation:

        .. math::

            J_{D,0} = \\frac{B}{A}\\,M_{D,0}\\,\\sqrt{G M_{\\rm BH} R_{D,0}}

        Override for models without ``disk._A`` / ``disk._B`` attributes.
        """
        xi = disk._B / disk._A
        return xi * cls.M_D_0.cgs * np.sqrt(const.G.cgs * cls.M_BH.cgs * cls.R_D_0.to(u.cm))

    @classmethod
    def _initial_conditions(cls):
        """Return (or build) the initial-condition dict."""
        if "ic" not in cls._cache:
            disk = cls._disk()
            J_D_0 = cls._compute_J_D_0(disk)
            cls._cache["ic"] = {"M_D_0": cls.M_D_0, "J_D_0": J_D_0}
        return cls._cache["ic"]

    @classmethod
    def _runtime_parameters(cls):
        """Return the runtime parameter dict for this model configuration."""
        if "rp" not in cls._cache:
            cls._cache["rp"] = {
                "M_BH": cls.M_BH,
                "R_in": cls.R_IN,
                "alpha": cls.ALPHA,
            }
        return cls._cache["rp"]

    @classmethod
    def _solve_result(cls):
        """Return (or compute) the shared ODE solve result."""
        if "result" not in cls._cache:
            disk = cls._disk()
            rp = cls._runtime_parameters()
            ic = cls._initial_conditions()
            import astropy.units as _u

            t_start = cls.T_SPAN_FRACTION[0] * cls.BOOTSTRAP_T0 * _u.s
            t_end = cls.T_SPAN_FRACTION[1] * cls.BOOTSTRAP_T0 * _u.s
            cls._cache["result"] = disk.solve(
                initial_conditions=ic,
                runtime_parameters=rp,
                t_span=(t_start, t_end),
                max_steps=cls.N_EVAL,
            )
        return cls._cache["result"]

    @classmethod
    def _fresh_result(cls):
        """Return a new OneZoneAccretionResult wrapping the same result array."""
        r = cls._solve_result()
        return OneZoneAccretionResult(
            model=r.model,
            result_array=r.result_array,
            run_params=r.run_params,
            success=r.success,
            message=r.message,
        )

    # ================================================================== #
    # Parameter Processing                                               #
    # ================================================================== #

    def test_process_runtime_parameters_returns_expected_keys(self):
        """Processed run_params contains the correct key for each declared parameter."""
        disk = self._disk()
        run_params = disk.process_runtime_parameters(self._runtime_parameters())
        for key, spec in disk.RUNTIME_PARAMETERS.items():
            expected = f"log_{key}" if spec["log_transform"] else key
            assert expected in run_params, f"Expected '{expected}' in run_params"

    def test_process_runtime_parameters_values_finite(self):
        """All processed runtime-parameter values are finite."""
        disk = self._disk()
        run_params = disk.process_runtime_parameters(self._runtime_parameters())
        for key, val in run_params.items():
            assert np.isfinite(val), f"run_params['{key}'] = {val} is not finite"

    def test_process_runtime_parameters_missing_required_raises(self):
        """Omitting a required runtime parameter raises ValueError."""
        disk = self._disk()
        required_key = next(k for k, spec in disk.RUNTIME_PARAMETERS.items() if spec["default"] is None)
        trimmed = {k: v for k, v in self._runtime_parameters().items() if k != required_key}
        with pytest.raises(ValueError, match=required_key):
            disk.process_runtime_parameters(trimmed)

    def test_process_initial_conditions_returns_array(self):
        """process_initial_conditions returns a finite float array of the right shape."""
        disk = self._disk()
        y0 = disk.process_initial_conditions(self._initial_conditions())
        assert isinstance(y0, np.ndarray)
        assert y0.shape == (len(disk.INITIAL_CONDITIONS),)
        assert np.all(np.isfinite(y0))

    def test_process_initial_conditions_missing_required_raises(self):
        """Omitting a required initial condition raises ValueError."""
        disk = self._disk()
        required_key = next(k for k, spec in disk.INITIAL_CONDITIONS.items() if spec["default"] is None)
        trimmed = {k: v for k, v in self._initial_conditions().items() if k != required_key}
        with pytest.raises(ValueError, match=required_key):
            disk.process_initial_conditions(trimmed)

    # ================================================================== #
    # Cython Integration Interface                                       #
    # ================================================================== #

    def test_build_cython_closure_returns_ready_object(self):
        """_build_cython_closure() returns a closure with all function pointers set."""
        disk = self._disk()
        closure = disk._build_cython_closure()
        assert closure.is_ready()

    def test_pack_cython_parameters_shape(self):
        """_pack_cython_parameters() returns a float64 array of the expected shape."""
        disk = self._disk()
        rp = disk.process_runtime_parameters(self._runtime_parameters())
        params = disk._pack_cython_parameters(rp)
        assert params.shape == self.EXPECTED_CYTHON_PARAM_SHAPE
        assert params.dtype == np.float64
        assert np.all(np.isfinite(params))

    # ================================================================== #
    # ODE Solve                                                          #
    # ================================================================== #

    def test_solve_succeeds(self):
        """solve() completes without failure."""
        assert self._solve_result().success

    def test_solve_returns_result_type(self):
        """solve() returns a OneZoneAccretionResult."""
        assert isinstance(self._solve_result(), OneZoneAccretionResult)

    def test_M_D_decreases(self):
        """Disk mass decreases over time (isolated viscous drain)."""
        result = self._solve_result()
        assert result.M_D[-1] < result.M_D[0]

    def test_M_D_initial_matches_input(self):
        """Initial disk mass in the solution matches the declared M_D_0."""
        result = self._solve_result()
        assert np.isclose(result.M_D[0].cgs.value, self.M_D_0.cgs.value, rtol=1.0e-6)

    def test_result_array_shape(self):
        """result_array has shape (n_result_fields, n_steps) with n_steps >= 1."""
        result = self._solve_result()
        assert result.result_array.ndim == 2
        assert result.result_array.shape[1] >= 1
        assert result.t.shape == (result.n_steps,)

    # ================================================================== #
    # Result Properties                                                  #
    # ================================================================== #

    def test_M_D_unit(self):
        assert self._solve_result().M_D.unit == u.g

    def test_M_D_solar_unit(self):
        assert self._solve_result().M_D_solar.unit == u.Msun

    def test_J_D_unit(self):
        assert self._solve_result().J_D.unit == u.g * u.cm**2 / u.s

    def test_data_returns_all_result_fields(self):
        """result.data produces every key declared in RESULT_FIELDS."""
        data = self._solve_result().data
        for key in self._disk().RESULT_FIELDS:
            assert key in data, f"Missing RESULT_FIELD: '{key}'"

    def test_data_quantities_finite(self):
        """All quantities in result.data are finite across the time grid."""
        data = self._solve_result().data
        for key, val in data.items():
            arr = val.value if hasattr(val, "value") else np.asarray(val)
            assert np.all(np.isfinite(arr)), f"Non-finite values in '{key}'"

    def test_to_table_colnames(self):
        """to_table() returns a QTable with columns for every RESULT_FIELD."""
        table = self._solve_result().to_table()
        for key in self._disk().RESULT_FIELDS:
            assert key in table.colnames

    # ================================================================== #
    # HDF5 Round-Trip                                                    #
    # ================================================================== #

    def test_to_hdf5_creates_file(self, tmp_path):
        p = tmp_path / "test.h5"
        self._solve_result().to_hdf5(p)
        assert p.exists()

    def test_from_hdf5_restores_t(self, tmp_path):
        p = tmp_path / "test.h5"
        result = self._solve_result()
        result.to_hdf5(p)
        loaded = OneZoneAccretionResult.from_hdf5(p)
        assert_allclose(np.asarray(loaded.t), np.asarray(result.t))

    def test_from_hdf5_restores_result_array(self, tmp_path):
        p = tmp_path / "test.h5"
        result = self._solve_result()
        result.to_hdf5(p)
        loaded = OneZoneAccretionResult.from_hdf5(p)
        assert_allclose(loaded.result_array, result.result_array)

    def test_from_hdf5_restores_run_params(self, tmp_path):
        p = tmp_path / "test.h5"
        result = self._solve_result()
        result.to_hdf5(p)
        loaded = OneZoneAccretionResult.from_hdf5(p)
        assert np.isclose(loaded.run_params["alpha"], result.run_params["alpha"])

    def test_from_hdf5_model_type_preserved(self, tmp_path):
        p = tmp_path / "test.h5"
        result = self._solve_result()
        result.to_hdf5(p)
        loaded = OneZoneAccretionResult.from_hdf5(p)
        assert type(loaded.model) is type(result.model)

    def test_to_hdf5_overwrite_false_raises(self, tmp_path):
        p = tmp_path / "test.h5"
        self._solve_result().to_hdf5(p)
        with pytest.raises(FileExistsError):
            self._solve_result().to_hdf5(p, overwrite=False)

    def test_to_hdf5_overwrite_true_succeeds(self, tmp_path):
        p = tmp_path / "test.h5"
        self._solve_result().to_hdf5(p)
        self._solve_result().to_hdf5(p, overwrite=True)

    def test_format_mismatch_raises(self, tmp_path):
        import h5py

        p = tmp_path / "bad.h5"
        with h5py.File(p, "w") as f:
            f.attrs["format"] = "SomethingElse"
        with pytest.raises(ValueError, match="format"):
            OneZoneAccretionResult.from_hdf5(p)

    # ================================================================== #
    # Model Spec                                                         #
    # ================================================================== #

    def test_to_spec_dict_has_target(self):
        """to_spec_dict() returns a dict with a 'target' key."""
        spec = self._disk().to_spec_dict()
        assert "target" in spec

    def test_to_spec_dict_target_contains_class_name(self):
        """The target string contains the model class name."""
        spec = self._disk().to_spec_dict()
        assert self.MODEL.__name__ in spec["target"]

    def test_spec_dict_is_json_serializable(self):
        """to_spec_dict() output is JSON-serializable."""
        spec = self._disk().to_spec_dict()
        json.dumps(spec)

    def test_from_spec_dict_round_trip(self):
        """from_spec_dict(to_spec_dict()) reconstructs an equivalent model instance."""
        disk = self._disk()
        spec = disk.to_spec_dict()
        loaded = self.MODEL.from_spec_dict(spec)
        assert isinstance(loaded, self.MODEL)

    # ================================================================== #
    # Model Dunder Methods                                               #
    # ================================================================== #

    def test_repr_is_str(self):
        """repr(disk) returns a str."""
        assert isinstance(repr(self._disk()), str)

    def test_repr_contains_class_name(self):
        """repr(disk) contains the concrete class name."""
        assert self.MODEL.__name__ in repr(self._disk())

    def test_str_is_str(self):
        """str(disk) returns a str."""
        assert isinstance(str(self._disk()), str)

    def test_str_contains_class_name(self):
        """str(disk) begins with the class name."""
        assert str(self._disk()).startswith(self.MODEL.__name__)

    def test_eq_reflexive(self):
        """A disk instance is equal to itself."""
        disk = self._disk()
        assert disk == disk

    def test_eq_equivalent_instances(self):
        """Two instances constructed with the same kwargs compare equal."""
        a = self.MODEL(**self.DISK_KWARGS)
        b = self.MODEL(**self.DISK_KWARGS)
        assert a == b

    def test_eq_different_type_returns_false(self):
        """Comparing a disk instance to a non-disk object returns False."""
        assert self._disk() != object()

    def test_not_hashable(self):
        """Disk instances are not hashable (mutable runtime cache)."""
        with pytest.raises(TypeError):
            hash(self._disk())

    def test_getitem_consistent_with_data(self):
        """result[key] returns the same value as result.data[key]."""
        result = self._fresh_result()
        key = next(iter(self._disk().RESULT_FIELDS))
        via_getitem = result[key]
        via_data = result.data[key]
        assert_allclose(
            via_getitem.value if hasattr(via_getitem, "value") else via_getitem,
            via_data.value if hasattr(via_data, "value") else via_data,
        )

    def test_getitem_unknown_key_raises(self):
        """result[key] raises KeyError for a key not in RESULT_FIELDS."""
        result = self._fresh_result()
        with pytest.raises(KeyError):
            _ = result["__nonexistent_field__"]

    # ================================================================== #
    # Result: t normalisation                                           #
    # ================================================================== #

    def test_t_is_plain_ndarray(self):
        """result.t is always a plain float64 ndarray (never an astropy Quantity)."""
        result = self._solve_result()
        assert isinstance(result.t, np.ndarray)
        assert not isinstance(result.t, u.Quantity)

    def test_t_dtype_is_float64(self):
        """result.t has dtype float64."""
        assert self._solve_result().t.dtype == np.float64

    def test_t_quantity_has_second_unit(self):
        """result.t_quantity carries the second unit."""
        assert self._solve_result().t_quantity.unit == u.s

    def test_t_quantity_values_match_t(self):
        """result.t_quantity.value equals result.t."""
        result = self._solve_result()
        assert_allclose(result.t_quantity.value, result.t)

    # ================================================================== #
    # Result: Dunder methods                                             #
    # ================================================================== #

    def test_len_positive(self):
        """len(result) is positive (at least the initial condition)."""
        assert len(self._solve_result()) >= 1

    def test_n_steps_positive(self):
        """result.n_steps is positive."""
        assert self._solve_result().n_steps >= 1

    def test_result_repr_is_str(self):
        """repr(result) returns a str."""
        assert isinstance(repr(self._solve_result()), str)

    def test_result_repr_contains_model_class(self):
        """repr(result) contains the concrete model class name."""
        assert self.MODEL.__name__ in repr(self._solve_result())

    def test_result_repr_contains_n_steps(self):
        """repr(result) contains the n_steps value."""
        result = self._solve_result()
        assert str(result.n_steps) in repr(result)

    def test_result_str_is_str(self):
        """str(result) returns a str."""
        assert isinstance(str(self._solve_result()), str)

    def test_result_str_starts_with_class_name(self):
        """str(result) begins with 'OneZoneAccretionResult'."""
        assert str(self._solve_result()).startswith("OneZoneAccretionResult")

    def test_result_eq_reflexive(self):
        """A result compares equal to itself."""
        r = self._solve_result()
        assert r == r

    def test_result_eq_two_equivalent_objects(self):
        """Two result objects wrapping the same arrays compare equal."""
        r = self._solve_result()
        r2 = self._fresh_result()
        assert r == r2

    def test_result_eq_different_type_returns_not_implemented(self):
        """Comparing a result to a non-result object returns False (not an error)."""
        assert self._solve_result() != object()

    def test_contains_builtin_key(self):
        """'M_D' in result returns True (built-in state key)."""
        assert "M_D" in self._solve_result()

    def test_contains_result_field(self):
        """A declared RESULT_FIELDS key is contained in result."""
        key = next(iter(self._disk().RESULT_FIELDS))
        assert key in self._solve_result()

    def test_not_contains_nonexistent_key(self):
        """An unknown key is not contained in result."""
        assert "__nonexistent__" not in self._solve_result()

    # ================================================================== #
    # Result: keys / field lists                                         #
    # ================================================================== #

    def test_field_list_contains_result_fields(self):
        """field_list includes every declared RESULT_FIELDS key."""
        fl = self._solve_result().field_list
        for key in self._disk().RESULT_FIELDS:
            assert key in fl

    def test_derived_field_list_includes_builtins(self):
        """derived_field_list always includes 't', 'M_D', 'J_D'."""
        dfl = self._solve_result().derived_field_list
        assert "t" in dfl
        assert "M_D" in dfl
        assert "J_D" in dfl

    def test_keys_returns_sorted_list(self):
        """keys() triggers reconstruction and returns a sorted list."""
        result = self._solve_result()
        k = result.keys()
        assert isinstance(k, list)
        assert k == sorted(k)

    def test_keys_includes_result_fields(self):
        """keys() includes all declared RESULT_FIELDS."""
        k = set(self._solve_result().keys())
        for key in self._disk().RESULT_FIELDS:
            assert key in k

    # ================================================================== #
    # Result: Field interaction methods                                  #
    # ================================================================== #

    def test_get_field_happy_path(self):
        """get_field(key) returns the same value as result[key]."""
        result = self._solve_result()
        key = next(iter(self._disk().RESULT_FIELDS))
        via_getitem = result[key]
        via_get = result.get_field(key)
        arr_gi = via_getitem.value if hasattr(via_getitem, "value") else np.asarray(via_getitem)
        arr_get = via_get.value if hasattr(via_get, "value") else np.asarray(via_get)
        assert_allclose(arr_gi, arr_get)

    def test_get_field_unknown_key_raises_key_error(self):
        """get_field raises KeyError for an unknown key with a helpful message."""
        result = self._solve_result()
        with pytest.raises(KeyError, match="not found"):
            result.get_field("__bad_key__")

    def test_has_field_known(self):
        """has_field returns True for a declared RESULT_FIELD."""
        key = next(iter(self._disk().RESULT_FIELDS))
        assert self._solve_result().has_field(key)

    def test_has_field_builtin(self):
        """has_field returns True for the built-in 'M_D' key."""
        assert self._solve_result().has_field("M_D")

    def test_has_field_unknown_returns_false(self):
        """has_field returns False for a nonexistent key."""
        assert not self._solve_result().has_field("__unknown__")

    def test_add_field_roundtrip(self):
        """add_field stores a value retrievable by get_field."""
        result = self._fresh_result()
        payload = np.ones(result.n_steps)
        result.add_field("custom_field", payload)
        retrieved = result.get_field("custom_field")
        assert_allclose(retrieved, payload)

    def test_add_field_overwrite_false_raises(self):
        """add_field raises ValueError when the field already exists and overwrite=False."""
        result = self._fresh_result()
        key = next(iter(self._disk().RESULT_FIELDS))
        with pytest.raises(ValueError, match="already exists"):
            result.add_field(key, np.ones(result.n_steps), overwrite=False)

    def test_add_field_overwrite_true_replaces(self):
        """add_field with overwrite=True silently replaces an existing field."""
        result = self._fresh_result()
        key = next(iter(self._disk().RESULT_FIELDS))
        sentinel = np.zeros(result.n_steps) - 999.0
        result.add_field(key, sentinel, overwrite=True)
        assert_allclose(result.get_field(key), sentinel)

    def test_remove_field_happy_path(self):
        """remove_field removes a user-added field from the cache."""
        result = self._fresh_result()
        result.add_field("user_scratch", np.zeros(result.n_steps))
        result.remove_field("user_scratch")
        assert not result.has_field("user_scratch")

    def test_remove_field_protected_result_field_raises(self):
        """remove_field raises ValueError for a declared RESULT_FIELDS key."""
        result = self._fresh_result()
        key = next(iter(self._disk().RESULT_FIELDS))
        with pytest.raises(ValueError, match="protected"):
            result.remove_field(key)

    def test_remove_field_builtin_raises(self):
        """remove_field raises ValueError for built-in state keys."""
        result = self._fresh_result()
        for builtin in ("t", "M_D", "J_D"):
            with pytest.raises(ValueError, match="protected"):
                result.remove_field(builtin)

    def test_remove_field_not_in_cache_raises(self):
        """remove_field raises KeyError for a field not in the cache."""
        result = self._fresh_result()
        with pytest.raises(KeyError):
            result.remove_field("__nonexistent__")

    # ================================================================== #
    # Result: Plotting utilities                                         #
    # ================================================================== #

    def test_plot_state_variables_returns_fig_axes(self):
        """plot_state_variables() returns a (fig, axes) pair."""
        import matplotlib.pyplot as plt

        result = self._solve_result()
        fig, axes = result.plot_state_variables()
        assert fig is not None
        assert len(np.asarray(axes).ravel()) == 2
        plt.close(fig)

    def test_plot_state_variables_accepts_existing_fig_axes(self):
        """plot_state_variables() draws into provided fig/axes without error."""
        import matplotlib.pyplot as plt

        result = self._solve_result()
        fig_in, axes_in = plt.subplots(1, 2)
        fig_out, axes_out = result.plot_state_variables(fig=fig_in, axes=axes_in)
        assert fig_out is fig_in
        plt.close(fig_in)

    def test_plot_field_returns_fig_ax(self):
        """plot_field(key) returns a (fig, ax) pair."""
        import matplotlib.pyplot as plt

        result = self._solve_result()
        key = next(iter(self._disk().RESULT_FIELDS))
        fig, ax = result.plot_field(key)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_field_accepts_existing_fig_ax(self):
        """plot_field() draws into provided fig/ax without error."""
        import matplotlib.pyplot as plt

        result = self._solve_result()
        key = next(iter(self._disk().RESULT_FIELDS))
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = result.plot_field(key, fig=fig_in, ax=ax_in)
        assert fig_out is fig_in
        assert ax_out is ax_in
        plt.close(fig_in)

    def test_plot_field_unknown_key_raises(self):
        """plot_field raises KeyError for an unknown field name."""
        result = self._solve_result()
        with pytest.raises(KeyError):
            result.plot_field("__bad_field__")

    def test_plot_all_fields_returns_fig_axes(self):
        """plot_all_fields() returns a (fig, axes) pair with the right number of cells."""
        import matplotlib.pyplot as plt

        result = self._solve_result()
        ncols = 3
        n_fields = len(self._disk().RESULT_FIELDS)
        nrows = (n_fields + ncols - 1) // ncols
        fig, axes = result.plot_all_fields(ncols=ncols)
        assert fig is not None
        assert np.asarray(axes).ravel().shape[0] == nrows * ncols
        plt.close(fig)

    # ================================================================== #
    # HDF5 Extra Fields Persistence                                      #
    # ================================================================== #

    def test_hdf5_saves_extra_fields_group(self, tmp_path):
        """to_hdf5 writes an 'extra_fields' group when user fields are present."""
        import h5py

        result = self._fresh_result()
        result.add_field("custom", np.ones(result.n_steps))
        p = tmp_path / "extra.h5"
        result.to_hdf5(p)
        with h5py.File(p, "r") as f:
            assert "extra_fields" in f

    def test_hdf5_no_extra_fields_group_when_empty(self, tmp_path):
        """to_hdf5 omits the 'extra_fields' group when no user fields are present."""
        import h5py

        result = self._fresh_result()
        p = tmp_path / "noextra.h5"
        result.to_hdf5(p)
        with h5py.File(p, "r") as f:
            assert "extra_fields" not in f

    def test_hdf5_extra_fields_roundtrip(self, tmp_path):
        """User-added fields survive a to_hdf5 / from_hdf5 round-trip."""
        result = self._fresh_result()
        payload = np.linspace(1.0, 2.0, result.n_steps)
        result.add_field("user_field", payload)
        p = tmp_path / "extra.h5"
        result.to_hdf5(p)
        loaded = OneZoneAccretionResult.from_hdf5(p)
        assert loaded.has_field("user_field")
        assert_allclose(loaded.get_field("user_field"), payload)

    def test_hdf5_data_values_match_after_roundtrip(self, tmp_path):
        """All RESULT_FIELDS values are identical before and after HDF5 round-trip."""
        result = self._fresh_result()
        p = tmp_path / "rt.h5"
        result.to_hdf5(p)
        loaded = OneZoneAccretionResult.from_hdf5(p)
        for key in self._disk().RESULT_FIELDS:
            orig = result[key]
            restored = loaded[key]
            orig_arr = orig.value if hasattr(orig, "value") else np.asarray(orig)
            rest_arr = restored.value if hasattr(restored, "value") else np.asarray(restored)
            assert_allclose(rest_arr, orig_arr, rtol=1e-10, err_msg=f"Mismatch for field '{key}'")

    # ================================================================== #
    # Diagnostic Plots                                                   #
    # ================================================================== #

    def test_state_variables_plot(self, diagnostic_plots, diagnostic_plots_dir):
        """Time-evolution of M_D and J_D (state variables)."""
        if not diagnostic_plots:
            pytest.skip("Diagnostic plots disabled.")

        result = self._solve_result()
        t = result.t

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].loglog(t, result.M_D.to(u.Msun).value)
        axes[0].set_xlabel("t [s]")
        axes[0].set_ylabel(r"$M_D\;[M_\odot]$")
        axes[0].set_title("Disk mass")

        axes[1].loglog(t, result.J_D.value)
        axes[1].set_xlabel("t [s]")
        axes[1].set_ylabel(r"$J_D\;[\mathrm{g\,cm^2\,s^{-1}}]$")
        axes[1].set_title("Disk angular momentum")

        fig.suptitle(self.MODEL.__name__)
        fig.tight_layout()
        fig.savefig(diagnostic_plots_dir / f"{self.MODEL.__name__}_state_variables.png", dpi=150)
        plt.close(fig)

    def test_result_fields_plot(self, diagnostic_plots, diagnostic_plots_dir):
        """One subplot per RESULT_FIELD showing log-time evolution."""
        if not diagnostic_plots:
            pytest.skip("Diagnostic plots disabled.")

        disk = self._disk()
        data = self._solve_result().reconstruct()
        t = data["t"].value
        fields = list(disk.RESULT_FIELDS.keys())

        ncols = 3
        nrows = (len(fields) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.asarray(axes).ravel()

        for ax, key in zip(axes, fields):
            val = data[key]
            y = val.value if hasattr(val, "value") else np.asarray(val)
            units_str = disk.RESULT_FIELDS[key]["units"] or "—"
            ax.loglog(t, np.abs(y))
            ax.set_xlabel("t [s]")
            ax.set_ylabel(f"[{units_str}]")
            ax.set_title(key)

        for ax in axes[len(fields) :]:
            ax.set_visible(False)

        fig.suptitle(f"{self.MODEL.__name__} — result fields", fontsize=13)
        fig.tight_layout()
        fig.savefig(diagnostic_plots_dir / f"{self.MODEL.__name__}_result_fields.png", dpi=150)
        plt.close(fig)
