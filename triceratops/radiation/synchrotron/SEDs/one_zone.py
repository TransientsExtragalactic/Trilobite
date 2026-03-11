"""
One-Zone Synchrotron Spectral Energy Distributions (SEDs).

This module implements a library of **one-zone phenomenological synchrotron
spectral energy distributions (SEDs)** for power-law electron populations,
following the standard theoretical framework developed in the literature
(e.g. :footcite:t:`GranotSari2002SpectralBreaks`).

The models describe emission from a **single homogeneous synchrotron-emitting
region** containing relativistic electrons with a power-law energy distribution
and a uniform magnetic field. The implemented SEDs correspond to the canonical
spectral regimes encountered in synchrotron theory, including radiative cooling
and synchrotron self-absorption.

The core design philosophy is to construct spectra using **log-space SED
composition**: complex synchrotron spectra are assembled by combining
(scale-free) smoothed broken power-law components. Each component introduces a
controlled change in spectral slope at a characteristic break frequency without
altering the global normalization. Working in logarithmic space ensures:

- numerical stability across many decades in frequency,
- correct asymptotic spectral slopes,
- smooth transitions between spectral segments,
- modular construction of multi-break spectra.

The module currently supports the canonical **one-zone synchrotron regimes**:

- optically thin spectra without cooling,
- slow- and fast-cooling synchrotron emission,
- synchrotron self-absorption (SSA),
- combinations of cooling and SSA breaks.

These models assume a **single homogeneous emission region** and therefore do
not include:

- spatially stratified or multi-zone emission,
- detailed radiative transfer effects,
- equal-arrival-time surface effects in relativistic flows,
- time-dependent particle evolution.

The provided SEDs should therefore be interpreted as **phenomenological spectral
models** suitable for fitting broadband observations or for order-of-magnitude
physical inference.

In addition to spectral evaluation, the module provides **closure relations**
that map between phenomenological SED parameters and physical model parameters
(e.g. magnetic field strength, emitting volume, electron Lorentz factors) under
assumed microphysical prescriptions.

.. note::

    A complete theoretical discussion of SED construction, including derivations
    and physical interpretation of the spectral regimes, is provided in
    :ref:`synch_sed_theory`. A user-guide description of this module can be
    found in :ref:`synchrotron_seds`, including usage examples.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from astropy import cosmology as cosmo
from astropy import units as u

from triceratops.radiation.constants import electron_rest_energy_cgs
from triceratops.utils.cosmology import get_cosmology, resolve_cosmological_distances
from triceratops.utils.log import triceratops_logger
from triceratops.utils.misc_utils import ensure_in_units
from triceratops.utils.sr_utils import compute_doppler_factor

from ..closure import _compute_ssa_BR_from_spectrum_dm22
from ..utils import c_1_cgs, compute_c5_parameter, compute_c6_parameter
from ._one_zone_closure import (
    COOLING_INV_FUNCTION_REGISTRY,
    SSA_COOLING_INV_FUNCTION_REGISTRY,
    SSA_INV_FUNCTION_REGISTRY,
    _inv_log_powerlaw_sbpl_sed,
)
from ._one_zone_functions import (
    COOLING_SED_FUNCTION_REGISTRY,
    SSA_COOLING_SED_FUNCTION_REGISTRY,
    SSA_SED_FUNCTION_REGISTRY,
    _log_powerlaw_sbpl_sed,
    smoothed_BPL,
)
from ._one_zone_normalization import (
    _log_normalize_powerlaw_sbpl_sed,
    _log_normalize_powerlaw_sbpl_sed_cool,
    _log_normalize_powerlaw_sbpl_sed_ssa,
    _log_normalize_powerlaw_sbpl_sed_ssa_cool,
)
from ._one_zone_ssa import (
    compute_ssa_frequencies_with_cooling,
    compute_ssa_frequencies_without_cooling,
    select_ssa_sed_regime_from_candidates_with_cooling,
    select_ssa_sed_regime_from_candidates_without_cooling,
)

# =============================================================
# Type Checking and Management
# =============================================================
if TYPE_CHECKING:
    from triceratops._typing import (
        _ArrayLike,
        _UnitBearingArrayLike,
        _UnitBearingScalarLike,
    )


# =============================================================
# SED Base Class
# =============================================================
# To help compartmentalize the name space and prevent any issues
# with clarity, we provide a small base class for SEDs to serve as a
# guide both for SED implementation and for documentation.
class SynchrotronSED(ABC):
    """
    Base class for synchrotron SED implementation.

    The :class:`SynchrotronSED` is a simple compartment for defining the
    structure of a specific spectral energy distribution. For each SED,
    one needs to provide

    1. The :meth:`sed` method (vis-a-vis the low-level ``_log_opt_sed`` method), which
       simply provides the phenomenological SED shape as a function of frequency and parameters. For example
       a broken power-law SED would implement the appropriate power-law segments and breaks in this method.

    2. The :meth:`from_params_to_physics` (vis-a-vis the low-level ``_opt_from_params_to_physics`` method), which
       provides the mapping from phenomenological SED parameters (e.g., break frequencies, flux normalizations, etc.)
       to physical parameters (e.g., magnetic field strength, radius, etc.) based on closure relations.
    3. The :meth:`from_physics_to_params` (vis-a-vis the low-level ``_opt_from_physics_to_params`` method), which
       provides the mapping from physical parameters (e.g., magnetic field strength, radius, etc.) to phenomenological
       SED parameters (e.g., break frequencies, flux normalizations, etc.) based on closure relations.

    Components (2) and (3) are **NOT REQUIRED** for an SED to be functional; however, they are generally necessary
    for inference. The necessary extent of the infrastructure is left to the implementer.

    .. hint::

        All of the existing SEDs are implemented using this base class as a guide. If you're
        in need of check out the other classes in this module.

    Instantiation
    --------------
    Instantiation of the SED class may be used to pre-compute class-wide constants, evaluate or construct kernels,
    etc; however, it should **NOT** be used to store SED-specific parameters. All SED-specific parameters
    should be passed directly to the relevant methods. This ensures that, during high-load inference tasks,
    no unnecessary re-instantiation of SED objects is required.
    """

    # ============================================================ #
    # Instantiation and basic structure                            #
    # ============================================================ #
    def __init__(self, *args, **kwargs):
        r"""
        Instantiate the SED object.

        Parameters
        ----------
        args:
            Positional arguments for SED instantiation.
        kwargs:
            Keyword arguments for SED instantiation.

        Notes
        -----
        The SED instantiation should NOT be used to store SED-specific parameters. All SED-specific parameters
        should be passed directly to the relevant methods. This ensures that, during high-load inference tasks,
        no unnecessary re-instantiation of SED objects is required.
        """
        pass

    def __call__(self, nu, **parameters):
        r"""
        Evaluate the SED at the given frequency.

        This is a thin wrapper around :meth:`sed` and exists purely for
        convenience, allowing SED objects to be used as callables.
        """
        return self.sed(nu, **parameters)

    def __repr__(self):
        return f"<{self.__class__.__name__}()>"

    # ============================================================ #
    # SED Function Implementation                                  #
    # ============================================================ #
    # Here should be the implementation of the SED function itself,
    # which is a function of nu and some set of additional parameters.
    @abstractmethod
    def _log_opt_sed(self, nu: "_ArrayLike", **parameters):
        r"""
        Low-level optimized log-space SED evaluation.

        This method implements the core numerical kernel for the synchrotron
        spectral energy distribution (SED) in **logarithmic space**. It is intended
        for performance-critical use and therefore assumes that all inputs are
        already provided in a validated, unit-consistent form.

        No unit handling, type checking, or safety checks are performed.

        Parameters
        ----------
        nu : float or array-like
            Natural logarithm of the frequency at which to evaluate the SED,

            .. math::

                \nu \equiv \ln\!\left(\frac{\nu_{\rm phys}}{\mathrm{Hz}}\right),

            where :math:`\nu_{\rm phys}` is the physical frequency expressed in
            Hz-equivalent CGS units.
        **parameters
            Additional dimensionless or CGS-valued parameters required for the
            SED calculation. The exact set of required parameters is
            implementation-specific.

        Returns
        -------
        float or array-like
            Natural logarithm of the synchrotron SED evaluated at the specified
            frequency.
        """
        raise NotImplementedError

    @abstractmethod
    def sed(self, nu: "_UnitBearingArrayLike", **parameters):
        r"""
        User-facing synchrotron SED evaluation.

        This method provides a high-level, user-friendly interface for computing
        the synchrotron spectral energy distribution (SED). It is responsible for
        handling unit validation and coercion, basic shape checking, and any other
        user-facing conveniences before dispatching to the low-level optimized
        backend.

        Internally, this method should convert inputs into the dimensionless,
        log-space form expected by the optimized implementation
        (:meth:`_log_opt_sed`).

        Parameters
        ----------
        nu : float, array-like, or astropy.units.Quantity
            Frequency at which to evaluate the SED. If provided without units,
            frequencies are assumed to be in Hz. If provided as an
            :class:`astropy.units.Quantity`, the value will be converted to
            Hz-equivalent CGS units before evaluation.

            The frequency may be specified as a scalar (to evaluate a single
            spectrum) or as a one-dimensional array (to evaluate the SED over
            a frequency grid).
        **parameters
            Additional parameters required for the SED calculation. These may
            include phenomenological SED parameters (e.g. break frequencies,
            normalization constants) or physical model parameters, depending
            on the specific SED implementation.

        Returns
        -------
        float, array-like, or astropy.units.Quantity
            The synchrotron SED evaluated at the specified frequency. Implementations
            may return either plain numerical values or
            :class:`astropy.units.Quantity` objects with appropriate physical units.

        Notes
        -----
        - Subclasses must implement this method.
        - The returned SED should be consistent with the conventions and units
          adopted by the corresponding low-level implementation.
        """
        raise NotImplementedError

    # =========================================================== #
    # Closure Relations Implementation                            #
    # =========================================================== #
    # Here we implement the closure relations to go forward and backward
    # between the physics parameters and the phenomenological SED parameters.
    def from_params_to_physics(self, **parameters):
        r"""
        Convert phenomenological SED parameters into physical parameters.

        This method provides a **user-facing interface** for mapping phenomenological
        SED parameters—such as break frequencies, peak fluxes, or normalization
        constants—into underlying physical quantities like magnetic field strength,
        emitting radius, characteristic electron energies, or energy densities.

        The mapping implemented by this method is **model-dependent** and typically
        relies on analytic closure relations derived from synchrotron theory, often
        supplemented by additional microphysical assumptions. Examples include:

        - assumptions about particle acceleration efficiency,
        - equipartition or near-equipartition between fields and particles,
        - prescriptions for the minimum Lorentz factor :math:`\gamma_m`,
        - geometric assumptions encoded via solid angle or filling factor terms.

        As a result, the inferred physical parameters should be interpreted within
        the context of the specific microphysical model adopted by the implementing
        subclass.

        This method is optional: an SED implementation is fully functional without
        it. However, closure relations are generally required for inference workflows,
        parameter estimation, and for coupling SEDs to dynamical or microphysical
        models.

        Parameters
        ----------
        **parameters
            Keyword arguments specifying phenomenological SED parameters. The exact
            set of required parameters is model-dependent and determined by the
            implementing subclass.

        Returns
        -------
        dict
            Dictionary containing inferred physical parameters. The contents,
            naming conventions, and units are implementation-specific.

        Notes
        -----
        - This method may perform unit validation, coercion, or shape checking
          before dispatching to the low-level optimized implementation
          :meth:`_opt_from_params_to_physics`.
        - The mapping is not guaranteed to be unique; degeneracies may be present
          depending on the assumed microphysics.
        """
        raise NotImplementedError

    def _opt_from_params_to_physics(self, **parameters):
        r"""
        Low-level optimized conversion from SED parameters to physical parameters.

        This method implements the same phenomenological-to-physical parameter
        mapping as :meth:`from_params_to_physics`, but assumes that all inputs are
        provided as dimensionless scalars or NumPy arrays in consistent CGS units.

        The mapping may encode analytic closure relations and microphysical
        assumptions specific to the SED model, but **no validation or safety checks**
        are performed at this level.

        Parameters
        ----------
        **parameters
            Keyword arguments specifying phenomenological SED parameters in CGS or
            dimensionless form. The exact set of required parameters is
            implementation-specific.

        Returns
        -------
        dict
            Dictionary containing inferred physical parameters in CGS units.

        Notes
        -----
        - This method is intended for internal use in performance-critical contexts.
        - It should be called only after unit handling and basic validation have been
          performed by :meth:`from_params_to_physics`.
        """
        raise NotImplementedError

    def _opt_from_physics_to_params(self, **parameters):
        r"""
        Low-level optimized conversion from physical parameters to SED parameters.

        This method implements the inverse mapping of
        :meth:`_opt_from_params_to_physics`, converting physical quantities—such as
        magnetic field strength, system size, particle energy scales, or energy
        densities—into phenomenological SED parameters like break frequencies or
        peak fluxes.

        The mapping assumes a specific set of synchrotron closure relations and
        microphysical prescriptions adopted by the implementing subclass.

        All inputs are assumed to be provided in CGS units or as dimensionless
        scalars. No unit validation, consistency checks, or physical sanity checks
        are performed.

        Parameters
        ----------
        **parameters
            Keyword arguments specifying physical parameters in CGS units. The exact
            set of required parameters is implementation-specific.

        Returns
        -------
        dict
            Dictionary containing phenomenological SED parameters in CGS or
            dimensionless form.

        Notes
        -----
        - This method is intended for internal use in performance-critical contexts.
        - Inverse mappings may not exist or may not be unique for all SED models.
        """
        raise NotImplementedError

    def from_physics_to_params(self, **parameters):
        r"""
        Convert physical parameters into phenomenological SED parameters.

        This method provides a **user-facing interface** for mapping physical
        quantities—such as magnetic field strength, emitting radius, particle
        energy scales, or energy densities—into phenomenological SED parameters
        like break frequencies, normalization constants, or spectral amplitudes.

        The conversion is based on analytic closure relations from synchrotron
        theory and typically incorporates additional microphysical assumptions
        (e.g., acceleration efficiency, geometry, or equipartition conditions)
        defined by the implementing subclass.

        This functionality is primarily used in inference workflows, where
        physical model parameters are sampled and must be translated into
        observable SED quantities.

        Parameters
        ----------
        **parameters
            Keyword arguments specifying physical parameters. The exact set of
            required parameters is model-dependent and determined by the
            implementing subclass.

        Returns
        -------
        dict
            Dictionary containing phenomenological SED parameters.

        Notes
        -----
        - This method may perform unit validation, coercion, or shape checking
          before dispatching to the low-level optimized implementation
          :meth:`_opt_from_physics_to_params`.
        - Subclasses that do not support inversion of closure relations may leave
          this method unimplemented.
        """
        raise NotImplementedError


class MultiSpectrumSynchrotronSED(SynchrotronSED, ABC):
    r"""
    Base class for synchrotron SEDs with multiple discrete spectral regimes.

    Many synchrotron models admit multiple global spectral "regimes" defined by
    the ordering of characteristic frequencies (e.g. :math:`\nu_a, \nu_m, \nu_c`).
    This class provides a standard pattern:

    1. Determine a regime label from the (global) model parameters.
    2. Optionally compute *derived* parameters needed to evaluate that regime
       (e.g. :math:`\nu_a` inferred from :math:`F_{\nu,\mathrm{pk}}`, geometry, etc.).
    3. Dispatch to a regime-specific optimized kernel.

    Subclasses implement:

    - :meth:`_compute_sed_regime`
    - :meth:`determine_sed_regime`
    - :meth:`_log_opt_sed_from_regime`

    Notes
    -----
    All "opt" methods operate on **unitless CGS scalars / NumPy arrays** and
    should not perform validation. Concrete subclasses must define whether
    the optimized backend expects linear frequencies ``nu`` or log-frequencies
    ``log_nu`` (see :meth:`_expects_log_frequency`).
    """

    #: Optional mapping/enum describing available regime functions.
    SPECTRUM_FUNCTIONS = None
    r"""
    Optional registry of synchrotron spectral regime functions.

    This attribute, when defined, provides a mapping or enumeration that
    associates **spectral regime identifiers** with the corresponding
    callable objects responsible for evaluating or constructing the SED
    in that regime.

    Typical use cases include:

    - Mapping a regime enum (e.g. :class:`SynchrotronSEDRegime`) to a
      concrete SED implementation.
    - Providing a lookup table for regime-specific normalization or
      evaluation logic.
    - Enabling dynamic dispatch based on the ordering of characteristic
      frequencies (e.g. :math:`\nu_a`, :math:`\nu_m`, :math:`\nu_c`).

    If set to ``None``, the class is assumed to represent a **single,
    fixed spectral regime** and does not support internal regime dispatch.

    Notes
    -----
    - Subclasses implementing multiple spectral regimes should override
      this attribute.
    - The objects stored in this registry are expected to be **callable**
      and to follow a consistent interface for SED evaluation or
      construction.
    """

    # ============================================================ #
    # Regime Management                                            #
    # ============================================================ #
    @abstractmethod
    def _compute_sed_regime(self, **parameters) -> tuple[Any, dict[str, Any]]:
        r"""
        Determine the global SED regime and any derived parameters.

        This method encodes the logic used to classify the SED into a discrete
        physical regime based on the ordering of characteristic frequencies.
        It may additionally compute *derived* parameters required to evaluate
        the spectrum (e.g. an inferred SSA frequency :math:`\nu_a`).

        The returned regime applies **globally** and does not depend on the
        sampling frequency grid.

        Parameters
        ----------
        **parameters
            Model parameters required to determine the regime. Typical examples
            include characteristic frequencies (or their log-values), peak flux
            normalizations, geometric factors, and microphysical parameters.

        Returns
        -------
        (regime, derived)
            regime : int or Enum-like
                Regime identifier. The interpretation is defined by the subclass.
            derived : dict
                Derived parameters needed for evaluation in the selected regime.
                This may include values such as ``log_nu_a`` or other cached
                quantities used by the regime-specific kernel.

        Notes
        -----
        - This method should be *fast* and free of allocations when possible.
        - No unit checking or validation should occur here.
        """
        raise NotImplementedError

    @abstractmethod
    def determine_sed_regime(self, **parameters) -> Any:
        r"""
        Determine the physical synchrotron spectral regime (public API).

        This method is intended for diagnostic and introspection use. It should
        perform any necessary unit handling and validation, then delegate to the
        optimized implementation (:meth:`_compute_sed_regime`).

        Parameters
        ----------
        **parameters
            User-facing model parameters. Typical examples include quantities
            such as :math:`\nu_m`, :math:`\nu_c`, :math:`F_{\nu,\mathrm{pk}}`, etc.

        Returns
        -------
        regime : int or Enum-like
            Regime identifier defined by the subclass.

        Notes
        -----
        - The regime does not depend on the frequency grid used for evaluation.
        - Subclasses should ensure that the regime returned here is consistent
          with the behavior of :meth:`sed`.
        """
        raise NotImplementedError

    # ============================================================ #
    # Regime-Specific SED Kernel                                   #
    # ============================================================ #
    @abstractmethod
    def _log_opt_sed_from_regime(
        self,
        nu: "_ArrayLike",
        regime: Any,
        **parameters,
    ):
        r"""
        Evaluate the log-space SED for a pre-determined regime.

        This method computes the logarithm of the SED for a single regime.
        All branching on ``regime`` should occur here (or above), and this
        method should assume that any derived parameters required for the regime
        have already been computed.

        Parameters
        ----------
        nu : float or array-like
            Frequency grid in optimized form. If :meth:`_expects_log_frequency`
            returns True, this is ``log_nu = log(ν)``. Otherwise it is linear ν.
        regime : int or Enum-like
            Regime identifier returned by :meth:`_compute_sed_regime`.
        **parameters
            Parameters required by the kernel, including any derived values
            produced by :meth:`_compute_sed_regime`.

        Returns
        -------
        float or array-like
            Logarithm of the SED evaluated at the given frequencies.

        Notes
        -----
        This is the performance-critical kernel. Implementations should avoid
        allocations and unnecessary branching when possible.
        """
        raise NotImplementedError

    # ============================================================ #
    # Log-Space Orchestration                                      #
    # ============================================================ #
    def _log_opt_sed(self, nu: "_ArrayLike", **parameters):
        r"""
        Log-space optimized SED evaluation with regime dispatch.

        This method determines the appropriate SED regime from the input
        parameters and dispatches to the corresponding regime-specific kernel.

        Subclasses should generally **not override** this method. Instead,
        implement :meth:`_compute_sed_regime` and :meth:`_log_opt_sed_from_regime`.

        Parameters
        ----------
        nu : float or array-like
            Frequency grid in optimized form. If :meth:`_expects_log_frequency`
            returns True, this is ``log_nu = log(ν)``. Otherwise it is linear ν.
        **parameters
            Parameters required for both regime determination and SED evaluation.

        Returns
        -------
        float or array-like
            Logarithm of the SED evaluated at the given frequencies.
        """
        regime, derived = self._compute_sed_regime(**parameters)
        merged = dict(parameters)
        merged.update(derived)
        return self._log_opt_sed_from_regime(nu, regime, **merged)


# ============================================================ #
# SED Implementations                                          #
# ============================================================ #
# Now we can include concrete implementations of various SEDs. Not
# all of the SEDs we plan to implement are currently implemented in the
# codebase, but we provide a few examples here to illustrate the structure.
class PowerLaw_SynchrotronSED(SynchrotronSED):
    r"""
    The canonical 1-zone power-law synchrotron SED (uncooled, optically thin).

    This class implements the standard synchrotron spectral energy distribution
    for an **uncooled, optically thin** population of relativistic electrons with
    a power-law energy distribution. See :ref:`synchrotron_theory` for a discussion
    of the relevant background physics and :ref:`synch_sed_theory` for a detailed
    derivation of this and related SEDs.

    This SED is applicable only in the simplest regimes where both radiative
    cooling and synchrotron self-absorption can be neglected. Typical examples
    include young systems whose cooling timescales are long compared to the
    dynamical time, or low-density environments where the emission remains
    optically thin across the observed band.

    .. important::

        This emission model contains some caveats:

        - The SED is strictly valid only for a single homogeneous emission region
          with a uniform magnetic field and a single power-law electron distribution. Real systems
          may exhibit spatial or temporal variations that deviate from this idealization.
        - While the core SED is implemented in the **rest frame** of the emitting material, normalization and
          closure both assume that any bulk motion is on axis with the observer. Off-axis viewing angles or
          complex geometries may require additional corrections not captured by this model. Such corrections
          can be avoided by treating the emitting volume fraction :math:`f_V` as degenerate with the true
          geometry (allowing for beaming).

    .. rubric:: Spectral Structure

    This SED consists of a single optically thin synchrotron spectrum with one
    physical spectral break at the injection frequency :math:`\nu_m`, and an
    optional high-frequency exponential cutoff at :math:`\nu_{\max}`:

    - :math:`F_\nu \propto \nu^{1/3}` for :math:`\nu < \nu_m`, corresponding to the
      low-frequency optically thin synchrotron tail.
    - :math:`F_\nu \propto \nu^{-(p-1)/2}` for :math:`\nu_m < \nu < \nu_{\max}`,
      corresponding to emission from the accelerated power-law electron population.
    - An exponential suppression above :math:`\nu_{\max}`.

    .. hint::

        For a detailed derivation and discussion of this spectrum, see
        :ref:`synch_sed_theory`.

    .. rubric:: SED Parameters

    The parameters entering this SED fall into three conceptual categories.

    .. tab-set::

        .. tab-item:: Free parameters (phenomenological)

            These parameters define the observable structure of the SED and are
            typically inferred directly from broadband data.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - Normalization flux density
                  - :math:`F_{\nu,\mathrm{norm}}`
                  - Flux density of the **dominant optically thin emitting
                    population** used to normalize the SED.
                    For this SED, the dominant optically thin emission occurs at
                    the injection frequency :math:`\nu_m`, so
                    :math:`F_{\nu,\mathrm{norm}} = F_\nu(\nu_m)`.
                * - Injection frequency
                  - :math:`\nu_m`
                  - Synchrotron characteristic frequency of the minimum-energy
                    electrons.
                * - Maximum frequency
                  - :math:`\nu_{\max}`
                  - High-frequency exponential cutoff.

        .. tab-item:: Hyper-parameters

            These parameters control the *shape* of the spectrum but are not
            usually directly constrained by broadband observations.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - Electron power-law index
                  - :math:`p`
                  - Index of the injected electron energy distribution,
                    :math:`N(\gamma) \propto \gamma^{-p}`.
                * - Smoothing parameter
                  - :math:`s`
                  - Controls the sharpness of the spectral transition at
                    :math:`\nu_m`.

    .. rubric:: Normalization and Closure

    The SED is **always normalized using** :math:`F_{\nu,\mathrm{norm}}`, defined
    as the flux density of the dominant optically thin emitting electron
    population. The location of this normalization frequency depends on the
    physical regime.

    In the present optically thin, uncooled case, the dominant emission occurs
    at :math:`\nu_m`, so the normalization coincides with the injection break.
    In more complex SEDs (e.g. with cooling or self-absorption), the normalization
    may instead correspond to a different characteristic frequency.

    Optional closure relations are provided to map physical model parameters
    (e.g. magnetic field strength, emitting volume, electron energy fractions)
    onto the phenomenological SED parameters under assumed microphysical
    conditions such as equipartition.

    See Also
    --------
    :class:`SynchrotronSED`
        Base class for synchrotron SED implementations.
    :class:`MultiSpectrumSynchrotronSED`
        Base class for multi-regime synchrotron SEDs.
    :class:`PowerLaw_Cooling_SynchrotronSED`
        Synchrotron SED including a radiative cooling break.
    :class:`PowerLaw_SSA_SynchrotronSED`
        Synchrotron SED including synchrotron self-absorption.
    :class:`PowerLaw_Cooling_SSA_SynchrotronSED`
        Synchrotron SED including both cooling and self-absorption.

    Examples
    --------
    The simplest use case is to evaluate the SED directly from phenomenological
    parameters inferred from data.

    .. code-block:: python

        import numpy as np
        import astropy.units as u
        from triceratops.radiation.synchrotron import (
            PowerLaw_SynchrotronSED,
        )

        sed = PowerLaw_SynchrotronSED()

        nu = np.logspace(8, 18, 300) * u.Hz

        F_nu = sed.sed(
            nu=nu,
            F_norm=1e-26 * u.erg / (u.cm**2 * u.s * u.Hz),
            nu_m=1e12 * u.Hz,
            nu_max=1e18 * u.Hz,
            p=2.4,
        )

    This evaluates the optically thin synchrotron spectrum with a normalization
    at :math:`\nu_m = 10^{12}\,\mathrm{Hz}` and an electron index :math:`p=2.4`.

    More commonly, the SED may be constructed from physical model parameters
    using the equipartition-based closure relation.

    .. code-block:: python

        params = sed.from_physics_to_params(
            B=0.1 * u.G,
            V=1e48 * u.cm**3,
            D_L=100 * u.Mpc,
            gamma_min=100.0,
            gamma_max=1e6,
            p=2.3,
            epsilon_E=0.1,
            epsilon_B=0.1,
            pitch_average=True,
        )

        F_nu = sed.sed(
            nu=nu,
            **params,
            p=2.3,
        )
    """

    # ============================================================ #
    # SED Function Implementation                                  #
    # ============================================================ #
    # Here should be the implementation of the SED function itself,
    # which is a function of nu and some set of additional parameters.
    def _log_opt_sed(
        self,
        log_nu: "_ArrayLike",
        log_F_norm: float,
        log_nu_m: float,
        log_nu_max: float = np.inf,
        p: float = 2.5,
        s: float = -0.5,
    ):
        r"""
        Compute the logarithm of the optically thin power-law synchrotron SED.

        This method implements the **core spectral shape** of the canonical,
        optically thin, uncooled synchrotron SED entirely in log-space. It is the
        lowest-level representation of the SED used internally by this class and
        by inference routines.

        The spectrum is constructed as a smoothed broken power law with a single
        physical break at the injection frequency :math:`\nu_m` and an exponential
        cutoff at :math:`\nu_{\max}`. The overall normalization is applied
        multiplicatively via the peak flux density.

        Parameters
        ----------
        log_nu : array-like
            Natural logarithm of the observing frequency (in Hz).
            This may be a scalar or an array and is assumed to be dimensionless.
        log_F_norm : float
            The natural logarithm of the normalization flux density, corresponding to the
            corresponding flux density at the dominant optically thin frequency (in this case, :math:`\nu_m`).
            Because this SED is optically thin at peak, this is also the flux density at the injection break.
        log_nu_m : float
            Natural logarithm of the injection (peak) frequency
            :math:`\nu_m`.
        log_nu_max : float, optional
            Natural logarithm of the maximum synchrotron frequency
            :math:`\nu_{\max}`. Above this frequency, the spectrum is exponentially
            suppressed. The default is :math:`+\infty`, corresponding to no cutoff.
        p : float, optional
            Power-law index of the injected electron energy distribution,
            :math:`N(\gamma) \propto \gamma^{-p}`.
        s : float, optional
            Smoothing parameter controlling the sharpness of the spectral break
            at :math:`\nu_m`. Larger values correspond to sharper transitions.

        Returns
        -------
        log_F_nu : array-like
            Natural logarithm of the flux density :math:`F_\nu` evaluated at
            ``log_nu``.

        Notes
        -----
        - This method performs **no unit handling** and assumes all inputs are
          dimensionless and expressed in CGS-consistent logarithmic form.
        - This method does **not** perform any physical normalization; it merely
          applies the spectral shape and multiplies by the supplied peak flux.
        - All higher-level interfaces (e.g. :meth:`sed`) are thin wrappers around
          this method.

        See Also
        --------
        _log_powerlaw_sbpl_sed :
            Low-level implementation of the smoothed broken power-law shape.
        """
        return (
            _log_powerlaw_sbpl_sed(
                log_nu,
                log_nu_m,
                log_nu_max,
                p,
                s,
            )
            + log_F_norm
        )

    def sed(
        self,
        nu: "_UnitBearingArrayLike",
        F_norm: "_UnitBearingScalarLike",
        nu_m: "_UnitBearingScalarLike",
        nu_max: "_UnitBearingScalarLike" = np.inf * u.Hz,
        p: float = 2.5,
        s: float = -0.5,
    ):
        r"""
        Evaluate the optically thin power-law synchrotron SED.

        This is the **public, unit-aware interface** for evaluating the canonical
        power-law synchrotron SED. It performs unit validation and conversion,
        dispatches to the internal log-space implementation, and returns the
        flux density with appropriate physical units.

        Parameters
        ----------
        nu : array-like or quantity
            Observing frequencies at which to evaluate the SED. If unit-bearing,
            must be convertible to Hz.
        F_norm : ~numpy.ndarray or float or ~astropy.units.Quantity
            The normalization flux density, corresponding to the
            corresponding flux density at the dominant optically thin frequency (in this case, :math:`\nu_m`).
            Because this SED is optically thin at peak, this is also the flux density at the injection break.
        nu_m : ~numpy.ndarray or float or ~astropy.units.Quantity
            Injection (peak) frequency :math:`\nu_m`. If unit-bearing, must be
            convertible to Hz.
        nu_max : ~numpy.ndarray or float or ~astropy.units.Quantity, optional
            Maximum synchrotron frequency :math:`\nu_{\max}`. Frequencies above
            this value are exponentially suppressed. The default corresponds to
            no cutoff.
        p : float, optional
            Electron power-law index.
        s : float, optional
            Smoothing parameter controlling the sharpness of the spectral break.

        Returns
        -------
        F_nu : array-like or quantity
            Flux density evaluated at ``nu``, returned with units of
            ``erg cm^-2 s^-1 Hz^-1``.

        Notes
        -----
        - This method is a thin wrapper around the internal log-space method
          :meth:`_log_opt_sed`.
        - All numerical evaluation is performed in logarithmic space for
          numerical stability.
        - This method does not enforce any physical closure; the parameters
          ``F_norm``, ``nu_m``, and ``nu_max`` are treated as phenomenological
          inputs.

        Examples
        --------
        Evaluate a simple power-law synchrotron SED:

        .. code-block:: python

            sed = PowerLaw_SynchrotronSED()
            nu = np.logspace(8, 18, 200) * u.Hz
            F_nu = sed.sed(
                nu=nu,
                F_norm=1e-26 * u.erg / (u.cm**2 * u.s * u.Hz),
                nu_m=1e12 * u.Hz,
                p=2.4,
            )

        """
        # Handle units
        nu = ensure_in_units(nu, "Hz")
        nu_m = ensure_in_units(nu_m, "Hz")
        nu_max = ensure_in_units(nu_max, "Hz")
        F_norm = ensure_in_units(F_norm, "erg cm^-2 s^-1 Hz^-1")

        # Convert to log-space dimensionless CGS
        log_nu = np.log(nu)
        log_nu_m = np.log(nu_m)
        log_nu_max = np.log(nu_max)
        log_F_norm = np.log(F_norm)

        # Dispatch to optimized implementation
        log_sed = self._log_opt_sed(
            log_nu=log_nu,
            log_F_norm=log_F_norm,
            log_nu_m=log_nu_m,
            log_nu_max=log_nu_max,
            p=p,
            s=s,
        )

        return np.exp(log_sed) * u.erg / (u.cm**2 * u.s * u.Hz)

    # =========================================================== #
    # Closure Relations Implementation                            #
    # =========================================================== #
    # Here we implement the closure relations to go forward and backward
    # between the physics parameters and the phenomenological SED parameters.
    def _opt_from_physics_to_params(
        self,
        log_B: float,
        log_R: float,
        log_D_L: float,
        log_gamma_min: float,
        log_gamma_max: float = np.inf,
        p: float = 2.5,
        f_V: float = 1.0,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        alpha: float = 1.0,
        gamma_bulk: float = 1.0,
        redshift: float = 0.0,
        pitch_average: bool = False,
    ):
        r"""
        Convert physical synchrotron parameters to phenomenological SED parameters.

        This method implements the **closure relation** that maps the physical
        description of a one-zone synchrotron source to the phenomenological
        parameters of the canonical broken power-law SED.

        The mapping assumes a homogeneous emitting region with a power-law
        electron distribution and determines the SED normalization under the
        equipartition assumptions described in :ref:`synch_sed_theory`.

        All quantities are provided in **logarithmic form** to ensure numerical
        stability and compatibility with sampling-based inference workflows.

        Parameters
        ----------
        log_B : float
            Natural logarithm of the magnetic field strength (Gauss).

        log_R : float
            Natural logarithm of the effective emitting radius (cm).

        log_D_L : float
            Natural logarithm of the luminosity distance (cm).

        log_gamma_min : float
            Natural logarithm of the minimum electron Lorentz factor
            :math:`\gamma_{\min}`.

        log_gamma_max : float, optional
            Natural logarithm of the maximum electron Lorentz factor
            :math:`\gamma_{\max}`. The default corresponds to no upper cutoff.

        p : float, optional
            Power-law index of the electron distribution.

        f_V: float, optional
            Filling factor of the emitting region, encoding geometric effects.

        epsilon_E : float, optional
            Fraction of post-shock energy carried by relativistic electrons.

        epsilon_B : float, optional
            Fraction of post-shock energy stored in magnetic fields.

        alpha : float, optional
            Electron pitch angle in radians. Ignored if
            ``pitch_average=True``.

        gamma_bulk : float, optional
            Bulk Lorentz factor of the emitting region.

        redshift : float, optional
            Cosmological redshift of the source.

        pitch_average : bool, optional
            If ``True``, use pitch-angle averaged synchrotron emissivity.

        Returns
        -------
        dict
            Dictionary containing the SED parameters

            - ``F_norm`` :  the flux normalization
            - ``F_peak`` : the peak flux
            - ``nu_m`` :  the injection frequency
            - ``nu_max`` : the maximum synchrotron frequency

        Notes
        -----
        This method is a thin wrapper around
        :func:`_log_normalize_powerlaw_sbpl_sed`, which performs the
        actual normalization calculation.
        """
        results = _log_normalize_powerlaw_sbpl_sed(
            log_B=log_B,
            log_R=log_R,
            log_D_L=log_D_L,
            log_gamma_min=log_gamma_min,
            log_gamma_max=log_gamma_max,
            p=p,
            f_V=f_V,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            alpha=alpha,
            gamma_bulk=gamma_bulk,
            redshift=redshift,
            pitch_average=pitch_average,
        )

        return {
            "F_norm": np.exp(results["log_F_norm"]),
            "F_peak": np.exp(results["log_F_peak"]),
            "nu_m": np.exp(results["log_nu_m"]),
            "nu_max": np.exp(results["log_nu_max"]),
        }

    def from_physics_to_params(
        self,
        B: "_UnitBearingScalarLike",
        R: "_UnitBearingScalarLike",
        gamma_min: float,
        gamma_max: float = np.inf,
        p: float = 2.5,
        f_V: float = 1.0,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        alpha: float = 1.0,
        gamma_bulk=1.0,
        redshift: float = None,
        luminosity_distance: "_UnitBearingScalarLike" = None,
        angular_diameter_distance: "_UnitBearingScalarLike" = None,
        proper_distance: "_UnitBearingScalarLike" = None,
        cosmology: cosmo.Cosmology = None,
        pitch_average: bool = False,
    ):
        r"""
        Construct phenomenological SED parameters from physical model parameters.

        This method provides the **public, unit-aware interface** to the synchrotron
        closure relation implemented by :meth:`_opt_from_physics_to_params`. It maps
        physical parameters describing the emitting region and electron population
        onto the phenomenological parameters defining the canonical power-law
        synchrotron spectral energy distribution (SED).

        All physical inputs may be provided as scalars, arrays, or unit-bearing
        quantities. Internally, values are converted to CGS units and evaluated in
        log-space for numerical stability.

        Cosmological quantities are resolved automatically. The user may specify a
        redshift directly or provide one of several distance measures. If necessary,
        missing quantities are computed using the supplied cosmology.

        Parameters
        ----------
        B : float, array-like, or ~astropy.units.Quantity
            Magnetic field strength in the emitting region. Must be convertible
            to Gauss.

        R : float, array-like, or ~astropy.units.Quantity
            Effective radius of the emitting region. Must be convertible to cm.

        gamma_min : float
            Minimum Lorentz factor of the electron population
            :math:`\gamma_{\min}`.

        gamma_max : float, optional
            Maximum electron Lorentz factor :math:`\gamma_{\max}`.
            The default corresponds to no upper cutoff.

        p : float, optional
            Power-law index of the injected electron energy distribution.

        f_V : float, optional
            Filling factor of the emitting region.

        epsilon_E : float, optional
            Fraction of post-shock internal energy carried by relativistic electrons.

        epsilon_B : float, optional
            Fraction of post-shock internal energy stored in magnetic fields.

        alpha : float, optional
            Electron pitch angle in radians. Ignored if ``pitch_average=True``.

        gamma_bulk : float, optional
            Bulk Lorentz factor of the emitting region. This determines the
            Doppler boosting applied to observed frequencies and fluxes.

        redshift : float, optional
            Cosmological redshift of the source.

        luminosity_distance : float, array-like, or ~astropy.units.Quantity, optional
            Luminosity distance to the source. Must be convertible to cm.

        angular_diameter_distance : float, array-like, or ~astropy.units.Quantity, optional
            Angular diameter distance to the source.

        proper_distance : float, array-like, or ~astropy.units.Quantity, optional
            Proper (comoving radial) distance to the source.

        cosmology : ~astropy.cosmology.Cosmology, optional
            Cosmology used to convert between redshift and distance measures.
            If not provided, the default cosmology returned by
            :func:`get_cosmology` is used.

        pitch_average : bool, optional
            If ``True``, use pitch-angle averaged synchrotron emissivity.
            Otherwise a fixed pitch angle ``alpha`` is used.

        Returns
        -------
        dict
            Dictionary containing the phenomenological SED parameters

            - ``F_norm`` : flux normalization
              (:math:`\mathrm{erg\,cm^{-2}\,s^{-1}\,Hz^{-1}}`)
            - ``F_peak`` : peak flux density
              (:math:`\mathrm{erg\,cm^{-2}\,s^{-1}\,Hz^{-1}}`)
            - ``nu_m`` : injection (characteristic) frequency
              (:math:`\mathrm{Hz}`)
            - ``nu_max`` : maximum synchrotron frequency
              (:math:`\mathrm{Hz}`)

        Notes
        -----
        This closure assumes a **single-zone homogeneous emission region** and
        determines the electron normalization using equipartition parameters
        (:math:`\epsilon_E`, :math:`\epsilon_B`). These assumptions are commonly
        used in phenomenological synchrotron modeling but are not physically
        required.

        Cosmological inputs are internally resolved into a consistent pair
        of values for the luminosity distance :math:`D_L` and redshift :math:`z`
        using :func:`resolve_cosmological_distances`.

        This method performs only the parameter conversion; it does not enforce
        spectral ordering constraints such as radiative cooling or synchrotron
        self-absorption.

        See Also
        --------
        _opt_from_physics_to_params :
            Log-space implementation of the closure relation.

        sed :
            Evaluate the synchrotron SED using the returned parameters.
        """
        # ------------------------------------------------------------------ #
        # Resolve cosmological information.
        # ------------------------------------------------------------------ #
        cosmology = get_cosmology(cosmology=cosmology)
        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        D_L = distances["luminosity_distance"]
        redshift = distances["redshift"]

        # Enforce units
        B = ensure_in_units(B, "G")
        R = ensure_in_units(R, "cm")
        D_L = ensure_in_units(D_L, "cm")

        # Convert to log-space
        log_B = np.log(B)
        log_R = np.log(R)
        log_D_L = np.log(D_L)
        log_gamma_min = np.log(gamma_min)
        log_gamma_max = np.log(gamma_max)

        # Dispatch to optimized log-space closure
        params = self._opt_from_physics_to_params(
            log_B=log_B,
            log_R=log_R,
            log_D_L=log_D_L,
            log_gamma_min=log_gamma_min,
            log_gamma_max=log_gamma_max,
            p=p,
            f_V=f_V,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            alpha=alpha,
            pitch_average=pitch_average,
            gamma_bulk=gamma_bulk,
            redshift=redshift,
        )

        # Convert back to physical units
        return {
            "F_norm": params["F_norm"] * u.erg / (u.cm**2 * u.s * u.Hz),
            "F_peak": params["F_peak"] * u.erg / (u.cm**2 * u.s * u.Hz),
            "nu_m": params["nu_m"] * u.Hz,
            "nu_max": params["nu_max"] * u.Hz,
        }

    def _opt_from_params_to_physics(
        self,
        log_F_peak: "_ArrayLike",
        log_nu_peak: "_ArrayLike",
        log_D_L: "_ArrayLike",
        gamma_min: float = 1.0,
        gamma_max: float = np.inf,
        p: float = 3.0,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        f_V: float = 0.5,
        gamma_bulk: float = 1.0,
        redshift: float = 0.0,
        alpha: float = None,
        pitch_average: bool = False,
    ):
        r"""
        Infer physical source parameters from phenomenological SED parameters.

        This method implements the **default inverse closure relation** for the
        canonical optically thin, uncooled synchrotron SED. It maps the observed
        spectral peak location and peak flux density onto a corresponding set of
        source parameters under a specific set of microphysical and geometric
        assumptions.

        In this default closure, the observed synchrotron SED is interpreted as
        arising from a **single-zone, homogeneous emitting region** containing a
        power-law distribution of relativistic electrons, with the spectrum peaking
        at the characteristic synchrotron frequency of the minimum-energy electrons.
        The inversion assumes that the observed peak corresponds to the same
        phenomenological normalization used elsewhere in this class, i.e. to the
        dominant optically thin emitting population.

        This is a low-level, log-space routine intended for internal use and for
        inference workflows where numerical stability matters. It performs no unit
        handling and assumes all inputs are already in logarithmic CGS-consistent
        form.

        .. important::

            This method provides a **default closure**, not a unique or universally
            correct physical interpretation of the SED. Different closure relations
            may be equally reasonable, or more appropriate, depending on the source
            class, observing band, and physical regime.

        .. warning::

            The inferred parameters are only meaningful when the assumptions of the
            underlying SED model are approximately satisfied. In particular, this
            inversion can be misleading if any of the following are important:

            - radiative cooling,
            - synchrotron self-absorption,
            - multi-zone emission,
            - strong geometric asymmetry,
            - significant departures from a simple power-law electron population,
            - strong relativistic beaming with angular structure or off-axis viewing,
            - time-dependent evolution across the emitting region.

            For highly relativistic transients such as GRB afterglows, this closure
            should generally be treated as yielding at best an **order-of-magnitude
            estimate**, not a precision physical inference.

        Parameters
        ----------
        log_F_peak : array-like
            Natural logarithm of the observed peak flux density in
            ``erg cm^-2 s^-1 Hz^-1``.

            In the context of this default closure, this is interpreted as the flux
            density of the dominant optically thin emitting population. For the
            present SED class, that corresponds to the injection break / optically
            thin peak.
        log_nu_peak : array-like
            Natural logarithm of the observed peak frequency in Hz.

            In this default closure, this is interpreted as the observed
            characteristic synchrotron frequency associated with the minimum-energy
            electrons.
        log_D_L : array-like
            Natural logarithm of the luminosity distance in cm.
        gamma_min : float, optional
            Minimum electron Lorentz factor, :math:`\gamma_{\min}`.
        gamma_max : float, optional
            Maximum electron Lorentz factor, :math:`\gamma_{\max}`.
            The default is :math:`+\infty`.
        p : float, optional
            Power-law index of the non-thermal electron distribution,
            :math:`N(\gamma) \propto \gamma^{-p}`.
        epsilon_E : float, optional
            Fraction of internal energy carried by relativistic electrons.
        epsilon_B : float, optional
            Fraction of internal energy stored in magnetic fields.
        f_V : float, optional
            Effective volume filling factor of the emitting region.

            This enters the inversion through the assumed source geometry and
            therefore directly affects the inferred physical scale.
        gamma_bulk : float, optional
            Bulk Lorentz factor of the emitting region.

            A simple on-axis Doppler correction is applied before inversion. No
            account is made of angular structure, equal-arrival-time surface
            effects, or off-axis viewing.
        redshift : float, optional
            Cosmological redshift of the source. Used to transform the observed SED
            peak into the comoving frame before inversion.
        alpha : float, optional
            Electron pitch angle in radians.

            Required when ``pitch_average=False``. Ignored when
            ``pitch_average=True``.
        pitch_average : bool, optional
            If ``True``, use pitch-angle-averaged synchrotron emissivity.
            If ``False``, a fixed pitch angle specified by ``alpha`` is used.

        Returns
        -------
        params : dict
            Dictionary containing the inferred physical parameters:

            - ``"R"`` : inferred source radius in cm,
            - ``"B"`` : inferred magnetic field strength in G.

            These are returned as plain dimensionless floating-point values in CGS
            units because this is the low-level internal implementation.

        Notes
        -----
        **Interpretation of the inversion**

        This routine does not recover a unique physical truth from the data. Rather,
        it finds the values of :math:`R` and :math:`B` that reproduce the supplied
        peak flux and peak frequency **within the default closure assumptions**.

        **What is assumed**

        The inversion assumes all of the following:

        - a single homogeneous emission zone,
        - a power-law electron distribution,
        - an optically thin spectrum near the relevant peak,
        - no cooling break affecting the fitted peak,
        - no synchrotron self-absorption shaping the fitted peak,
        - the peak is associated with :math:`\gamma_{\min}`,
        - the source can be described by the equipartition-style normalization used
          by the forward closure,
        - the observer is effectively on-axis relative to any bulk motion.

        **What is *not* modeled**

        This routine does not model:

        - time evolution,
        - multi-component spectra,
        - off-axis relativistic geometry,
        - radiative transfer through an absorbing synchrotron plasma,
        - non-power-law electron populations,
        - deviations from the assumed closure prescription.

        If any of those effects are important, users should implement or choose a
        different closure relation rather than relying on this default inversion.

        See Also
        --------
        from_params_to_physics :
            Public, unit-aware wrapper around this method.
        _opt_from_physics_to_params :
            Forward closure relation from physical parameters to SED parameters.
        """
        # ------------------------------------------------------------------ #
        # Pitch-angle treatment
        # ------------------------------------------------------------------ #
        if pitch_average:
            if alpha is not None:
                warnings.warn(
                    "Pitch averaging is enabled, so the provided `alpha` is ignored. "
                    "(Set `alpha=None` to suppress this warning.)",
                    stacklevel=2,
                )
            sin_alpha = None
        else:
            if alpha is None:
                raise ValueError("Pitch averaging is disabled, so a fixed pitch angle must be provided via `alpha`.")
            sin_alpha = np.sin(alpha)

        # ------------------------------------------------------------------ #
        # Transform observed peak quantities into the comoving frame.
        #
        # This applies the same minimal relativistic treatment used elsewhere
        # in the default closure: an on-axis Doppler correction plus the usual
        # cosmological redshift correction.
        #
        # IMPORTANT: this is intentionally simple and is not a substitute for a
        # full relativistic radiative-transfer treatment.
        # ------------------------------------------------------------------ #
        doppler_factor = compute_doppler_factor(gamma_bulk, 0.0)

        log_nu_peak_comoving = log_nu_peak - np.log(doppler_factor) + np.log1p(redshift)
        log_F_peak_comoving = log_F_peak - 3.0 * np.log(doppler_factor) + np.log1p(redshift)

        # ------------------------------------------------------------------ #
        # Invert the default closure in the comoving frame.
        # ------------------------------------------------------------------ #
        log_R, log_B = _inv_log_powerlaw_sbpl_sed(
            log_nu_peak_comoving,
            log_F_peak_comoving,
            log_D_L,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            p=p,
            epsilon_e=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            sin_alpha=sin_alpha,
        )

        return {
            "R": np.exp(log_R),
            "B": np.exp(log_B),
        }

    def from_params_to_physics(
        self,
        F_peak: "_UnitBearingScalarLike",
        nu_peak: "_UnitBearingScalarLike",
        gamma_min: float = 1.0,
        gamma_max: float = np.inf,
        p: float = 3.0,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        f_V: float = 0.5,
        gamma_bulk: float = 1.0,
        redshift: float = None,
        luminosity_distance: "_UnitBearingScalarLike" = None,
        angular_diameter_distance: "_UnitBearingScalarLike" = None,
        proper_distance: "_UnitBearingScalarLike" = None,
        cosmology: cosmo.Cosmology = None,
        alpha: float = None,
        pitch_average: bool = False,
    ):
        r"""
        Infer physical source parameters from observed SED peak quantities.

        This is the **public, unit-aware interface** for the default inverse closure
        relation of :class:`PowerLaw_SynchrotronSED`. It converts an observed peak
        flux density and peak frequency into inferred physical source parameters
        under the assumptions of the default single-zone optically thin synchrotron
        model.

        The inversion is performed using the same phenomenological interpretation
        adopted by the forward closure: the supplied peak flux and peak frequency are
        assumed to correspond to the dominant optically thin synchrotron-emitting
        electron population. In the present SED family, that means the observed peak
        is identified with the injection-scale synchrotron frequency.

        .. important::

            This method provides a **convenient default closure**, not a uniquely
            privileged physical model. Users should treat it as one physically
            motivated mapping from observables to source parameters, not as the only
            permissible interpretation of the data.

        .. warning::

            The returned values can be badly misleading if the spectrum being fit is
            not actually described by the assumptions of this SED class and its
            closure. In particular, do **not** interpret the results literally if the
            observed peak is shaped by synchrotron self-absorption, radiative
            cooling, multiple emitting zones, or strong time evolution.

        Parameters
        ----------
        F_peak : ~numpy.ndarray or float or ~astropy.units.Quantity
            Observed peak flux density.

            Must be convertible to ``erg cm^-2 s^-1 Hz^-1``. Within this default
            closure, this is interpreted as the flux density of the dominant
            optically thin emitting population.
        nu_peak : ~numpy.ndarray or float or ~astropy.units.Quantity
            Observed peak frequency.

            Must be convertible to Hz. Within this default closure, this is
            interpreted as the characteristic synchrotron frequency of the
            minimum-energy electrons.
        gamma_min : float, optional
            Minimum electron Lorentz factor, :math:`\gamma_{\min}`.
        gamma_max : float, optional
            Maximum electron Lorentz factor, :math:`\gamma_{\max}`.
        p : float, optional
            Power-law index of the electron energy distribution.
        epsilon_E : float, optional
            Fraction of internal energy carried by relativistic electrons.
        epsilon_B : float, optional
            Fraction of internal energy stored in magnetic fields.
        f_V : float, optional
            Effective volume filling factor of the emitting region.
        gamma_bulk : float, optional
            Bulk Lorentz factor of the source. A simple on-axis Doppler correction is
            applied when converting observed quantities into the comoving frame.
        redshift : float, optional
            Source redshift. If not given explicitly, it may be inferred indirectly
            from the provided cosmological distance information.
        luminosity_distance : ~numpy.ndarray or float or ~astropy.units.Quantity, optional
            Luminosity distance to the source. Must be convertible to cm.
        angular_diameter_distance : ~numpy.ndarray or float or ~astropy.units.Quantity, optional
            Angular-diameter distance to the source.
        proper_distance : ~numpy.ndarray or float or ~astropy.units.Quantity, optional
            Proper distance to the source.
        cosmology : `astropy.cosmology.Cosmology`, optional
            Cosmology used to resolve missing cosmological distance information.
            If not provided, the configured default cosmology is used.
        alpha : float, optional
            Electron pitch angle in radians. Required if ``pitch_average=False``.
        pitch_average : bool, optional
            If ``True``, use pitch-angle-averaged synchrotron emissivity.
            If ``False``, a fixed pitch angle must be supplied via ``alpha``.

        Returns
        -------
        params : dict
            Dictionary containing the inferred physical parameters:

            - ``"R"`` : source radius, returned in cm,
            - ``"B"`` : magnetic field strength, returned in G.

        Notes
        -----
        **Physical meaning**

        The returned :math:`R` and :math:`B` are not direct measurements. They are
        the values implied by the input observables **if** the source obeys the
        default closure relation assumed by this class.

        **Core assumptions**

        This inversion assumes:

        - a single-zone homogeneous source,
        - optically thin synchrotron emission near the relevant peak,
        - no cooling break affecting that peak,
        - no self-absorption shaping that peak,
        - a power-law electron distribution with index ``p``,
        - a closure prescription parameterized by ``epsilon_E`` and ``epsilon_B``,
        - a filling-factor description of the emitting volume via ``f_V``,
        - simple on-axis bulk relativistic boosting when ``gamma_bulk > 1``.

        **When not to use this method**

        This method is generally *not* appropriate when:

        - the observed peak is actually an SSA turnover,
        - the source is cooling-dominated,
        - the spectrum contains multiple components,
        - the emitting region is strongly stratified or multi-zone,
        - relativistic geometry and equal-arrival-time effects matter,
        - the physical closure differs from the equipartition-style prescription
          assumed here.

        In such cases, a different SED subclass or a custom closure relation should
        be preferred.

        **Cosmological handling**

        This method accepts several ways of specifying distance information. The
        helper :func:`resolve_cosmological_distances` is used to enforce a
        self-consistent set of cosmological quantities before the inversion is
        performed.

        See Also
        --------
        _opt_from_params_to_physics :
            Low-level log-space implementation of the inverse closure.
        from_physics_to_params :
            Forward closure from physical source parameters to SED parameters.
        sed :
            Evaluate the SED for a given set of phenomenological parameters.

        Examples
        --------
        Infer a characteristic source radius and magnetic field from a measured
        optically thin synchrotron peak:

        .. code-block:: python

            results = sed.from_params_to_physics(
                F_peak=1e-26 * u.erg / (u.cm**2 * u.s * u.Hz),
                nu_peak=1e10 * u.Hz,
                luminosity_distance=100 * u.Mpc,
                p=2.5,
                gamma_min=10.0,
                epsilon_E=0.1,
                epsilon_B=0.1,
                pitch_average=True,
            )

            R = results["R"]
            B = results["B"]

        The returned values should be interpreted as those implied by the **default
        closure assumptions** of this class, not as model-independent measurements.
        """
        # ------------------------------------------------------------------ #
        # Resolve cosmological information.
        # ------------------------------------------------------------------ #
        cosmology = get_cosmology(cosmology=cosmology)
        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        D_L = distances["luminosity_distance"]
        redshift = distances["redshift"]

        # ------------------------------------------------------------------ #
        # Unit enforcement and conversion to internal log-space representation.
        # ------------------------------------------------------------------ #
        log_F_peak = np.log(ensure_in_units(F_peak, "erg cm^-2 s^-1 Hz^-1"))
        log_nu_peak = np.log(ensure_in_units(nu_peak, "Hz"))
        log_D_L = np.log(ensure_in_units(D_L, "cm"))

        # ------------------------------------------------------------------ #
        # Dispatch to the low-level inverse closure.
        # ------------------------------------------------------------------ #
        results = self._opt_from_params_to_physics(
            log_F_peak=log_F_peak,
            log_nu_peak=log_nu_peak,
            log_D_L=log_D_L,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            p=p,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            gamma_bulk=gamma_bulk,
            redshift=redshift,
            alpha=alpha,
            pitch_average=pitch_average,
        )

        return {
            "R": results["R"] * u.cm,
            "B": results["B"] * u.G,
        }


class PowerLaw_Cooling_SSA_SynchrotronSED(MultiSpectrumSynchrotronSED):
    r"""
    Synchrotron spectral energy distribution with cooling and self-absorption.

    This class implements the **full piecewise synchrotron spectrum** for a
    power-law electron population, including:

    - Radiative cooling (fast, slow, and non-cooling regimes),
    - Synchrotron self-absorption (SSA),
    - Stratified SSA corrections where applicable,
    - A high-frequency exponential cutoff.

    The implementation follows the standard GRB / supernova afterglow
    formalism (e.g. :footcite:t:`GranotSari2002SpectralBreaks`, :footcite:t:`2020MNRAS.493.3521B`,
    :footcite:t:`duran2013radius`, :footcite:t:`2025ApJ...992L..18S`, :footcite:t:`GaoSynchrotronReview2013`,
    etc.)

    Specifically we implement spectra using **log-space SED surgery** with scale-free smoothed broken
    power laws (SFBPLs). The spectrum is assembled by:

    1. Determining the **global spectral regime** from characteristic frequencies,
    2. Computing any **derived break frequencies** required by that regime,
    3. Dispatching to a **regime-specific optimized kernel**,
    4. Applying an overall normalization via the peak flux density.

    .. hint::

        For a detailed derivation of the spectral segments and break orderings, see
        :ref:`synchrotron_theory` and :ref:`synch_sed_theory`.

    .. rubric:: Spectral Structure

    The SED is globally classified into one of several discrete regimes based on
    the ordering of the characteristic frequencies:

    - Injection frequency :math:`\nu_m`,
    - Cooling frequency :math:`\nu_c`,
    - Self-absorption frequency :math:`\nu_a` (computed internally),
    - Maximum synchrotron frequency :math:`\nu_{\max}`.

    Each regime corresponds to a specific ordering (e.g.
    :math:`\nu_a < \nu_c < \nu_m < \nu_{\max}`) and therefore to a unique set of
    spectral slopes. The regime selection is **global** and does not depend on the
    frequency grid used for evaluation. As such, when the SED is called, the parameters
    are used once to determine the regime and then to evaluate the SED at all requested frequencies.

    .. rubric:: SED Parameters

    The parameters entering this SED fall into three conceptual categories.

    .. tab-set::

        .. tab-item:: Free parameters (phenomenological)

            These parameters define the observable structure of the SED and are
            typically inferred directly from data.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - Peak flux density
                  - :math:`F_{\nu,\mathrm{norm}}`
                  - The equivalent flux at the dominant electron frequency in
                    an optically thin medium. This sets the SED peak. :meth:`from_physics_to_params` maps
                    physical parameters onto this quantity.
                * - Injection frequency
                  - :math:`\nu_m`
                  - Synchrotron frequency of minimum-energy electrons
                * - Cooling frequency
                  - :math:`\nu_c`
                  - Frequency corresponding to the cooling Lorentz factor
                * - Maximum frequency
                  - :math:`\nu_{\max}`
                  - High-energy cutoff frequency
                * - (Optional) stratified SSA frequency
                  - :math:`\nu_{ac}`
                  - Transition frequency for stratified SSA regimes

        .. tab-item:: Hyper-parameters

            These parameters control the *shape* and smoothness of the spectrum
            but are not usually directly inferred from broadband data.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - Electron power-law index
                  - :math:`p`
                  - Index of the injected electron distribution
                * - Smoothing parameter
                  - :math:`s`
                  - Controls sharpness of spectral breaks
                * - Emission solid angle
                  - :math:`\Omega`
                  - Effective emitting area divided by distance squared
                * - Minimum Lorentz factor
                  - :math:`\gamma_m`
                  - Minimum electron Lorentz factor

        .. tab-item:: Derived parameters (internal)

            These quantities are **not user inputs**, but are computed internally
            from the free parameters and microphysical assumptions.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - SSA frequency
                  - :math:`\nu_a`
                  - Self-absorption break frequency
                * - Cooling regime
                  - —
                  - Fast, slow, or non-cooling classification
                * - Regime identifier
                  - —
                  - Discrete index selecting the SED kernel

    See Also
    --------
    :class:`SynchrotronSED` : Base class for synchrotron SED implementations.
    :class:`MultiSpectrumSynchrotronSED` : Base class for multi-regime synchrotron SEDs.
    :class:`PowerLaw_Cooling_SynchrotronSED` : Synchrotron SED with cooling break.
    :class:`PowerLaw_SSA_SynchrotronSED` : Synchrotron SED with synchrotron self-absorption.
    :class:`PowerLaw_SynchrotronSED` : Canonical optically-thin power-law synchrotron SED.

    References
    ----------
    .. footbibliography::
    """

    # ============================================================ #
    # Declare the spectrum functions mapping                       #
    # ============================================================ #
    # These are the synchrotron SED functions with SSA and cooling. We use
    # this enum class to trace through the runtime which regime we're in.
    SPECTRUM_FUNCTIONS = SSA_COOLING_SED_FUNCTION_REGISTRY
    SPECTRUM_INVERSION_FUNCTIONS = SSA_COOLING_INV_FUNCTION_REGISTRY

    # ============================================================ #
    # Regime Management                                            #
    # ============================================================ #
    # Regime management in this class is non-trivial as the SSA frequency
    # must be self-consistently computed from the other parameters for each
    # regime. This requires some careful bookkeeping. We implement this in the
    # _compute_sed_regime method below.
    def _compute_sed_regime(
        self,
        log_F_norm: float,
        log_nu_m: float,
        log_nu_c: float,
        log_nu_max: float,
        log_omega: float,
        log_gamma_m: float,
        p: float = 3.0,
    ):
        r"""
        Determine the synchrotron spectral regime and SSA frequency.

        This low-level method classifies the synchrotron spectrum into a **single,
        global spectral regime** based on the ordering of the characteristic
        frequencies and returns both:

        1. A discrete **regime identifier**, and
        2. The corresponding **self-absorption frequency** :math:`\nu_a`
           appropriate to that regime.

        The classification proceeds in two stages:

        1. Determine whether the system is **fast-cooling**, **slow-cooling**, or
           **non-cooling** by comparing :math:`\nu_c` and :math:`\nu_m`.
        2. Within that cooling class, select the specific SSA spectrum by comparing
           the candidate self-absorption frequencies to :math:`\nu_m` and
           :math:`\nu_c`.

        All inputs and outputs are assumed to be in **natural logarithmic space**.

        Parameters
        ----------
        log_F_norm : float
            Natural logarithm of the normalization flux density corresponding to the *optically thin* equivalent
            emission at the dominant electron frequency. See
            :ref:`sed_normalization` for details on how to determine the appropriate normalization.
        log_nu_m : float
            Natural logarithm of the injection frequency
            :math:`\log \nu_m`.
        log_nu_c : float
            Natural logarithm of the cooling frequency
            :math:`\log \nu_c`.
        log_nu_max : float
            Natural logarithm of the maximum synchrotron frequency
            :math:`\log \nu_{\max}`.
        log_omega : float
            Natural logarithm of the effective emission solid angle
            :math:`\log \Omega`.
        log_gamma_m : float
            Natural logarithm of the minimum electron Lorentz factor
            :math:`\log \gamma_m`.

        Returns
        -------
        tuple
            A two-element tuple ``(regime, log_nu_a)`` where:

            - ``regime`` is a member of
              :class:`~_SynchrotronSSACoolingSEDFunctions` identifying the spectral
              branch to evaluate.
            - ``log_nu_a`` is the natural logarithm of the self-absorption frequency
              appropriate to that regime.

        Notes
        -----
        - This method performs **no unit checking** and assumes all inputs are valid.
        - The returned regime applies **globally** to the spectrum.
        - Errors are raised if the frequency ordering is inconsistent with any
          supported spectral configuration.
        """
        # Begin by calculating the SSA frequency from the other parameters. This also
        # tells some regime information. 0=Fast cooling, 1=Slow cooling, 2=No cooling.
        log_nu_ssa, cooling_regime = compute_ssa_frequencies_with_cooling(
            log_F_norm, log_omega, log_nu_m, log_nu_c, log_nu_max, log_gamma_m, p
        )
        return select_ssa_sed_regime_from_candidates_with_cooling(
            log_nu_ssa,
            cooling_regime,
            log_nu_m,
            log_nu_c,
            log_nu_max,
        )

    def determine_sed_regime(
        self,
        F_norm: "_UnitBearingScalarLike",
        nu_m: "_UnitBearingScalarLike",
        nu_c: "_UnitBearingScalarLike",
        nu_max: "_UnitBearingScalarLike" = np.inf,
        omega: float = 1.0,
        gamma_m: float = 1.0,
    ):
        r"""
        Determine the synchrotron spectral regime from physical parameters.

        This is the **user-facing interface** for regime determination. It performs
        unit validation and coercion, converts all inputs to logarithmic CGS form,
        and dispatches to the optimized internal routine
        :meth:`_compute_sed_regime`.

        This method is intended for **diagnostic and introspection purposes** and
        does not compute the SED itself.

        Parameters
        ----------
        F_norm : ~astropy.units.Quantity or float
            Normalization flux density corresponding to the *optically thin* equivalent
            emission at the dominant electron frequency. This should be a :class:`~astropy.units.Quantity`
            with units convertible to :math:`\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, \mathrm{Hz}^{-1}`
            or a float in CGS units. See :ref:`sed_normalization` for details on how to determine the appropriate
            normalization.
        nu_m : ~astropy.units.Quantity or float
            Injection frequency :math:`\nu_m`. May be either a :class:`~astropy.units.Quantity`
            with units convertible to Hz or a float in CGS units.
        nu_c : ~astropy.units.Quantity or float
            Cooling frequency :math:`\nu_c`. May be either a :class:`~astropy.units.Quantity`
            with units convertible to Hz or a float in CGS units.
        nu_max : ~astropy.units.Quantity or float
            Maximum synchrotron frequency :math:`\nu_{\max}`. May be either a :class:`~astropy.units.Quantity`
            with units convertible to Hz or a float in CGS units. The default (``np.inf``) corresponds to no cutoff.
        omega : float
            Effective emission solid angle. This should be a floating-point number
            representing the ratio of the emitting area to the square of the distance
            to the observer (i.e. :math:`\Omega = A / D^2`).
        gamma_m : float
            Minimum electron Lorentz factor. This should be a dimensionless floating-point number.

        Returns
        -------
        regime : callable
            A member of the :attr:`~PowerLaw_Cooling_SSA_SynchrotronSED.SPECTRUM_FUNCTIONS` enumeration identifying the
            applicable synchrotron spectral regime.

            Each enum member encodes a specific ordering of the characteristic
            synchrotron frequencies (e.g. :math:`\nu_a`, :math:`\nu_m`, :math:`\nu_c`,
            :math:`\nu_{\max}`) and uniquely determines the corresponding spectral
            structure.

            Enum members are **callable** and may be invoked to construct or dispatch
            the appropriate SED implementation, or to access regime-specific metadata
            such as break ordering and spectral slopes.

        Notes
        -----
        - The returned regime applies **globally** to the SED.
        - This method does not depend on the frequency array ``nu``.
        """
        # Handle units and then coerce to log space
        F_norm = ensure_in_units(F_norm, "erg / (cm**2 * s * Hz)")
        nu_m = ensure_in_units(nu_m, "Hz")
        nu_c = ensure_in_units(nu_c, "Hz")
        nu_max = ensure_in_units(nu_max, "Hz")

        # Coerce to log space
        log_F_peak = np.log(F_norm)
        log_nu_m = np.log(nu_m)
        log_nu_c = np.log(nu_c)
        log_nu_max = np.log(nu_max)
        log_omega = np.log(omega)
        log_gamma_m = np.log(gamma_m)

        # Dispatch to the low-level regime computation
        regime, _ = self._compute_sed_regime(
            log_F_norm=log_F_peak,
            log_nu_m=log_nu_m,
            log_nu_c=log_nu_c,
            log_nu_max=log_nu_max,
            log_omega=log_omega,
            log_gamma_m=log_gamma_m,
        )
        return regime

    # ============================================================ #
    # Regime-Specific SED Kernel                                   #
    # ============================================================ #
    # noinspection PyCallingNonCallable
    def _log_opt_sed_from_regime(
        self,
        regime: "str",
        log_nu: "_ArrayLike",
        log_nu_m: float,
        log_nu_a: float,
        log_nu_c: float,
        log_nu_max: float,
        log_F_norm: float,
        log_nu_ac: float = None,
        p: float = 3.0,
        s: float = -1.0,
    ):
        r"""
        Evaluate the log-space SED for a fixed spectral regime.

        This method dispatches to the **regime-specific numerical kernel**
        corresponding to the supplied regime identifier and applies the overall
        flux normalization.

        All branching logic on spectral shape is confined to this method.

        Parameters
        ----------
        regime : enum
            Spectral regime identifier from
            :class:`~_SynchrotronSSACoolingSEDFunctions`.
        log_nu : array-like
            Natural logarithm of the evaluation frequencies.
        log_nu_m, log_nu_a, log_nu_c, log_nu_max : float
            Natural logarithms of the characteristic frequencies.
        log_F_norm : float
            Natural logarithm of the flux normalization corresponding to the *optically thin* equivalent
            emission at the dominant electron frequency. See
            :ref:`sed_normalization` for details on how to determine the appropriate normalization.
        log_nu_ac : float, optional
            Natural logarithm of the stratified SSA transition frequency.
        p : float
            Electron power-law index.
        s : float
            Smoothing parameter for spectral breaks.

        Returns
        -------
        array-like
            Natural logarithm of the SED evaluated at ``log_nu``.

        Notes
        -----
        - This method assumes the regime is **already validated**.
        - All SED kernels are evaluated in log-space for numerical stability.
        """
        # Given that we have the regime, we can dispatch to the appropriate function.
        if regime == "Spectrum1":
            return self.SPECTRUM_FUNCTIONS[regime](log_nu, log_nu_m, log_nu_a, log_nu_max, p, s) + log_F_norm
        elif regime == "Spectrum2":
            return (
                self.SPECTRUM_FUNCTIONS[regime](log_nu, log_nu_m, log_nu_a, log_nu_max, p, s)
                + log_F_norm
                - ((p - 1) / 2) * (log_nu_a - log_nu_m)
            )
        elif regime == "Spectrum3":
            return self.SPECTRUM_FUNCTIONS[regime](log_nu, log_nu_m, log_nu_c, log_nu_a, log_nu_max, p, s) + log_F_norm
        elif regime == "Spectrum4":
            return (
                self.SPECTRUM_FUNCTIONS[regime](log_nu, log_nu_m, log_nu_c, log_nu_a, log_nu_max, p, s)
                + log_F_norm
                - ((p - 1) / 2) * (log_nu_a - log_nu_m)
            )
        elif regime == "Spectrum5":
            return (
                self.SPECTRUM_FUNCTIONS[regime](
                    log_nu, log_nu_m, log_nu_c, log_nu_a, log_nu_ac or log_nu_a, log_nu_max, p, s
                )
                + log_F_norm
            )
        elif regime == "Spectrum6":
            return (
                self.SPECTRUM_FUNCTIONS[regime](log_nu, log_nu_m, log_nu_a, log_nu_ac or log_nu_a, log_nu_max, p, s)
                + log_F_norm
                - (1 / 2) * (log_nu_a - log_nu_c)
            )
        elif regime == "Spectrum7":
            return (
                self.SPECTRUM_FUNCTIONS[regime](log_nu, log_nu_m, log_nu_a, log_nu_max, p, s)
                + log_F_norm
                + (-1 / 2) * (log_nu_m - log_nu_c)
                + (-p / 2) * (log_nu_a - log_nu_m)
            )
        elif regime == "Spectrum8":
            return (
                self.SPECTRUM_FUNCTIONS[regime](log_nu, log_nu_m, log_nu_a, log_nu_max, p, s)
                + log_F_norm
                + (-1 / 2) * (log_nu_m - log_nu_c)
                + (-p / 2) * (log_nu_a - log_nu_m)
            )
        else:
            raise RuntimeError("Unrecognized regime in _log_opt_sed_from_regime.")

    def _log_opt_sed(
        self,
        log_nu: "_ArrayLike",
        log_nu_m: float,
        log_nu_c: float,
        log_nu_max: float,
        log_F_norm: float,
        log_omega: float,
        log_gamma_m: float,
        log_nu_ac: float = None,
        p: float = 3.0,
        s: float = -1.0,
    ):
        r"""
        Optimized log-space SED evaluation with regime determination.

        This method orchestrates the full SED evaluation by:

        1. Determining the global spectral regime,
        2. Computing the corresponding self-absorption frequency,
        3. Dispatching to the appropriate regime-specific kernel.

        All inputs are assumed to be in **natural logarithmic CGS units**.

        Returns
        -------
        array-like
            Natural logarithm of the synchrotron SED evaluated at ``log_nu``.
        """
        # Determine the regime
        regime, log_nu_a = self._compute_sed_regime(
            log_F_norm,
            log_nu_m,
            log_nu_c,
            log_nu_max,
            log_omega,
            log_gamma_m,
        )
        # Dispatch to the appropriate regime function
        return self._log_opt_sed_from_regime(
            regime,
            log_nu,
            log_nu_m,
            log_nu_a,
            log_nu_c,
            log_nu_max,
            log_F_norm,
            log_nu_ac,
            p,
            s,
        )

    def sed(
        self,
        nu: "_UnitBearingArrayLike",
        nu_m: "_UnitBearingScalarLike",
        nu_c: "_UnitBearingScalarLike",
        F_norm: "_UnitBearingScalarLike",
        nu_max: "_UnitBearingScalarLike" = np.inf,
        nu_ac: "_UnitBearingScalarLike" = None,
        omega: float = 4 * np.pi,
        gamma_m: float = 1,
        p: float = 3,
        s: float = -1,
    ):
        r"""
        Evaluate the synchrotron spectral energy distribution.

        This is the **primary user-facing interface** for computing the synchrotron
        SED. It performs unit validation, converts inputs to logarithmic CGS form,
        and dispatches to the optimized internal implementation.

        Parameters
        ----------
        nu : ~astropy.units.Quantity or float or array-like
            Frequencies at which to evaluate the SED.
        nu_m, nu_c, nu_max : ~astropy.units.Quantity or float
            Injection, cooling, and maximum synchrotron frequencies.
        F_norm : ~astropy.units.Quantity or float
            The normalization flux density corresponding to the *optically thin* equivalent
            emission at the dominant electron frequency. See :ref:`sed_normalization` for details on how
            to determine the appropriate normalization.
        nu_ac : ~astropy.units.Quantity or float, optional
            Stratified SSA transition frequency.
        omega : float
            Effective emission solid angle.
        gamma_m : float
            Minimum electron Lorentz factor.
        p : float
            Electron power-law index.
        s : float
            Spectral smoothing parameter.

        Returns
        -------
        astropy.units.Quantity
            Flux density :math:`F_\nu` evaluated at ``nu``.

        Notes
        -----
        - This is the **only method** that enforces units.
        - All internal calculations are performed in log-space.
        """
        # Enforce units on each of the inputs
        nu = ensure_in_units(nu, "Hz")
        nu_m = ensure_in_units(nu_m, "Hz")
        nu_c = ensure_in_units(nu_c, "Hz")
        F_norm = ensure_in_units(F_norm, "erg s^-1 cm^-2 Hz^-1")
        nu_max = ensure_in_units(nu_max, "Hz")
        nu_ac = ensure_in_units(nu_ac, "Hz") if nu_ac is not None else None

        # Dispatch to the optimized log-space SED
        log_sed = self._log_opt_sed(
            np.log(nu),
            np.log(nu_m),
            np.log(nu_c),
            np.log(nu_max),
            np.log(F_norm),
            np.log(omega),
            np.log(gamma_m),
            log_nu_ac=np.log(nu_ac) if nu_ac is not None else None,
            p=p,
            s=s,
        )

        return np.exp(log_sed) * u.erg / (u.s * u.cm**2 * u.Hz)

    # =========================================================== #
    # Closure Relations Implementation                            #
    # =========================================================== #
    # Here we implement the closure relations to go forward and backward
    # between the physics parameters and the phenomenological SED parameters.
    def _opt_from_physics_to_params(
        self,
        log_B: float,
        log_R: float,
        log_D_L: float,
        log_D_A: float,
        log_gamma_min: float,
        log_gamma_c: float,
        log_gamma_max: float = np.inf,
        p: float = 2.5,
        f_V: float = 1.0,
        f_A: float = 1.0,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        alpha: float = 1.0,
        gamma_bulk: float = 1.0,
        redshift: float = 0.0,
        pitch_average: bool = False,
    ):
        r"""
        Convert physical synchrotron source parameters to phenomenological SED parameters.

        This low-level method evaluates the SSA+cooling closure relation entirely in
        logarithmic CGS units. It computes the characteristic synchrotron frequencies,
        optically thin normalization, self-absorption frequency, and globally
        consistent synchrotron regime.

        Parameters
        ----------
        log_B : float
            Natural logarithm of the magnetic field strength (G).
        log_R : float
            Natural logarithm of the characteristic source radius (cm).
        log_D_L : float
            Natural logarithm of the luminosity distance (cm).
        log_D_A: float
            Natural logarithm of the angular diameter distance (cm).
        log_gamma_min : float
            Natural logarithm of the minimum electron Lorentz factor.
        log_gamma_c : float
            Natural logarithm of the cooling Lorentz factor.
        log_gamma_max : float, optional
            Natural logarithm of the maximum electron Lorentz factor.
        p : float, optional
            Electron power-law index.
        f_V : float, optional
            Volume filling factor.
        f_A: float, optional
            Area filling factor.
        epsilon_E : float, optional
            Fraction of post-shock energy in electrons.
        epsilon_B : float, optional
            Fraction of post-shock energy in magnetic fields.
        alpha : float, optional
            Pitch angle in radians when ``pitch_average=False``.
        gamma_bulk : float, optional
            Bulk Lorentz factor of the emitting region.
        redshift : float, optional
            Source redshift.
        pitch_average : bool, optional
            If ``True``, assume pitch-angle averaged synchrotron emissivity.

        Returns
        -------
        dict
            Dictionary containing phenomenological parameters:

            - ``F_norm``
            - ``F_peak``
            - ``nu_m``
            - ``nu_c``
            - ``nu_a``
            - ``nu_max``
            - ``nu_peak``
            - ``regime``
        """
        result = _log_normalize_powerlaw_sbpl_sed_ssa_cool(
            log_B=log_B,
            log_R=log_R,
            log_D_L=log_D_L,
            log_D_A=log_D_A,
            log_gamma_min=log_gamma_min,
            log_gamma_c=log_gamma_c,
            log_gamma_max=log_gamma_max,
            p=p,
            f_V=f_V,
            f_A=f_A,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            alpha=alpha,
            gamma_bulk=gamma_bulk,
            redshift=redshift,
            pitch_average=pitch_average,
        )
        return {
            "F_norm": np.exp(result["log_F_norm"]),
            "F_peak": np.exp(result["log_F_peak"]),
            "nu_m": np.exp(result["log_nu_m"]),
            "nu_c": np.exp(result["log_nu_c"]),
            "nu_a": np.exp(result["log_nu_a"]),
            "nu_max": np.exp(result["log_nu_max"]),
            "nu_peak": np.exp(result["log_nu_peak"]),
            "regime": result["regime"],
        }

    def from_physics_to_params(
        self,
        B: "_UnitBearingScalarLike",
        R: "_UnitBearingScalarLike",
        gamma_min: float,
        gamma_c: float,
        gamma_max: float = np.inf,
        p: float = 2.5,
        f_V: float = 1.0,
        f_A: float = 1.0,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        alpha: float = 1.0,
        gamma_bulk: float = 1.0,
        redshift: float = None,
        luminosity_distance: "_UnitBearingScalarLike" = None,
        angular_diameter_distance: "_UnitBearingScalarLike" = None,
        proper_distance: "_UnitBearingScalarLike" = None,
        cosmology: cosmo.Cosmology = None,
        pitch_average: bool = False,
    ):
        r"""
        Construct phenomenological SSA+cooling SED parameters from physical inputs.

        This is the public, unit-aware interface to the SSA+cooling closure relation.
        It resolves cosmological distance information, converts inputs to CGS,
        evaluates the low-level log-space closure, and returns unit-bearing
        phenomenological SED parameters.

        Parameters
        ----------
        B : float, array-like, or ~astropy.units.Quantity
            Magnetic field strength. Must be convertible to Gauss.
        R : float, array-like, or ~astropy.units.Quantity
            Characteristic source radius. Must be convertible to cm.
        gamma_min : float
            Minimum electron Lorentz factor :math:`\gamma_{\min}`.
        gamma_c : float
            Cooling Lorentz factor :math:`\gamma_c`.
        gamma_max : float, optional
            Maximum electron Lorentz factor :math:`\gamma_{\max}`.
        p : float, optional
            Electron power-law index.
        f_V : float, optional
            Volume filling factor.
        f_A: float, optional
            Area filling factor.
        epsilon_E : float, optional
            Fraction of post-shock internal energy carried by relativistic electrons.
        epsilon_B : float, optional
            Fraction of post-shock internal energy stored in magnetic fields.
        alpha : float, optional
            Electron pitch angle in radians. Ignored if ``pitch_average=True``.
        gamma_bulk : float, optional
            Bulk Lorentz factor of the emitting region.
        redshift : float, optional
            Cosmological redshift of the source.
        luminosity_distance : float, array-like, or ~astropy.units.Quantity, optional
            Luminosity distance to the source.
        angular_diameter_distance : float, array-like, or ~astropy.units.Quantity, optional
            Angular diameter distance to the source.
        proper_distance : float, array-like, or ~astropy.units.Quantity, optional
            Proper distance to the source.
        cosmology : ~astropy.cosmology.Cosmology, optional
            Cosmology used to resolve missing distance/redshift information.
        pitch_average : bool, optional
            If ``True``, use pitch-angle averaged synchrotron emissivity.

        Returns
        -------
        dict
            Dictionary containing unit-bearing phenomenological parameters:

            - ``F_norm``
            - ``nu_m``
            - ``nu_c``
            - ``nu_a``
            - ``nu_max``
            - ``nu_peak``
            - ``regime``

        Notes
        -----
        Cosmological inputs are resolved into a self-consistent luminosity distance
        and redshift using :func:`resolve_cosmological_distances`, matching the
        behavior of :class:`PowerLaw_SynchrotronSED`.

        The effective solid angle ``Omega`` is treated as a direct phenomenological
        input and is not inferred from the cosmology.
        """
        # ------------------------------------------------------------------ #
        # Resolve cosmological information.
        # ------------------------------------------------------------------ #
        cosmology = get_cosmology(cosmology=cosmology)
        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        D_L = distances["luminosity_distance"]
        D_A = distances["angular_diameter_distance"]
        redshift = distances["redshift"]

        # ------------------------------------------------------------------ #
        # Unit enforcement.
        # ------------------------------------------------------------------ #
        B = ensure_in_units(B, "G")
        R = ensure_in_units(R, "cm")
        D_L = ensure_in_units(D_L, "cm")
        D_A = ensure_in_units(D_A, "cm")

        # ------------------------------------------------------------------ #
        # Convert to logarithmic CGS units.
        # ------------------------------------------------------------------ #
        log_B = np.log(B)
        log_R = np.log(R)
        log_D_L = np.log(D_L)
        log_D_A = np.log(D_A)

        log_gamma_min = np.log(gamma_min)
        log_gamma_c = np.log(gamma_c)
        log_gamma_max = np.log(gamma_max)

        # ------------------------------------------------------------------ #
        # Dispatch to the low-level closure.
        # ------------------------------------------------------------------ #
        params_log = self._opt_from_physics_to_params(
            log_B=log_B,
            log_R=log_R,
            log_D_L=log_D_L,
            log_D_A=log_D_A,
            log_gamma_min=log_gamma_min,
            log_gamma_c=log_gamma_c,
            log_gamma_max=log_gamma_max,
            p=p,
            f_V=f_V,
            f_A=f_A,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            alpha=alpha,
            gamma_bulk=gamma_bulk,
            redshift=redshift,
            pitch_average=pitch_average,
        )

        # ------------------------------------------------------------------ #
        # Convert back to physical units.
        # ------------------------------------------------------------------ #
        return {
            "F_norm": params_log["F_norm"] * u.erg / (u.s * u.cm**2 * u.Hz),
            "F_peak": params_log["F_peak"] * u.erg / (u.s * u.cm**2 * u.Hz),
            "nu_m": params_log["nu_m"] * u.Hz,
            "nu_c": params_log["nu_c"] * u.Hz,
            "nu_a": params_log["nu_a"] * u.Hz,
            "nu_max": params_log["nu_max"] * u.Hz,
            "nu_peak": params_log["nu_peak"] * u.Hz,
            "regime": params_log["regime"],
        }

    def _opt_from_params_to_physics(
        self,
        regime: str,
        log_F_peak: "_ArrayLike",
        log_nu_peak: "_ArrayLike",
        log_D_L: "_ArrayLike",
        log_D_A: "_ArrayLike" = None,
        gamma_min: float = 1.0,
        gamma_c: float = np.inf,
        gamma_max: float = np.inf,
        p: float = 3.0,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        f_V: float = 0.5,
        f_A: float = 0.5,
        gamma_bulk: float = 1.0,
        redshift: float = 0.0,
        alpha: float = None,
        pitch_average: bool = False,
    ):
        r"""
        Low-level inversion of synchrotron peak observables to physical parameters.

        This method implements the **log-space inverse closure relation** for the
        synchrotron SED including **radiative cooling and synchrotron
        self-absorption (SSA)**.

        It maps observed spectral peak quantities

        - :math:`F_{\nu,\mathrm{pk}}`
        - :math:`\nu_{\mathrm{pk}}`

        to the corresponding **source radius** :math:`R` and **magnetic field**
        :math:`B` under a specific synchrotron spectral regime.

        This routine is the optimized internal implementation used by
        :meth:`from_params_to_physics`.  All inputs must already be expressed in
        **natural logarithmic CGS units**, and no unit validation is performed.

        .. important::

            Unlike the simpler optically-thin synchrotron SED, **the inversion is not
            unique without specifying the spectral regime**.  Different SSA and
            cooling regimes correspond to different relationships between
            observables and physical parameters.

            For this reason the caller must explicitly provide the **regime
            identifier**.

        .. warning::

            This method implements a **closure relation**, not a direct physical
            measurement.  The inferred values of :math:`R` and :math:`B` are those
            required to reproduce the supplied observables **under the assumptions
            of the chosen regime and closure prescription**.

            If the supplied regime does not match the true spectral configuration,
            the inferred parameters may be significantly incorrect.

        Parameters
        ----------
        regime : str
            Identifier for the synchrotron spectral regime.  Must correspond to one
            of the regimes defined in :attr:`SPECTRUM_INVERSION_FUNCTIONS`.

            Each regime encodes a specific ordering of the characteristic
            synchrotron frequencies (e.g. :math:`\nu_a`, :math:`\nu_m`,
            :math:`\nu_c`, :math:`\nu_{\max}`) and therefore determines the
            analytic inversion formula used.

        log_F_peak : array-like
            Natural logarithm of the observed peak flux density
            :math:`F_{\nu,\mathrm{pk}}` in CGS units
            (:math:`\mathrm{erg\,cm^{-2}\,s^{-1}\,Hz^{-1}}`).

        log_nu_peak : array-like
            Natural logarithm of the observed peak frequency (Hz).

        log_D_L : array-like
            Natural logarithm of the luminosity distance (cm).

        log_D_A : array-like, optional
            Natural logarithm of the angular-diameter distance (cm).

            This quantity is required for regimes where the spectral peak occurs
            in the **optically thick SSA portion of the spectrum**, because those
            inversions depend explicitly on the **angular size of the emitting
            region**.

        gamma_min : float, optional
            Minimum Lorentz factor of the accelerated electron distribution.

        gamma_c : float, optional
            Cooling Lorentz factor.

            Must be finite for regimes where cooling explicitly shapes the
            spectrum.

        gamma_max : float, optional
            Maximum electron Lorentz factor.

        p : float, optional
            Power-law index of the electron energy distribution.

        epsilon_E : float, optional
            Fraction of post-shock internal energy carried by relativistic
            electrons.

        epsilon_B : float, optional
            Fraction of post-shock internal energy stored in magnetic fields.

        f_V : float, optional
            Volume filling factor of the emitting region.

        f_A : float, optional
            Area filling factor used in SSA inversions where the emitting surface
            area must be specified.

        gamma_bulk : float, optional
            Bulk Lorentz factor of the emitting region.

            A simple on-axis Doppler transformation is applied to convert the
            observed peak quantities into the comoving frame prior to inversion.

        redshift : float, optional
            Cosmological redshift of the source.

        alpha : float, optional
            Electron pitch angle in radians.

            Required if ``pitch_average=False``.

        pitch_average : bool, optional
            If ``True``, the pitch-angle-averaged synchrotron emissivity is used.

        Returns
        -------
        dict
            Dictionary containing inferred physical parameters

            - ``R`` : source radius (cm)
            - ``B`` : magnetic field strength (G)

            Values are returned in **linear CGS units**.

        Notes
        -----
        **Relativistic corrections**

        Observed quantities are transformed into the comoving frame using

        .. math::

            \nu' = \nu \frac{1+z}{\delta}

            F' = F \frac{1+z}{\delta^3}

        assuming on-axis emission with Doppler factor
        :math:`\delta(\Gamma, \theta=0)`.

        **Closure assumptions**

        The inversion assumes

        - a single homogeneous emission zone,
        - synchrotron emission from a power-law electron population,
        - equipartition-style microphysical closure,
        - a spherical or quasi-spherical emitting geometry,
        - on-axis relativistic boosting if bulk motion is present.

        **Why regimes matter**

        Different regimes correspond to different peak-forming mechanisms:

        - optically thin emission near :math:`\nu_m`
        - optically thick SSA peaks
        - cooling-dominated peaks

        Each of these produces **different scalings of**
        :math:`R` and :math:`B` with the observables.

        See Also
        --------
        from_params_to_physics
            Public, unit-aware wrapper for this inversion.
        determine_sed_regime
            Utility for determining the appropriate spectral regime.
        """
        # ------------------------------------------------------------------ #
        # Pitch-angle treatment
        # ------------------------------------------------------------------ #
        if pitch_average:
            if alpha is not None:
                warnings.warn(
                    "Pitch averaging is enabled, so the provided `alpha` is ignored. "
                    "(Set `alpha=None` to suppress this warning.)",
                    stacklevel=2,
                )
            sin_alpha = None
        else:
            if alpha is None:
                raise ValueError("Pitch averaging is disabled, so a fixed pitch angle must be provided via `alpha`.")
            sin_alpha = np.sin(alpha)

        # ------------------------------------------------------------------ #
        # Handle the regime-specification
        # ------------------------------------------------------------------ #
        if regime not in self.SPECTRUM_INVERSION_FUNCTIONS:
            raise ValueError(
                f"Unrecognized SED regime: {regime}."
                f" Available regimes: {list(self.SPECTRUM_INVERSION_FUNCTIONS.keys())}"
            )

        inversion_function = self.SPECTRUM_INVERSION_FUNCTIONS[regime]
        # ------------------------------------------------------------------ #
        # Transform observed peak quantities into the comoving frame.
        #
        # This applies the same minimal relativistic treatment used elsewhere
        # in the default closure: an on-axis Doppler correction plus the usual
        # cosmological redshift correction.
        #
        # IMPORTANT: this is intentionally simple and is not a substitute for a
        # full relativistic radiative-transfer treatment.
        # ------------------------------------------------------------------ #
        doppler_factor = compute_doppler_factor(gamma_bulk, 0.0)

        log_nu_peak_comoving = log_nu_peak - np.log(doppler_factor) + np.log1p(redshift)
        log_F_peak_comoving = log_F_peak - 3.0 * np.log(doppler_factor) + np.log1p(redshift)

        # ------------------------------------------------------------------ #
        # Invert the default closure in the comoving frame.
        # ------------------------------------------------------------------ #
        if regime == "Spectrum1":  # nu_a < nu_m < nu_max < nu_c.
            log_R, log_B = inversion_function(
                log_nu_peak_comoving,
                log_F_peak_comoving,
                log_D_L,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                p=p,
                epsilon_e=epsilon_E,
                epsilon_B=epsilon_B,
                f_V=f_V,
                sin_alpha=sin_alpha,
            )
        elif regime == "Spectrum2":  # nu_m < nu_a < nu_max < nu_c.
            if log_D_A is None:
                raise ValueError("Angular diameter distance `D_A` must be provided for Spectrum2 regime.")

            log_R, log_B = inversion_function(
                log_nu_peak_comoving,
                log_F_peak_comoving,
                log_D_L,
                log_D_A,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                p=p,
                epsilon_e=epsilon_E,
                epsilon_B=epsilon_B,
                f_V=f_V,
                f_A=f_A,
                sin_alpha=sin_alpha,
            )
        elif regime == "Spectrum3":  # nu_a < nu_m < nu_c < nu_max.
            log_R, log_B = inversion_function(
                log_nu_peak_comoving,
                log_F_peak_comoving,
                log_D_L,
                gamma_min=gamma_min,
                gamma_c=gamma_c,
                gamma_max=gamma_max,
                p=p,
                epsilon_e=epsilon_E,
                epsilon_B=epsilon_B,
                f_V=f_V,
                sin_alpha=sin_alpha,
            )
        elif regime == "Spectrum4":  # nu_m < nu_a < nu_c < nu_max.
            if log_D_A is None:
                raise ValueError("Angular diameter distance `D_A` must be provided for Spectrum4 regime.")
            if not np.all(np.isfinite(gamma_c)):
                raise ValueError("Cooling Lorentz factor `gamma_c` must be finite for Spectrum4 regime.")

            log_R, log_B = inversion_function(
                log_nu_peak_comoving,
                log_F_peak_comoving,
                log_D_L,
                log_D_A,
                gamma_min=gamma_min,
                gamma_c=gamma_c,
                gamma_max=gamma_max,
                p=p,
                epsilon_e=epsilon_E,
                epsilon_B=epsilon_B,
                f_V=f_V,
                f_A=f_A,
                sin_alpha=sin_alpha,
            )
        elif regime == "Spectrum5":  # nu_a < nu_c < nu_m < nu_max.
            if not np.all(np.isfinite(gamma_c)):
                raise ValueError("Cooling Lorentz factor `gamma_c` must be finite for Spectrum5 regime.")

            log_R, log_B = inversion_function(
                log_nu_peak_comoving,
                log_F_peak_comoving,
                log_D_L,
                gamma_min=gamma_min,
                gamma_c=gamma_c,
                gamma_max=gamma_max,
                p=p,
                epsilon_e=epsilon_E,
                epsilon_B=epsilon_B,
                f_V=f_V,
                sin_alpha=sin_alpha,
            )
        elif regime == "Spectrum6":  # nu_c < nu_a < nu_m < nu_max.
            if log_D_A is None:
                raise ValueError("Angular diameter distance `D_A` must be provided for Spectrum6 regime.")
            if not np.all(np.isfinite(gamma_c)):
                raise ValueError("Cooling Lorentz factor `gamma_c` must be finite for Spectrum6 regime.")

            log_R, log_B = inversion_function(
                log_nu_peak_comoving,
                log_F_peak_comoving,
                log_D_L,
                log_D_A,
                gamma_min=gamma_min,
                gamma_c=gamma_c,
                gamma_max=gamma_max,
                p=p,
                epsilon_e=epsilon_E,
                epsilon_B=epsilon_B,
                f_V=f_V,
                f_A=f_A,
                sin_alpha=sin_alpha,
            )
        elif regime == "Spectrum7":  # nu_c, nu_m < nu_a < nu_max.
            if log_D_A is None:
                raise ValueError("Angular diameter distance `D_A` must be provided for Spectrum7 regime.")
            if not np.all(np.isfinite(gamma_c)):
                raise ValueError("Cooling Lorentz factor `gamma_c` must be finite for Spectrum7 regime.")

            log_R, log_B = inversion_function(
                log_nu_peak_comoving,
                log_F_peak_comoving,
                log_D_L,
                log_D_A,
                gamma_min=gamma_min,
                gamma_c=gamma_c,
                gamma_max=gamma_max,
                p=p,
                epsilon_e=epsilon_E,
                epsilon_B=epsilon_B,
                f_V=f_V,
                f_A=f_A,
                sin_alpha=sin_alpha,
            )
        else:
            raise ValueError(
                f"Unrecognized SED regime: {regime}."
                f" Available regimes: {list(self.SPECTRUM_INVERSION_FUNCTIONS.keys())}"
            )

        return {
            "R": np.exp(log_R),
            "B": np.exp(log_B),
        }

    def from_params_to_physics(
        self,
        regime: str,
        F_peak: "_UnitBearingScalarLike",
        nu_peak: "_UnitBearingScalarLike",
        gamma_min: float = 1.0,
        gamma_c: float = np.inf,
        gamma_max: float = np.inf,
        p: float = 3.0,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        f_V: float = 0.5,
        f_A: float = 0.5,
        gamma_bulk: float = 1.0,
        redshift: float = None,
        luminosity_distance: "_UnitBearingScalarLike" = None,
        angular_diameter_distance: "_UnitBearingScalarLike" = None,
        proper_distance: "_UnitBearingScalarLike" = None,
        cosmology: cosmo.Cosmology = None,
        alpha: float = None,
        pitch_average: bool = False,
    ):
        r"""
        Infer physical source parameters from observed synchrotron peak quantities.

        This method provides the **public, unit-aware interface** for the inverse
        synchrotron closure relation implemented by
        :meth:`_opt_from_params_to_physics`.

        It converts observed peak quantities

        - peak flux density :math:`F_{\nu,\mathrm{pk}}`
        - peak frequency :math:`\nu_{\mathrm{pk}}`

        into estimates of the **source radius** :math:`R` and **magnetic field**
        :math:`B`, assuming the emission follows one of the spectral regimes
        described by :class:`PowerLaw_Cooling_SSA_SynchrotronSED`.

        The calculation proceeds by

        1. resolving cosmological distance information,
        2. converting all quantities into CGS units,
        3. transforming observables into logarithmic form,
        4. dispatching to the optimized inversion routine.

        .. important::

            The caller **must specify the synchrotron spectral regime**.  This
            inversion cannot determine the regime automatically because multiple
            spectral configurations may produce identical peak observables.

        .. warning::

            The inferred parameters depend strongly on the assumed regime and
            closure relation.  Incorrect regime identification will generally lead
            to incorrect physical parameters.

        Parameters
        ----------
        regime : str
            Identifier of the synchrotron spectral regime.

            The spectrum is identified as ``"SpectrumN"``, where ``N`` is an
            integer corresponding to the regimes defined in
            :ref:`synch_sed_theory` and implemented internally in
            :attr:`SPECTRUM_INVERSION_FUNCTIONS`.

            Each regime corresponds to a specific ordering of the characteristic
            synchrotron frequencies (e.g. :math:`\nu_a`, :math:`\nu_m`,
            :math:`\nu_c`, :math:`\nu_{\max}`) and therefore determines the
            analytic inversion formula used to infer the physical parameters.
            The user must ensure that the supplied regime is consistent with the
            observed spectral configuration.

        F_peak : ~numpy.ndarray or float or ~astropy.units.Quantity
            Observed (lab-frame) peak flux density.

            Must be convertible to
            :math:`\mathrm{erg\,cm^{-2}\,s^{-1}\,Hz^{-1}}`.

            If an array is provided, the inversion is performed **element-wise**
            and the returned physical parameters will have the same shape.
            This makes it possible to evaluate the closure relation across
            posterior samples, Monte Carlo realizations, or time-series data.

        nu_peak : ~numpy.ndarray or float or ~astropy.units.Quantity
            Observed (lab-frame) peak frequency.

            Must be convertible to Hz.

            As with ``F_peak``, array inputs are supported and will produce
            element-wise solutions for the inferred parameters.

        gamma_min : float, optional
            Minimum Lorentz factor of the accelerated electron distribution.

            By default this is ``1.0``, corresponding to a non-relativistic
            minimum electron energy. In many astrophysical applications
            (e.g. shock acceleration), values of
            :math:`\gamma_{\min} \sim 10-10^3` may be more physically
            appropriate.
        gamma_c: float, optional
            Cooling Lorentz factor.

            This parameter is relevant for regimes where radiative cooling explicitly shapes the spectrum.
            It represents the Lorentz factor at which electrons cool on the dynamical timescale of the system.
            If cooling is negligible, this can be set to a very large value (e.g. ``np.inf``) to effectively
            remove its influence from the inversion.

        gamma_max : float, optional
            Maximum Lorentz factor of the accelerated electron distribution.

            The default value of ``np.inf`` corresponds to no high-energy
            cutoff. This assumption is typically valid provided that the
            synchrotron cutoff frequency lies well above the observed band.

        p : float, optional
            Power-law index of the electron Lorentz factor distribution,

            .. math::

                N(\gamma) \propto \gamma^{-p}.

            The default value ``p = 3.0`` is commonly adopted for
            shock-accelerated electron populations.

        epsilon_E : float, optional
            Fraction of the post-shock internal energy carried by relativistic
            electrons.

            This parameter enters the equipartition-style closure relation used
            to normalize the electron distribution.

        epsilon_B : float, optional
            Fraction of the post-shock internal energy stored in magnetic
            fields.

            Together with ``epsilon_E``, this parameter determines the relative
            partition of energy between particles and magnetic fields in the
            emitting region.

        f_V : float, optional
            Effective **volume filling factor** of the emitting region.

            In optically thin emission scenarios, this parameter represents the
            fraction of the spherical volume that actively emits synchrotron
            radiation.

            Physically, this accounts for situations where the emission arises
            from a thin shell or clumpy medium rather than uniformly filling
            the entire volume.

            For example:

            - ``f_V = 1`` corresponds to a uniformly filled sphere.
            - ``f_V < 1`` may represent emission from a thin shocked shell or
              fragmented ejecta.

        f_A : float, optional
            Effective **area filling factor** of the emitting region.

            This parameter becomes relevant in regimes where the spectral peak
            is produced by **synchrotron self-absorption (SSA)**. In such cases,
            the inversion depends on the projected emitting area rather than
            the full volume.

            Physically, ``f_A`` represents the fraction of the projected
            spherical surface that contributes to the optically thick emission.

            For example:

            - ``f_A = 1`` corresponds to a uniformly emitting spherical surface.
            - ``f_A < 1`` may arise from patchy or filamentary emission regions.

            In many simple models one may take ``f_A ≈ f_V``, but the two
            parameters are conceptually distinct and correspond to **different
            geometric aspects** of the emitting region.

        gamma_bulk : float, optional
            Bulk Lorentz factor of the emitting region.

            If non-unity, the observed peak quantities are transformed into the
            comoving frame using relativistic Doppler corrections prior to
            performing the inversion.

            The implementation assumes **on-axis emission**, corresponding to a
            Doppler factor

            .. math::

                \delta(\Gamma, \theta=0).

        redshift : float, optional
            Cosmological redshift of the source.

            This parameter enters the conversion between observed and comoving
            frequencies and fluxes and also determines the cosmological
            distances if they are not supplied explicitly.

        luminosity_distance : quantity, optional
            Luminosity distance to the source.

            Must be convertible to cm.

            If supplied, this value is used directly in the inversion.

        angular_diameter_distance : quantity, optional
            Angular-diameter distance to the source.

            Must be convertible to cm.

            This quantity is required in some SSA regimes where the inversion
            depends explicitly on the **angular size of the emitting region**.

        proper_distance : quantity, optional
            Proper distance to the source.

            This can be used as an alternative specification of the cosmological
            distance.

        cosmology : ~astropy.cosmology.Cosmology, optional
            Cosmology used to compute missing distance quantities.

            The distances required by the inversion are resolved using the
            helper function :func:`resolve_cosmological_distances`, which
            processes the supplied cosmological parameters as follows:

            1. If explicit distances are provided (e.g. ``luminosity_distance``),
               those values are used directly.

            2. If only a redshift is provided, the necessary distances are
               computed using the supplied ``cosmology``.

            3. If no cosmology is provided, the default cosmology returned by
               :func:`get_cosmology` is used.

            This mechanism ensures that the inversion has access to all
            required cosmological distance measures while allowing users to
            specify whichever subset of parameters is most convenient.

        alpha : float, optional
            Electron pitch angle in radians.

            This parameter is required when ``pitch_average=False`` and
            specifies the angle between the electron velocity and the magnetic
            field.

        pitch_average : bool, optional
            If ``True``, the synchrotron emissivity is averaged over an
            isotropic distribution of pitch angles.

            In this case the explicit pitch angle ``alpha`` is ignored and the
            appropriate angle-averaged emissivity coefficients are used.

            Pitch-angle averaging is often appropriate when the electron
            distribution is isotropic in momentum space.

        Returns
        -------
        dict
            Dictionary containing inferred source parameters

            - ``R`` : source radius (cm)
            - ``B`` : magnetic field strength (G)

        Notes
        -----
        **Interpretation**

        These values are **model-dependent estimates**, not direct measurements.

        They represent the values of :math:`R` and :math:`B` that reproduce the
        observed peak under the assumptions of

        - the selected synchrotron spectral regime,
        - a single-zone homogeneous emitting region,
        - a power-law electron population,
        - equipartition-style microphysical closure.

        **When this method may fail**

        The inversion may produce misleading results if

        - the true spectrum contains multiple emitting zones,
        - the peak is produced by a process not described by the selected regime,
        - strong radiative transfer effects modify the spectrum,
        - relativistic equal-arrival-time surfaces dominate the emission,
        - the geometry deviates strongly from the assumed configuration.

        In such cases, users should treat the inferred parameters as
        **order-of-magnitude estimates only**.

        See Also
        --------
        determine_sed_regime
            Identify the synchrotron spectral regime.
        from_physics_to_params
            Forward closure relation.
        sed
            Evaluate the SED for a given parameter set.
        """
        # ------------------------------------------------------------------ #
        # Resolve cosmological information.
        # ------------------------------------------------------------------ #
        cosmology = get_cosmology(cosmology=cosmology)
        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        D_L = distances["luminosity_distance"]
        D_A = distances["angular_diameter_distance"]
        redshift = distances["redshift"]

        # ------------------------------------------------------------------ #
        # Unit enforcement and conversion to internal log-space representation.
        # ------------------------------------------------------------------ #
        log_F_peak = np.log(ensure_in_units(F_peak, "erg cm^-2 s^-1 Hz^-1"))
        log_nu_peak = np.log(ensure_in_units(nu_peak, "Hz"))
        log_D_L = np.log(ensure_in_units(D_L, "cm"))
        log_D_A = np.log(ensure_in_units(D_A, "cm"))

        # ------------------------------------------------------------------ #
        # Dispatch to the low-level inverse closure.
        # ------------------------------------------------------------------ #
        results = self._opt_from_params_to_physics(
            regime=regime,
            log_F_peak=log_F_peak,
            log_nu_peak=log_nu_peak,
            log_D_L=log_D_L,
            log_D_A=log_D_A,
            gamma_min=gamma_min,
            gamma_c=gamma_c,
            gamma_max=gamma_max,
            p=p,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            f_A=f_A,
            gamma_bulk=gamma_bulk,
            redshift=redshift,
            alpha=alpha,
            pitch_average=pitch_average,
        )

        return {
            "R": results["R"] * u.cm,
            "B": results["B"] * u.G,
        }


class PowerLaw_Cooling_SynchrotronSED(MultiSpectrumSynchrotronSED):
    r"""
    Synchrotron SED with cooling but no self-absorption.

    This class implements the **canonical optically thin synchrotron spectral
    energy distribution** produced by a power-law electron energy distribution
    subject to radiative cooling, **without synchrotron self-absorption (SSA)**.

    It is appropriate for emission regions where the synchrotron self-absorption
    turnover frequency :math:`\nu_a` lies well below the lowest frequency of
    interest, and the emitting plasma can be treated as homogeneous.

    The spectrum is constructed using **smoothed broken power laws (SBPLs)** and
    supports all standard cooling regimes encountered in synchrotron theory.
    Regime selection is **global** and based entirely on the ordering of the
    characteristic break frequencies.

    For a detailed theoretical derivation of the spectral slopes and normalization
    conventions, see :ref:`synchrotron_theory`.

    .. rubric:: Spectral Structure

    The synchrotron spectrum is classified into one of three cooling regimes
    based on the ordering of the characteristic frequencies:

    - Injection frequency :math:`\nu_m`,
    - Cooling frequency :math:`\nu_c`,
    - Maximum synchrotron frequency :math:`\nu_{\max}`.

    The supported regimes are:

    **Fast cooling** (:math:`\nu_c < \nu_m`)

    .. math::

        F_\nu \propto
        \begin{cases}
            \nu^{1/3}, & \nu < \nu_c \
            \nu^{-1/2}, & \nu_c < \nu < \nu_m \
            \nu^{-p/2}, & \nu > \nu_m
        \end{cases}

    **Slow cooling** (:math:`\nu_m < \nu_c < \nu_{\max}`)

    .. math::

        F_\nu \propto
        \begin{cases}
            \nu^{1/3}, & \nu < \nu_m \
            \nu^{-(p-1)/2}, & \nu_m < \nu < \nu_c \
            \nu^{-p/2}, & \nu > \nu_c
        \end{cases}

    **Effectively non-cooling** (:math:`\nu_c > \nu_{\max}`)

    .. math::

        F_\nu \propto
        \begin{cases}
            \nu^{1/3}, & \nu < \nu_m \
            \nu^{-(p-1)/2}, & \nu > \nu_m
        \end{cases}

    The selected regime applies **globally** to the spectrum and does not depend
    on the frequency grid used for evaluation.

    .. rubric:: Parameters

    The parameters entering this SED fall into three conceptual categories.

    .. tab-set::

        .. tab-item:: Free parameters (phenomenological)

            These parameters define the observable structure of the SED and are
            typically inferred directly from broadband data.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - Flux normalization
                  - :math:`F_{\nu,\mathrm{norm}}`
                  - Flux density normalization corresponding to the *optically thin* equivalent
                    emission at the dominant electron frequency. See :ref:`sed_normalization` for details
                    on how to determine the appropriate normalization.
                * - Injection frequency
                  - :math:`\nu_m`
                  - Synchrotron frequency of minimum-energy electrons
                * - Cooling frequency
                  - :math:`\nu_c`
                  - Frequency corresponding to the cooling Lorentz factor
                * - Maximum frequency
                  - :math:`\nu_{\max}`
                  - High-energy synchrotron cutoff frequency

        .. tab-item:: Hyper-parameters

            These parameters control the *shape* and smoothness of the spectrum
            but are not usually tightly constrained by broadband observations.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - Electron power-law index
                  - :math:`p`
                  - Index of the injected electron energy distribution
                * - Smoothing parameter
                  - :math:`s`
                  - Controls the sharpness of spectral breaks
                * - Minimum Lorentz factor
                  - :math:`\gamma_m`
                  - Minimum electron Lorentz factor (used in normalization)

        .. tab-item:: Derived quantities (internal)

            These quantities are **not user inputs**, but are determined internally
            from the supplied parameters.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Quantity
                  - Symbol
                  - Description
                * - Cooling regime
                  - —
                  - Fast, slow, or effectively non-cooling classification
                * - Regime identifier
                  - —
                  - Discrete index selecting the appropriate SED kernel

    .. rubric:: Normalization and Closure

    The SED is normalized using :math:`F_{\nu,\mathrm{norm}}`, defined as the flux
    density of the dominant optically thin emitting electron population. The
    frequency at which this population emits depends on the cooling regime.

    In the fast-cooling regime, normalization is anchored at the cooling frequency
    :math:`\nu_c`; in the slow- and non-cooling regimes, normalization is anchored
    at the injection frequency :math:`\nu_m`.

    Optional equipartition-based closure relations are provided to map physical
    model parameters onto the phenomenological SED parameters.

    See Also
    --------
    :class:`SynchrotronSED` : Base class for synchrotron SED implementations.
    :class:`MultiSpectrumSynchrotronSED` : Base class for multi-regime synchrotron SEDs.
    :class:`PowerLaw_SynchrotronSED` : Synchrotron SED without cooling or self-absorption.
    :class:`PowerLaw_SSA_SynchrotronSED` : Synchrotron SED with synchrotron self-absorption.
    :class:`PowerLaw_Cooling_SSA_SynchrotronSED` : Synchrotron SED with cooling and self-absorption.
    """

    SPECTRUM_FUNCTIONS = COOLING_SED_FUNCTION_REGISTRY
    SPECTRUM_INVERSION_FUNCTIONS = COOLING_INV_FUNCTION_REGISTRY

    # ------------------------------------------------------------ #
    # Regime determination                                        #
    # ------------------------------------------------------------ #
    def _compute_sed_regime(self, log_nu_m: float, log_nu_c: float, log_nu_max: float):
        r"""
        Determine the optically thin synchrotron cooling regime.

        This low-level method classifies the synchrotron spectrum based on the
        ordering of the characteristic frequencies in log-space:

        .. math::

            \nu_m \quad \text{(injection frequency)}, \qquad
            \nu_c \quad \text{(cooling frequency)}, \qquad
            \nu_{\max} \quad \text{(high-frequency cutoff)}.

        The regime determination is **global** and applies to the entire SED;
        it does not depend on the evaluation frequency array ``nu``.

        Parameters
        ----------
        log_nu_m : float
            Natural logarithm of the injection frequency :math:`\nu_m`.
        log_nu_c : float
            Natural logarithm of the cooling frequency :math:`\nu_c`.
        log_nu_max : float
            Natural logarithm of the maximum synchrotron frequency
            :math:`\nu_{\max}`.

        Returns
        -------
        regime : enum-like
            A member of :attr:`SPECTRUM_FUNCTIONS` identifying the cooling regime.
        metadata : dict
            Empty dictionary (present for API consistency with multi-regime SEDs
            that require auxiliary derived parameters).

        Notes
        -----
        The regimes correspond to the following orderings:

        - **Fast cooling**: :math:`\nu_c < \nu_m`
        - **Slow cooling**: :math:`\nu_m < \nu_c < \nu_{\max}`
        - **Non-cooling**: :math:`\nu_c > \nu_{\max}`

        This method assumes that all inputs are already in log-space and does
        not perform unit validation.
        """
        if log_nu_c <= log_nu_m:
            regime = "fast_cooling"
        elif log_nu_m < log_nu_c < log_nu_max:
            regime = "slow_cooling"
        elif log_nu_c > log_nu_max:
            regime = "no_cooling"
        else:
            raise RuntimeError("Could not determine cooling regime.")

        # Now complete, log that we have completed the step, and return.
        if triceratops_logger.isEnabledFor(logging.DEBUG):
            triceratops_logger.debug(
                "PowerLaw_Cooling_SSA_SynchrotronSED: selected regime.",
                extra={
                    "log_nu_m": log_nu_m,
                    "log_nu_c": log_nu_c,
                    "log_nu_max": log_nu_max,
                    "regime": regime,
                },
            )

        return regime, {}

    def determine_sed_regime(
        self, nu_m: "_UnitBearingScalarLike", nu_c: "_UnitBearingScalarLike", nu_max: "_UnitBearingScalarLike" = np.inf
    ):
        r"""
        Public interface for determining the synchrotron cooling regime.

        This method provides a user-facing wrapper around
        :meth:`_compute_sed_regime`. It handles unit validation and conversion
        to log-space before performing regime classification.

        Parameters
        ----------
        nu_m : ~astropy.units.Quantity or float
            Injection frequency :math:`\nu_m`.
        nu_c : ~astropy.units.Quantity or float
            Cooling frequency :math:`\nu_c`.
        nu_max : ~astropy.units.Quantity or float, optional
            Maximum synchrotron frequency :math:`\nu_{\max}`.
            Defaults to :math:`\infty`.

        Returns
        -------
        regime : enum-like
            Identifier specifying the **global cooling regime** of the synchrotron
            spectrum, determined by the ordering of :math:`\nu_m`, :math:`\nu_c`,
            and :math:`\nu_{\max}`.

            The returned value is a member of
            :attr:`PowerLaw_Cooling_SynchrotronSED.SPECTRUM_FUNCTIONS` and uniquely
            selects the corresponding optically thin spectral kernel.

        Notes
        -----
        - This method does **not** depend on the frequency array used to
          evaluate the SED.
        - The returned regime must be consistent with the spectrum produced
          by :meth:`sed` for the same parameters.
        """
        nu_m = ensure_in_units(nu_m, "Hz")
        nu_c = ensure_in_units(nu_c, "Hz")
        nu_max = ensure_in_units(nu_max, "Hz")

        regime, _ = self._compute_sed_regime(
            log_nu_m=np.log(nu_m),
            log_nu_c=np.log(nu_c),
            log_nu_max=np.log(nu_max),
        )

        return regime

    # ------------------------------------------------------------ #
    # Regime-specific kernel                                       #
    # ------------------------------------------------------------ #
    def _log_opt_sed_from_regime(
        self,
        log_nu: float,
        regime: Callable,
        log_nu_m: float,
        log_nu_c: float,
        log_nu_max: float,
        log_F_norm: float,
        p: float,
        s: float,
    ):
        r"""
        Evaluate the log-space synchrotron SED for a fixed cooling regime.

        This method implements the **numerical kernel** for computing the
        synchrotron SED once the cooling regime has been determined. All
        branching on the regime identifier occurs here.

        Parameters
        ----------
        log_nu : array-like
            Natural logarithm of the frequencies at which to evaluate the SED.
        regime : enum-like
            Cooling regime identifier returned by
            :meth:`_compute_sed_regime`.
        log_nu_m : float
            Natural logarithm of the injection frequency :math:`\nu_m`.
        log_nu_c : float
            Natural logarithm of the cooling frequency :math:`\nu_c`.
        log_nu_max : float
            Natural logarithm of the maximum synchrotron frequency
            :math:`\nu_{\max}`.
        log_F_norm : float
            Natural logarithm of the flux density normalization corresponding to
            the *optically thin* equivalent emission at the dominant electron frequency.
        p : float
            Electron energy power-law index.
        s : float
            SBPL smoothness parameter.

        Returns
        -------
        log_sed : array-like
            Natural logarithm of the flux density
            :math:`\log F_\nu` evaluated at ``log_nu``.

        Notes
        -----
        - The returned spectrum is **not normalized** until ``log_F_peak``
          is added.
        - This method assumes an optically thin emitting region and does not
          include synchrotron self-absorption.
        - No unit validation is performed.
        """
        if regime == "fast_cooling":
            log_sed = self.SPECTRUM_FUNCTIONS["fast_cooling"](log_nu, log_nu_m, log_nu_c, log_nu_max, p, s)
        elif regime == "slow_cooling":
            log_sed = self.SPECTRUM_FUNCTIONS["slow_cooling"](log_nu, log_nu_m, log_nu_c, log_nu_max, p, s)
        elif regime == "no_cooling":
            log_sed = self.SPECTRUM_FUNCTIONS["no_cooling"](log_nu, log_nu_m, log_nu_max, p, s)
        else:
            raise RuntimeError("Unrecognized cooling regime.")

        return log_sed + log_F_norm

    def _log_opt_sed(self, log_nu, log_nu_m, log_nu_c, log_nu_max, log_F_norm, p, s):
        r"""
        Optimized log-space SED evaluation with regime determination.

        This method orchestrates the full SED evaluation by:

        1. Determining the global spectral regime,
        2. Dispatching to the appropriate regime-specific kernel.

        All inputs are assumed to be in **natural logarithmic CGS units**.

        Parameters
        ----------
        log_nu : "_ArrayLike"
            Natural logarithm of the evaluation frequencies.
        log_nu_m : float
            Natural logarithm of the injection frequency.
        log_nu_c : float
            Natural logarithm of the cooling frequency.
        log_nu_max : float
            Natural logarithm of the maximum synchrotron frequency.
        log_F_norm : float
            Natural logarithm of the flux density normalization corresponding to
            the *optically thin* equivalent emission at the dominant electron frequency.
        p : float
            Electron power-law index.
        s : float
            Smoothing parameter for spectral breaks.


        Returns
        -------
        array-like
            Natural logarithm of the synchrotron SED evaluated at ``log_nu``.
        """
        # Determine the regime
        regime, _ = self._compute_sed_regime(
            log_nu_m,
            log_nu_c,
            log_nu_max,
        )
        # Dispatch to the appropriate regime function
        return self._log_opt_sed_from_regime(
            log_nu,
            regime,
            log_nu_m,
            log_nu_c,
            log_nu_max,
            log_F_norm,
            p,
            s,
        )

    def sed(
        self,
        nu: "_UnitBearingArrayLike",
        *,
        nu_m: "_UnitBearingScalarLike",
        nu_c: "_UnitBearingScalarLike",
        F_norm: "_UnitBearingScalarLike",
        nu_max: "_UnitBearingScalarLike" = np.inf,
        p: float = 2.5,
        s: float = -1.0,
    ):
        r"""
        Evaluate the optically thin synchrotron spectral energy distribution.

        This is the primary user-facing method for computing the synchrotron
        flux density :math:`F_\nu` produced by a power-law electron population
        subject to radiative cooling, **without synchrotron self-absorption**.

        The spectral shape is determined by the ordering of the characteristic
        break frequencies and is evaluated using smoothed broken power laws
        (SBPLs).

        Parameters
        ----------
        nu : ~astropy.units.Quantity or float-like or array-like
            Frequencies at which to evaluate the SED.
        nu_m : ~astropy.units.Quantity or float-like
            Injection (minimum electron) synchrotron frequency
            :math:`\nu_m`.
        nu_c : ~astropy.units.Quantity or float-like
            Cooling break frequency :math:`\nu_c`.
        F_norm : ~astropy.units.Quantity or float-like
            Flux density normalization corresponding to the *optically thin* equivalent emission
            at the dominant electron frequency. See :ref:`sed_normalization` for details
            on how to determine the appropriate normalization.
        nu_max : ~astropy.units.Quantity or float-like, optional
            Maximum synchrotron frequency :math:`\nu_{\max}`.
            Defaults to :math:`\infty`.
        p : float, optional
            Power-law index of the injected electron energy distribution.
            Default is ``2.5``.
        s : float, optional
            SBPL smoothing parameter controlling the sharpness of spectral
            breaks. Must be negative for physical behavior. Default is ``-1.0``.

        Returns
        -------
        flux : astropy.units.Quantity
            Flux density :math:`F_\nu` evaluated at ``nu``.

        Notes
        -----
        - Units are validated and coerced internally.
        - The cooling regime is determined **globally** from the ordering of
          :math:`\nu_m`, :math:`\nu_c`, and :math:`\nu_{\max}` and applies to
          the entire spectrum.
        - The returned SED is continuous and differentiable due to SBPL
          smoothing.
        """
        nu = ensure_in_units(nu, "Hz")
        nu_m = ensure_in_units(nu_m, "Hz")
        nu_c = ensure_in_units(nu_c, "Hz")
        nu_max = ensure_in_units(nu_max, "Hz")
        F_norm = ensure_in_units(F_norm, "erg s^-1 cm^-2 Hz^-1")

        log_sed = self._log_opt_sed(
            np.log(nu),
            log_nu_m=np.log(nu_m),
            log_nu_c=np.log(nu_c),
            log_nu_max=np.log(nu_max),
            log_F_norm=np.log(F_norm),
            p=p,
            s=s,
        )
        return np.exp(log_sed) * u.erg / (u.s * u.cm**2 * u.Hz)

    # =========================================================== #
    # Closure Relations Implementation                            #
    # =========================================================== #
    # Here we implement the closure relations to go forward and backward
    # between the physics parameters and the phenomenological SED parameters.
    def _opt_from_physics_to_params(
        self,
        log_B: float,
        log_R: float,
        log_D_L: float,
        log_gamma_min: float,
        log_gamma_c: float,
        log_gamma_max: float = np.inf,
        p: float = 2.5,
        f_V: float = 1.0,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        alpha: float = 1.0,
        gamma_bulk: float = 1.0,
        redshift: float = 0.0,
        pitch_average: bool = False,
    ):
        r"""
        Compute phenomenological cooling synchrotron SED parameters from physical inputs.

        This low-level method evaluates the cooling synchrotron closure relation in
        logarithmic CGS units and returns the corresponding phenomenological SED
        parameters in linear CGS units.

        Parameters
        ----------
        log_B : float
            Natural logarithm of the magnetic field strength in Gauss.
        log_R : float
            Natural logarithm of the characteristic source radius in cm.
        log_D_L : float
            Natural logarithm of the luminosity distance in cm.
        log_gamma_min : float
            Natural logarithm of the minimum electron Lorentz factor
            :math:`\gamma_{\min}`.
        log_gamma_c : float
            Natural logarithm of the cooling Lorentz factor
            :math:`\gamma_c`.
        log_gamma_max : float, optional
            Natural logarithm of the maximum electron Lorentz factor
            :math:`\gamma_{\max}`.
        p : float, optional
            Electron power-law index.
        f_V : float, optional
            Volume filling factor.
        epsilon_E : float, optional
            Fraction of post-shock internal energy carried by relativistic electrons.
        epsilon_B : float, optional
            Fraction of post-shock internal energy stored in magnetic fields.
        alpha : float, optional
            Electron pitch angle in radians.
        gamma_bulk : float, optional
            Bulk Lorentz factor of the emitting region.
        redshift : float, optional
            Source redshift.
        pitch_average : bool, optional
            If ``True``, use pitch-angle averaged synchrotron emissivity.

        Returns
        -------
        dict
            Dictionary containing linear CGS phenomenological parameters:

            - ``F_norm``
            - ``F_peak``
            - ``nu_m``
            - ``nu_c``
            - ``nu_max``
            - ``nu_peak``
            - ``regime``
        """
        result = _log_normalize_powerlaw_sbpl_sed_cool(
            log_B=log_B,
            log_R=log_R,
            log_D_L=log_D_L,
            log_gamma_min=log_gamma_min,
            log_gamma_c=log_gamma_c,
            log_gamma_max=log_gamma_max,
            p=p,
            f_V=f_V,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            alpha=alpha,
            gamma_bulk=gamma_bulk,
            redshift=redshift,
            pitch_average=pitch_average,
        )

        return {
            "F_norm": np.exp(result["log_F_norm"]),
            "F_peak": np.exp(result["log_F_peak"]),
            "nu_m": np.exp(result["log_nu_m"]),
            "nu_c": np.exp(result["log_nu_c"]),
            "nu_max": np.exp(result["log_nu_max"]),
            "nu_peak": np.exp(result["log_nu_peak"]),
            "regime": result["regime"],
        }

    def from_physics_to_params(
        self,
        B: "_UnitBearingScalarLike",
        R: "_UnitBearingScalarLike",
        gamma_min: float,
        gamma_c: float,
        gamma_max: float = np.inf,
        p: float = 2.5,
        f_V: float = 1.0,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        alpha: float = 1.0,
        gamma_bulk: float = 1.0,
        redshift: float = None,
        luminosity_distance: "_UnitBearingScalarLike" = None,
        angular_diameter_distance: "_UnitBearingScalarLike" = None,
        proper_distance: "_UnitBearingScalarLike" = None,
        cosmology: cosmo.Cosmology = None,
        pitch_average: bool = False,
    ):
        r"""
        Construct phenomenological cooling synchrotron SED parameters from physical inputs.

        This is the public, unit-aware interface to the cooling synchrotron closure
        relation without synchrotron self-absorption. It resolves cosmological
        distance information, converts inputs to CGS, evaluates the low-level
        closure, and returns unit-bearing phenomenological SED parameters.

        Parameters
        ----------
        B : float, array-like, or ~astropy.units.Quantity
            Magnetic field strength. Must be convertible to Gauss.
        R : float, array-like, or ~astropy.units.Quantity
            Characteristic source radius. Must be convertible to cm.
        gamma_min : float
            Minimum electron Lorentz factor :math:`\gamma_{\min}`.
        gamma_c : float
            Cooling Lorentz factor :math:`\gamma_c`.
        gamma_max : float, optional
            Maximum electron Lorentz factor :math:`\gamma_{\max}`.
        p : float, optional
            Electron power-law index.
        f_V : float, optional
            Volume filling factor.
        epsilon_E : float, optional
            Fraction of post-shock internal energy carried by relativistic electrons.
        epsilon_B : float, optional
            Fraction of post-shock internal energy stored in magnetic fields.
        alpha : float, optional
            Electron pitch angle in radians. Ignored if ``pitch_average=True``.
        gamma_bulk : float, optional
            Bulk Lorentz factor of the emitting region.
        redshift : float, optional
            Cosmological redshift of the source.
        luminosity_distance : float, array-like, or ~astropy.units.Quantity, optional
            Luminosity distance to the source.
        angular_diameter_distance : float, array-like, or ~astropy.units.Quantity, optional
            Angular diameter distance to the source.
        proper_distance : float, array-like, or ~astropy.units.Quantity, optional
            Proper distance to the source.
        cosmology : ~astropy.cosmology.Cosmology, optional
            Cosmology used to resolve missing distance/redshift information.
        pitch_average : bool, optional
            If ``True``, use pitch-angle averaged synchrotron emissivity.

        Returns
        -------
        dict
            Dictionary containing unit-bearing phenomenological parameters:

            - ``F_norm``
            - ``F_peak``
            - ``nu_m``
            - ``nu_c``
            - ``nu_max``
            - ``nu_peak``
            - ``regime``

        Notes
        -----
        Cosmological inputs are resolved into a self-consistent luminosity distance
        and redshift using :func:`resolve_cosmological_distances`.

        Although an angular-diameter distance may be supplied to the cosmology
        resolver, it is not used in this no-SSA cooling closure.
        """
        # ------------------------------------------------------------------ #
        # Resolve cosmological information.
        # ------------------------------------------------------------------ #
        cosmology = get_cosmology(cosmology=cosmology)
        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        D_L = distances["luminosity_distance"]
        redshift = distances["redshift"]

        # ------------------------------------------------------------------ #
        # Unit enforcement.
        # ------------------------------------------------------------------ #
        B = ensure_in_units(B, "G")
        R = ensure_in_units(R, "cm")
        D_L = ensure_in_units(D_L, "cm")

        # ------------------------------------------------------------------ #
        # Convert to logarithmic CGS units.
        # ------------------------------------------------------------------ #
        log_B = np.log(B)
        log_R = np.log(R)
        log_D_L = np.log(D_L)
        log_gamma_min = np.log(gamma_min)
        log_gamma_c = np.log(gamma_c)
        log_gamma_max = np.log(gamma_max)

        # ------------------------------------------------------------------ #
        # Dispatch to the low-level closure.
        # ------------------------------------------------------------------ #
        params = self._opt_from_physics_to_params(
            log_B=log_B,
            log_R=log_R,
            log_D_L=log_D_L,
            log_gamma_min=log_gamma_min,
            log_gamma_c=log_gamma_c,
            log_gamma_max=log_gamma_max,
            p=p,
            f_V=f_V,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            alpha=alpha,
            gamma_bulk=gamma_bulk,
            redshift=redshift,
            pitch_average=pitch_average,
        )

        # ------------------------------------------------------------------ #
        # Attach physical units.
        # ------------------------------------------------------------------ #
        return {
            "F_norm": params["F_norm"] * u.erg / (u.s * u.cm**2 * u.Hz),
            "F_peak": params["F_peak"] * u.erg / (u.s * u.cm**2 * u.Hz),
            "nu_m": params["nu_m"] * u.Hz,
            "nu_c": params["nu_c"] * u.Hz,
            "nu_max": params["nu_max"] * u.Hz,
            "nu_peak": params["nu_peak"] * u.Hz,
            "regime": params["regime"],
        }

    def _opt_from_params_to_physics(
        self,
        regime: str,
        log_F_peak: "_ArrayLike",
        log_nu_peak: "_ArrayLike",
        log_D_L: "_ArrayLike",
        gamma_min: float = 1.0,
        gamma_c: float = np.inf,
        gamma_max: float = np.inf,
        p: float = 3.0,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        f_V: float = 0.5,
        gamma_bulk: float = 1.0,
        redshift: float = 0.0,
        alpha: float = None,
        pitch_average: bool = False,
    ):
        r"""
        Low-level inversion of synchrotron peak observables to physical parameters.

        This method implements the **log-space inverse closure relation** for the
        synchrotron SED including **radiative cooling and synchrotron
        self-absorption (SSA)**.

        It maps observed spectral peak quantities

        - :math:`F_{\nu,\mathrm{pk}}`
        - :math:`\nu_{\mathrm{pk}}`

        to the corresponding **source radius** :math:`R` and **magnetic field**
        :math:`B` under a specific synchrotron spectral regime.

        This routine is the optimized internal implementation used by
        :meth:`from_params_to_physics`.  All inputs must already be expressed in
        **natural logarithmic CGS units**, and no unit validation is performed.

        .. important::

            Unlike the simpler optically-thin synchrotron SED, **the inversion is not
            unique without specifying the spectral regime**.  Different SSA and
            cooling regimes correspond to different relationships between
            observables and physical parameters.

            For this reason the caller must explicitly provide the **regime
            identifier**.

        .. warning::

            This method implements a **closure relation**, not a direct physical
            measurement.  The inferred values of :math:`R` and :math:`B` are those
            required to reproduce the supplied observables **under the assumptions
            of the chosen regime and closure prescription**.

            If the supplied regime does not match the true spectral configuration,
            the inferred parameters may be significantly incorrect.

        Parameters
        ----------
        regime : str
            Identifier for the synchrotron spectral regime.  Must correspond to one
            of the regimes defined in :attr:`SPECTRUM_INVERSION_FUNCTIONS`.

            Each regime encodes a specific ordering of the characteristic
            synchrotron frequencies (e.g. :math:`\nu_a`, :math:`\nu_m`,
            :math:`\nu_c`, :math:`\nu_{\max}`) and therefore determines the
            analytic inversion formula used.

        log_F_peak : array-like
            Natural logarithm of the observed peak flux density
            :math:`F_{\nu,\mathrm{pk}}` in CGS units
            (:math:`\mathrm{erg\,cm^{-2}\,s^{-1}\,Hz^{-1}}`).

        log_nu_peak : array-like
            Natural logarithm of the observed peak frequency (Hz).

        log_D_L : array-like
            Natural logarithm of the luminosity distance (cm).

        gamma_min : float, optional
            Minimum Lorentz factor of the accelerated electron distribution.

        gamma_c : float, optional
            Cooling Lorentz factor.

            Must be finite for regimes where cooling explicitly shapes the
            spectrum.

        gamma_max : float, optional
            Maximum electron Lorentz factor.

        p : float, optional
            Power-law index of the electron energy distribution.

        epsilon_E : float, optional
            Fraction of post-shock internal energy carried by relativistic
            electrons.

        epsilon_B : float, optional
            Fraction of post-shock internal energy stored in magnetic fields.

        f_V : float, optional
            Volume filling factor of the emitting region.

        gamma_bulk : float, optional
            Bulk Lorentz factor of the emitting region.

            A simple on-axis Doppler transformation is applied to convert the
            observed peak quantities into the comoving frame prior to inversion.

        redshift : float, optional
            Cosmological redshift of the source.

        alpha : float, optional
            Electron pitch angle in radians.

            Required if ``pitch_average=False``.

        pitch_average : bool, optional
            If ``True``, the pitch-angle-averaged synchrotron emissivity is used.

        Returns
        -------
        dict
            Dictionary containing inferred physical parameters

            - ``R`` : source radius (cm)
            - ``B`` : magnetic field strength (G)

            Values are returned in **linear CGS units**.

        Notes
        -----
        **Relativistic corrections**

        Observed quantities are transformed into the comoving frame using

        .. math::

            \nu' = \nu \frac{1+z}{\delta}

            F' = F \frac{1+z}{\delta^3}

        assuming on-axis emission with Doppler factor
        :math:`\delta(\Gamma, \theta=0)`.

        **Closure assumptions**

        The inversion assumes

        - a single homogeneous emission zone,
        - synchrotron emission from a power-law electron population,
        - equipartition-style microphysical closure,
        - a spherical or quasi-spherical emitting geometry,
        - on-axis relativistic boosting if bulk motion is present.

        **Why regimes matter**

        Different regimes correspond to different peak-forming mechanisms:

        - optically thin emission near :math:`\nu_m`
        - optically thick SSA peaks
        - cooling-dominated peaks

        Each of these produces **different scalings of**
        :math:`R` and :math:`B` with the observables.

        See Also
        --------
        from_params_to_physics
            Public, unit-aware wrapper for this inversion.
        determine_sed_regime
            Utility for determining the appropriate spectral regime.
        """
        # ------------------------------------------------------------------ #
        # Pitch-angle treatment
        # ------------------------------------------------------------------ #
        if pitch_average:
            if alpha is not None:
                warnings.warn(
                    "Pitch averaging is enabled, so the provided `alpha` is ignored. "
                    "(Set `alpha=None` to suppress this warning.)",
                    stacklevel=2,
                )
            sin_alpha = None
        else:
            if alpha is None:
                raise ValueError("Pitch averaging is disabled, so a fixed pitch angle must be provided via `alpha`.")
            sin_alpha = np.sin(alpha)

        # ------------------------------------------------------------------ #
        # Handle the regime-specification
        # ------------------------------------------------------------------ #
        if regime not in self.SPECTRUM_INVERSION_FUNCTIONS:
            raise ValueError(
                f"Unrecognized SED regime: {regime}."
                f" Available regimes: {list(self.SPECTRUM_INVERSION_FUNCTIONS.keys())}"
            )

        inversion_function = self.SPECTRUM_INVERSION_FUNCTIONS[regime]
        # ------------------------------------------------------------------ #
        # Transform observed peak quantities into the comoving frame.
        #
        # This applies the same minimal relativistic treatment used elsewhere
        # in the default closure: an on-axis Doppler correction plus the usual
        # cosmological redshift correction.
        #
        # IMPORTANT: this is intentionally simple and is not a substitute for a
        # full relativistic radiative-transfer treatment.
        # ------------------------------------------------------------------ #
        doppler_factor = compute_doppler_factor(gamma_bulk, 0.0)

        log_nu_peak_comoving = log_nu_peak - np.log(doppler_factor) + np.log1p(redshift)
        log_F_peak_comoving = log_F_peak - 3.0 * np.log(doppler_factor) + np.log1p(redshift)

        # ------------------------------------------------------------------ #
        # Invert the default closure in the comoving frame.
        # ------------------------------------------------------------------ #
        if regime == "fast_cooling":  # nu_c < nu_m < nu_max
            log_R, log_B = inversion_function(
                log_nu_peak_comoving,
                log_F_peak_comoving,
                log_D_L,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                gamma_c=gamma_c,
                p=p,
                epsilon_e=epsilon_E,
                epsilon_B=epsilon_B,
                f_V=f_V,
                sin_alpha=sin_alpha,
            )
        elif regime == "slow_cooling":  # nu_m < nu_c < nu_max
            log_R, log_B = inversion_function(
                log_nu_peak_comoving,
                log_F_peak_comoving,
                log_D_L,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                gamma_c=gamma_c,
                p=p,
                epsilon_e=epsilon_E,
                epsilon_B=epsilon_B,
                f_V=f_V,
                sin_alpha=sin_alpha,
            )
        elif regime == "no_cooling":  # nu_m < nu_max < nu_c
            log_R, log_B = inversion_function(
                log_nu_peak_comoving,
                log_F_peak_comoving,
                log_D_L,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                p=p,
                epsilon_e=epsilon_E,
                epsilon_B=epsilon_B,
                f_V=f_V,
                sin_alpha=sin_alpha,
            )
        else:
            raise ValueError(
                f"Unrecognized SED regime: {regime}."
                f" Available regimes: {list(self.SPECTRUM_INVERSION_FUNCTIONS.keys())}"
            )

        return {
            "R": np.exp(log_R),
            "B": np.exp(log_B),
        }

    def from_params_to_physics(
        self,
        regime: str,
        F_peak: "_UnitBearingScalarLike",
        nu_peak: "_UnitBearingScalarLike",
        gamma_min: float = 1.0,
        gamma_c: float = np.inf,
        gamma_max: float = np.inf,
        p: float = 3.0,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        f_V: float = 0.5,
        gamma_bulk: float = 1.0,
        redshift: float = None,
        luminosity_distance: "_UnitBearingScalarLike" = None,
        angular_diameter_distance: "_UnitBearingScalarLike" = None,
        proper_distance: "_UnitBearingScalarLike" = None,
        cosmology: cosmo.Cosmology = None,
        alpha: float = None,
        pitch_average: bool = False,
    ):
        r"""
        Infer physical source parameters from observed synchrotron peak quantities.

        This method provides the **public, unit-aware interface** for the inverse
        synchrotron closure relation implemented by
        :meth:`_opt_from_params_to_physics`.

        It converts observed peak quantities

        - peak flux density :math:`F_{\nu,\mathrm{pk}}`
        - peak frequency :math:`\nu_{\mathrm{pk}}`

        into estimates of the **source radius** :math:`R` and **magnetic field**
        :math:`B`, assuming the emission follows one of the spectral regimes
        described by :class:`PowerLaw_Cooling_SSA_SynchrotronSED`.

        The calculation proceeds by

        1. resolving cosmological distance information,
        2. converting all quantities into CGS units,
        3. transforming observables into logarithmic form,
        4. dispatching to the optimized inversion routine.

        .. important::

            The caller **must specify the synchrotron spectral regime**.  This
            inversion cannot determine the regime automatically because multiple
            spectral configurations may produce identical peak observables.

        .. warning::

            The inferred parameters depend strongly on the assumed regime and
            closure relation.  Incorrect regime identification will generally lead
            to incorrect physical parameters.

        Parameters
        ----------
        regime : str
            Identifier of the synchrotron spectral regime.

            The spectrum is identified as ``"SpectrumN"``, where ``N`` is an
            integer corresponding to the regimes defined in
            :ref:`synch_sed_theory` and implemented internally in
            :attr:`SPECTRUM_INVERSION_FUNCTIONS`.

            Each regime corresponds to a specific ordering of the characteristic
            synchrotron frequencies (e.g. :math:`\nu_a`, :math:`\nu_m`,
            :math:`\nu_c`, :math:`\nu_{\max}`) and therefore determines the
            analytic inversion formula used to infer the physical parameters.
            The user must ensure that the supplied regime is consistent with the
            observed spectral configuration.

        F_peak : ~numpy.ndarray or float or ~astropy.units.Quantity
            Observed (lab-frame) peak flux density.

            Must be convertible to
            :math:`\mathrm{erg\,cm^{-2}\,s^{-1}\,Hz^{-1}}`.

            If an array is provided, the inversion is performed **element-wise**
            and the returned physical parameters will have the same shape.
            This makes it possible to evaluate the closure relation across
            posterior samples, Monte Carlo realizations, or time-series data.

        nu_peak : ~numpy.ndarray or float or ~astropy.units.Quantity
            Observed (lab-frame) peak frequency.

            Must be convertible to Hz.

            As with ``F_peak``, array inputs are supported and will produce
            element-wise solutions for the inferred parameters.

        gamma_min : float, optional
            Minimum Lorentz factor of the accelerated electron distribution.

            By default this is ``1.0``, corresponding to a non-relativistic
            minimum electron energy. In many astrophysical applications
            (e.g. shock acceleration), values of
            :math:`\gamma_{\min} \sim 10-10^3` may be more physically
            appropriate.

        gamma_c: float, optional

            Cooling Lorentz factor.

            This parameter is required for regimes where radiative cooling explicitly shapes
            the spectrum (e.g. fast cooling). For regimes where cooling does not affect the
            spectral shape (e.g. no cooling), this parameter is ignored and can be set to any
            large value (e.g. ``np.inf``).

        gamma_max : float, optional
            Maximum Lorentz factor of the accelerated electron distribution.

            The default value of ``np.inf`` corresponds to no high-energy
            cutoff. This assumption is typically valid provided that the
            synchrotron cutoff frequency lies well above the observed band.

        p : float, optional
            Power-law index of the electron Lorentz factor distribution,

            .. math::

                N(\gamma) \propto \gamma^{-p}.

            The default value ``p = 3.0`` is commonly adopted for
            shock-accelerated electron populations.

        epsilon_E : float, optional
            Fraction of the post-shock internal energy carried by relativistic
            electrons.

            This parameter enters the equipartition-style closure relation used
            to normalize the electron distribution.

        epsilon_B : float, optional
            Fraction of the post-shock internal energy stored in magnetic
            fields.

            Together with ``epsilon_E``, this parameter determines the relative
            partition of energy between particles and magnetic fields in the
            emitting region.

        f_V : float, optional
            Effective **volume filling factor** of the emitting region.

            In optically thin emission scenarios, this parameter represents the
            fraction of the spherical volume that actively emits synchrotron
            radiation.

            Physically, this accounts for situations where the emission arises
            from a thin shell or clumpy medium rather than uniformly filling
            the entire volume.

            For example:

            - ``f_V = 1`` corresponds to a uniformly filled sphere.
            - ``f_V < 1`` may represent emission from a thin shocked shell or
              fragmented ejecta.

        gamma_bulk : float, optional
            Bulk Lorentz factor of the emitting region.

            If non-unity, the observed peak quantities are transformed into the
            comoving frame using relativistic Doppler corrections prior to
            performing the inversion.

            The implementation assumes **on-axis emission**, corresponding to a
            Doppler factor

            .. math::

                \delta(\Gamma, \theta=0).

        redshift : float, optional
            Cosmological redshift of the source.

            This parameter enters the conversion between observed and comoving
            frequencies and fluxes and also determines the cosmological
            distances if they are not supplied explicitly.

        luminosity_distance : quantity, optional
            Luminosity distance to the source.

            Must be convertible to cm.

            If supplied, this value is used directly in the inversion.

        angular_diameter_distance : quantity, optional
            Angular-diameter distance to the source.

            Must be convertible to cm.

            This quantity is required in some SSA regimes where the inversion
            depends explicitly on the **angular size of the emitting region**.

        proper_distance : quantity, optional
            Proper distance to the source.

            This can be used as an alternative specification of the cosmological
            distance.

        cosmology : ~astropy.cosmology.Cosmology, optional
            Cosmology used to compute missing distance quantities.

            The distances required by the inversion are resolved using the
            helper function :func:`resolve_cosmological_distances`, which
            processes the supplied cosmological parameters as follows:

            1. If explicit distances are provided (e.g. ``luminosity_distance``),
               those values are used directly.

            2. If only a redshift is provided, the necessary distances are
               computed using the supplied ``cosmology``.

            3. If no cosmology is provided, the default cosmology returned by
               :func:`get_cosmology` is used.

            This mechanism ensures that the inversion has access to all
            required cosmological distance measures while allowing users to
            specify whichever subset of parameters is most convenient.

        alpha : float, optional
            Electron pitch angle in radians.

            This parameter is required when ``pitch_average=False`` and
            specifies the angle between the electron velocity and the magnetic
            field.

        pitch_average : bool, optional
            If ``True``, the synchrotron emissivity is averaged over an
            isotropic distribution of pitch angles.

            In this case the explicit pitch angle ``alpha`` is ignored and the
            appropriate angle-averaged emissivity coefficients are used.

            Pitch-angle averaging is often appropriate when the electron
            distribution is isotropic in momentum space.

        Returns
        -------
        dict
            Dictionary containing inferred source parameters

            - ``R`` : source radius (cm)
            - ``B`` : magnetic field strength (G)

        Notes
        -----
        **Interpretation**

        These values are **model-dependent estimates**, not direct measurements.

        They represent the values of :math:`R` and :math:`B` that reproduce the
        observed peak under the assumptions of

        - the selected synchrotron spectral regime,
        - a single-zone homogeneous emitting region,
        - a power-law electron population,
        - equipartition-style microphysical closure.

        **When this method may fail**

        The inversion may produce misleading results if

        - the true spectrum contains multiple emitting zones,
        - the peak is produced by a process not described by the selected regime,
        - strong radiative transfer effects modify the spectrum,
        - relativistic equal-arrival-time surfaces dominate the emission,
        - the geometry deviates strongly from the assumed configuration.

        In such cases, users should treat the inferred parameters as
        **order-of-magnitude estimates only**.

        See Also
        --------
        determine_sed_regime
            Identify the synchrotron spectral regime.
        from_physics_to_params
            Forward closure relation.
        sed
            Evaluate the SED for a given parameter set.
        """
        # ------------------------------------------------------------------ #
        # Resolve cosmological information.
        # ------------------------------------------------------------------ #
        cosmology = get_cosmology(cosmology=cosmology)
        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        D_L = distances["luminosity_distance"]
        redshift = distances["redshift"]

        # ------------------------------------------------------------------ #
        # Unit enforcement and conversion to internal log-space representation.
        # ------------------------------------------------------------------ #
        log_F_peak = np.log(ensure_in_units(F_peak, "erg cm^-2 s^-1 Hz^-1"))
        log_nu_peak = np.log(ensure_in_units(nu_peak, "Hz"))
        log_D_L = np.log(ensure_in_units(D_L, "cm"))

        # ------------------------------------------------------------------ #
        # Dispatch to the low-level inverse closure.
        # ------------------------------------------------------------------ #
        results = self._opt_from_params_to_physics(
            regime=regime,
            log_F_peak=log_F_peak,
            log_nu_peak=log_nu_peak,
            log_D_L=log_D_L,
            gamma_min=gamma_min,
            gamma_c=gamma_c,
            gamma_max=gamma_max,
            p=p,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            gamma_bulk=gamma_bulk,
            redshift=redshift,
            alpha=alpha,
            pitch_average=pitch_average,
        )

        return {
            "R": results["R"] * u.cm,
            "B": results["B"] * u.G,
        }


class PowerLaw_SSA_SynchrotronSED(MultiSpectrumSynchrotronSED):
    r"""
    Synchrotron SED with synchrotron self-absorption and no radiative cooling.

    This class implements the **canonical non-cooling synchrotron spectral energy
    distribution** produced by a power-law electron population, including
    **synchrotron self-absorption (SSA)** but **excluding radiative cooling**.

    It is applicable to homogeneous, single-zone emission regions in which the
    synchrotron cooling frequency lies well above the maximum frequency of
    interest, while synchrotron self-absorption may occur at or near the injection
    frequency. Typical examples include early-time shocks or low-density systems
    where electron cooling is negligible but SSA shapes the low-frequency spectrum.

    The spectrum is constructed using **scale-free smoothed broken power laws
    (SFBPLs)** and evaluated entirely in log-space to ensure numerical stability and
    smooth transitions between spectral segments.

    For theoretical background and derivations, see :ref:`synchrotron_theory` and
    :ref:`synch_sed_theory`.

    .. rubric:: Spectral Structure

    The synchrotron spectrum is classified into one of two **self-absorption
    regimes**, based on the ordering of the characteristic frequencies:

    - Injection frequency :math:`\nu_m`,
    - Self-absorption frequency :math:`\nu_a`,
    - Maximum synchrotron frequency :math:`\nu_{\max}`.

    The supported configurations are:

    - **Optically thin at the injection frequency**
      (:math:`\nu_a < \nu_m < \nu_{\max}`)

    - **Optically thick at the injection frequency**
      (:math:`\nu_m < \nu_a < \nu_{\max}`)

    There is no physically relevant regime with
    :math:`\nu_a > \nu_{\max}` in the absence of radiative cooling.

    The selected regime applies **globally** to the spectrum and does not depend on
    the frequency grid used for evaluation.

    .. rubric:: SED Parameters

    The parameters entering this SED fall into three conceptual categories.

    .. tab-set::

        .. tab-item:: Free parameters (phenomenological)

            These parameters define the observable structure of the SED and are
            typically inferred directly from broadband data.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - Normalization flux density
                  - :math:`F_{\nu,\mathrm{norm}}`
                  - Flux density of the **dominant optically thin emitting population**.
                    In the non-cooling SSA case, this corresponds to the optically thin
                    flux at the injection frequency :math:`\nu_m`,
                    i.e. :math:`F_{\nu,\mathrm{norm}} = F_\nu(\nu_m)`.
                * - Injection frequency
                  - :math:`\nu_m`
                  - Synchrotron characteristic frequency of the minimum-energy
                    electrons.
                * - Maximum frequency
                  - :math:`\nu_{\max}`
                  - High-frequency synchrotron cutoff.

        .. tab-item:: Hyper-parameters

            These parameters control the *shape* of the spectrum but are not usually
            directly constrained by broadband observations.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - Electron power-law index
                  - :math:`p`
                  - Index of the injected electron energy distribution,
                    :math:`N(\gamma) \propto \gamma^{-p}`.
                * - Smoothing parameter
                  - :math:`s`
                  - Controls the sharpness of the SSA and injection-frequency
                    transitions.
                * - Emission solid angle
                  - :math:`\Omega`
                  - Effective emitting area divided by distance squared.
                * - Minimum Lorentz factor
                  - :math:`\gamma_m`
                  - Minimum electron Lorentz factor used in normalization.

        .. tab-item:: Derived parameters (internal)

            These quantities are **not user inputs**, but are computed internally
            from the supplied parameters.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Quantity
                  - Symbol
                  - Description
                * - Self-absorption frequency
                  - :math:`\nu_a`
                  - Synchrotron self-absorption break frequency.
                * - SSA regime
                  - —
                  - Discrete identifier selecting the global SSA spectral branch.

    .. rubric:: Normalization and Closure

    The SED is **always normalized using**
    :math:`F_{\nu,\mathrm{norm}}`, defined as the flux density of the dominant
    optically thin electron population.

    In the SSA-only, non-cooling case, this population emits at the injection
    frequency :math:`\nu_m`, so the normalization is anchored at
    :math:`F_\nu(\nu_m)` regardless of whether the spectrum is optically thick or
    thin at that frequency. If the spectrum is optically thick at the peak, the
    normalization is propagated self-consistently to the true spectral maximum.

    Optional closure relations are provided to map physical model parameters
    (e.g. magnetic field strength, emitting volume, electron energy fractions)
    onto the phenomenological SED parameters under assumed microphysical
    conditions such as equipartition.

    See Also
    --------
    :class:`SynchrotronSED`
        Base class for synchrotron SED implementations.
    :class:`MultiSpectrumSynchrotronSED`
        Base class for multi-regime synchrotron SEDs.
    :class:`PowerLaw_SynchrotronSED`
        Canonical optically thin synchrotron SED without cooling or SSA.
    :class:`PowerLaw_Cooling_SynchrotronSED`
        Synchrotron SED with radiative cooling and no SSA.
    :class:`PowerLaw_Cooling_SSA_SynchrotronSED`
        Synchrotron SED including both cooling and self-absorption.

    Examples
    --------
    Evaluate a synchrotron SED with self-absorption and no cooling:

    .. code-block:: python

        import numpy as np
        import astropy.units as u
        from triceratops.radiation.synchrotron import (
            PowerLaw_SSA_SynchrotronSED,
        )

        sed = PowerLaw_SSA_SynchrotronSED()

        nu = np.logspace(8, 14, 500) * u.Hz

        F_nu = sed.sed(
            nu=nu,
            nu_m=1e11 * u.Hz,
            F_norm=1e-26 * u.erg / (u.cm**2 * u.s * u.Hz),
            p=2.5,
            s=-0.05,
        )
    """

    SPECTRUM_FUNCTIONS = SSA_SED_FUNCTION_REGISTRY
    SPECTRUM_INVERSION_FUNCTIONS = SSA_INV_FUNCTION_REGISTRY

    # ============================================================ #
    # Regime determination                                        #
    # ============================================================ #
    def _compute_sed_regime(
        self,
        log_F_norm: float,
        log_nu_m: float,
        log_omega: float,
        log_gamma_m: float,
        p: float = 3.0,
    ):
        r"""
        Determine the SSA spectral regime and self-absorption frequency.

        This method identifies the **global synchrotron self-absorption regime**
        for the non-cooling power-law synchrotron spectrum and computes the
        corresponding self-absorption frequency :math:`\nu_a`.

        Two candidate SSA frequencies are computed assuming the two possible
        spectral orderings:

        - **Optically thin injection frequency**
          (:math:`\nu_a < \nu_m < \nu_{\max}`)
        - **Optically thick injection frequency**
          (:math:`\nu_m < \nu_a < \nu_{\max}`)

        The physically consistent regime is selected by enforcing the required
        ordering between the candidate :math:`\nu_a` and the injection frequency
        :math:`\nu_m`.

        All quantities are assumed to be in **natural logarithmic space**.

        Parameters
        ----------
        log_F_norm : float
            Natural logarithm of the optically thin normalization flux density.

        log_nu_m : float
            Natural logarithm of the injection frequency
            :math:`\log \nu_m`.

        log_omega : float
            Natural logarithm of the emission solid angle
            :math:`\log \Omega`.

        log_gamma_m : float
            Natural logarithm of the minimum electron Lorentz factor
            :math:`\log \gamma_m`.

        p : float, optional
            Electron power-law index.

        Returns
        -------
        tuple[str, float]
            ``(regime, log_nu_a)`` where

            - ``regime`` is the SSA spectral regime identifier
            - ``log_nu_a`` is the natural logarithm of the
              self-absorption frequency.

        Notes
        -----
        - This method performs **no unit validation**.
        - The selected regime applies **globally** to the SED.
        """
        # Compute candidate SSA frequencies for each regime
        log_nu_ssa = compute_ssa_frequencies_without_cooling(
            log_F_norm,
            log_omega,
            log_nu_m,
            p,
            log_gamma_m,
        )

        # Select the physically consistent SSA regime
        return select_ssa_sed_regime_from_candidates_without_cooling(
            log_nu_ssa,
            log_nu_m,
        )

    def determine_sed_regime(
        self,
        nu_m: "_UnitBearingScalarLike",
        F_norm: "_UnitBearingScalarLike",
        omega: float = 4 * np.pi,
        gamma_m: float = 1.0,
        p: float = 3.0,
    ):
        r"""
        Determine the global SSA synchrotron spectral regime from phenomenological inputs.

        This user-facing method determines whether a non-cooling synchrotron spectrum is
        **optically thin or optically thick at the injection frequency**. The
        synchrotron self-absorption frequency :math:`\nu_a` is computed internally from
        the supplied flux normalization and physical parameters.

        The classification is based on the ordering of:

        - the injection frequency :math:`\nu_m`, and
        - the inferred self-absorption frequency :math:`\nu_a`.

        Parameters
        ----------
        nu_m : ~astropy.units.Quantity or float
            Injection (minimum-electron) synchrotron frequency :math:`\nu_m`.
        F_norm : ~astropy.units.Quantity or float
            Flux density normalization anchored at :math:`\nu_m`,
            i.e. :math:`F_\nu(\nu_m)`.
        omega : float, optional
            Effective emission solid angle :math:`\Omega = A / D^2`.
            Default is :math:`4\pi`.
        gamma_m : float, optional
            Minimum electron Lorentz factor :math:`\gamma_m`.
            Default is ``1.0``.
        p : float, optional
            Power-law index of the injected electron energy distribution.

        Returns
        -------
        regime : enum-like
            Member of :attr:`SPECTRUM_FUNCTIONS` identifying the global synchrotron
            self-absorption regime.

        Notes
        -----
        - This method assumes a **non-cooling**, single-zone synchrotron source.
        - The SSA frequency is inferred analytically from the supplied normalization.
        - The returned regime applies **globally** to the SED.
        - Units are validated and coerced internally.
        """
        nu_m = ensure_in_units(nu_m, "Hz")
        F_norm = ensure_in_units(F_norm, "erg s^-1 cm^-2 Hz^-1")

        log_nu_m = np.log(nu_m)
        log_F_norm = np.log(F_norm)
        log_omega = np.log(omega)
        log_gamma_m = np.log(gamma_m)

        regime, _ = self._compute_sed_regime(
            log_F_norm=log_F_norm,
            log_nu_m=log_nu_m,
            log_omega=log_omega,
            log_gamma_m=log_gamma_m,
            p=p,
        )
        return regime

    # ============================================================ #
    # SSA frequency computation                                   #
    # ============================================================ #

    # ============================================================ #
    # Regime-specific kernel                                      #
    # ============================================================ #
    def _log_opt_sed_from_regime(
        self,
        log_nu,
        regime,
        log_nu_m,
        log_nu_a,
        log_nu_max,
        log_F_norm,
        p,
        s,
    ):
        r"""
        Evaluate the log-space synchrotron SED for a fixed SSA spectral regime.

        This method dispatches to the appropriate **regime-specific numerical
        kernel** for synchrotron emission with self-absorption and applies the
        overall flux normalization.

        All spectral-shape logic is contained in the regime-specific kernel
        functions; this method performs no regime determination.

        Parameters
        ----------
        log_nu : array-like
            Natural logarithm of the frequencies at which to evaluate the SED.
        regime : enum
            SSA spectral regime identifier returned by
            :meth:`_compute_sed_regime`.
        log_nu_m : float
            Natural logarithm of the injection frequency
            :math:`\nu_m`.
        log_nu_a : float
            Natural logarithm of the self-absorption frequency
            :math:`\nu_a`.
        log_nu_max : float
            Natural logarithm of the maximum synchrotron frequency
            :math:`\nu_{\max}`.
        log_F_norm : float
            Natural logarithm of the peak flux density
            :math:`F_{\nu,\mathrm{pk}}`.
        p : float
            Electron energy power-law index.
        s : float
            Smoothness parameter for the smoothed broken power-law (SBPL)
            transitions.

        Returns
        -------
        log_sed : array-like
            Natural logarithm of the flux density
            :math:`\log F_\nu` evaluated at ``log_nu``.

        Raises
        ------
        RuntimeError
            If an unrecognized SSA regime identifier is supplied.

        Notes
        -----
        - This method assumes a **non-cooling, SSA-only** synchrotron spectrum.
        - The optically thick branch includes an explicit normalization shift to ensure
          continuity with the anchored normalization at :math:`\nu_m`.
        - No unit validation is performed.
        """
        if regime == "optically_thin":
            log_sed = (
                self.SPECTRUM_FUNCTIONS["optically_thin"](log_nu, log_nu_m, log_nu_a, log_nu_max, p, s) + log_F_norm
            )
        elif regime == "optically_thick":
            log_sed = (
                self.SPECTRUM_FUNCTIONS["optically_thick"](log_nu, log_nu_m, log_nu_a, log_nu_max, p, s)
                + log_F_norm
                - ((p - 1) / 2) * (log_nu_a - log_nu_m)
            )
        else:
            raise RuntimeError("Unrecognized SSA regime.")

        return log_sed

    def _log_opt_sed(
        self,
        log_nu: "_ArrayLike",
        log_nu_m: float,
        log_nu_max: float,
        log_F_norm: float,
        log_omega: float,
        log_gamma_m: float,
        p: float = 3.0,
        s: float = -1.0,
    ):
        r"""
        Evaluate the log-space synchrotron SED with self-absorption and no cooling.

        This optimized internal method orchestrates the full SED evaluation by:

        1. Computing the synchrotron self-absorption frequency :math:`\nu_a`,
        2. Determining the global SSA spectral regime,
        3. Dispatching to the appropriate regime-specific SED kernel.

        All calculations are performed in **natural logarithmic CGS units**.

        Parameters
        ----------
        log_nu : array-like
            Natural logarithm of the frequencies at which to evaluate the SED.
        log_nu_m : float
            Natural logarithm of the injection frequency
            :math:`\nu_m`.
        log_nu_max : float
            Natural logarithm of the maximum synchrotron frequency
            :math:`\nu_{\max}`.
        log_F_norm : float
            Natural logarithm of the peak flux density
            :math:`F_{\nu,\mathrm{pk}}`.
        log_omega : float
            Natural logarithm of the effective emission solid angle
            :math:`\Omega = A / D^2`.
        log_gamma_m : float
            Natural logarithm of the minimum electron Lorentz factor
            :math:`\gamma_m`.
        p : float, optional
            Electron energy power-law index.
            Default is ``3.0``.
        s : float, optional
            SBPL smoothness parameter.
            Must be negative for physical behavior.

        Returns
        -------
        log_sed : array-like
            Natural logarithm of the synchrotron flux density
            :math:`\log F_\nu` evaluated at ``log_nu``.

        Notes
        -----
        - This method assumes a **non-cooling**, single-zone synchrotron source.
        - Synchrotron self-absorption is included using analytic scalings.
        - No unit validation is performed.
        """
        # Compute the self-absorption frequency
        regime, log_nu_a = self._compute_sed_regime(
            log_F_norm=log_F_norm,
            log_nu_m=log_nu_m,
            log_omega=log_omega,
            log_gamma_m=log_gamma_m,
            p=p,
        )

        # Dispatch to the regime-specific SED kernel
        return self._log_opt_sed_from_regime(
            log_nu,
            regime,
            log_nu_m,
            log_nu_a,
            log_nu_max,
            log_F_norm,
            p,
            s,
        )

    def sed(
        self,
        nu: "_UnitBearingArrayLike",
        *,
        nu_m: "_UnitBearingScalarLike",
        F_norm: "_UnitBearingScalarLike",
        nu_max: "_UnitBearingScalarLike" = np.inf,
        omega: float = 4 * np.pi,
        gamma_m: float = 1.0,
        p: float = 2.5,
        s: float = -1.0,
    ):
        r"""
        Evaluate the synchrotron spectral energy distribution with self-absorption and no radiative cooling.

        This method computes the flux density :math:`F_\nu` for a power-law
        electron population emitting synchrotron radiation in a homogeneous,
        single-zone region, including synchrotron self-absorption but **excluding
        radiative cooling**.

        Parameters
        ----------
        nu : ~astropy.units.Quantity or float-like or array-like
            Frequencies at which to evaluate the SED.
        nu_m : ~astropy.units.Quantity or float-like
            Injection (minimum-electron) synchrotron frequency
            :math:`\nu_m`.
        F_norm : ~astropy.units.Quantity or float-like
            The flux density normalization anchored at :math:`\nu_m`, i.e. :math:`F_\nu(\nu_m)`.
            See :ref:`sed_normalization` for details on how to compute this from physical parameters.
        nu_max : ~astropy.units.Quantity or float-like, optional
            Maximum synchrotron frequency
            :math:`\nu_{\max}`.
            Defaults to :math:`\infty`.
        omega : float, optional
            Effective emission solid angle
            :math:`\Omega = A / D^2`.
            Default is :math:`4\pi`.
        gamma_m : float, optional
            Minimum electron Lorentz factor
            :math:`\gamma_m`.
            Default is ``1.0``.
        p : float, optional
            Electron energy power-law index.
            Default is ``2.5``.
        s : float, optional
            SBPL smoothness parameter.
            Must be negative for physical behavior.

        Returns
        -------
        flux : astropy.units.Quantity
            Flux density :math:`F_\nu` evaluated at ``nu``.

        Notes
        -----
        - Units are validated and coerced internally.
        - The SSA regime is determined automatically and applies globally.
        - This model assumes a **non-cooling** synchrotron source.
        - The returned spectrum is continuous and differentiable due to SBPL
          smoothing.
        """
        nu = ensure_in_units(nu, "Hz")
        nu_m = ensure_in_units(nu_m, "Hz")
        nu_max = ensure_in_units(nu_max, "Hz")
        log_F_norm = ensure_in_units(F_norm, "erg s^-1 cm^-2 Hz^-1")

        log_nu = np.log(nu)
        log_nu_m = np.log(nu_m)
        log_nu_max = np.log(nu_max)
        log_F_norm = np.log(log_F_norm)
        log_omega = np.log(omega)
        log_gamma_m = np.log(gamma_m)

        log_sed = self._log_opt_sed(
            log_nu,
            log_nu_m=log_nu_m,
            log_nu_max=log_nu_max,
            log_F_norm=log_F_norm,
            log_omega=log_omega,
            log_gamma_m=log_gamma_m,
            p=p,
            s=s,
        )

        return np.exp(log_sed) * u.erg / (u.s * u.cm**2 * u.Hz)

    # ============================================================ #
    # Normalization closure relations                              #
    # ============================================================ #
    def _opt_from_physics_to_params(
        self,
        log_B: float,
        log_R: float,
        log_D_L: float,
        log_D_A: float,
        log_gamma_min: float,
        log_gamma_max: float = np.inf,
        p: float = 2.5,
        f_V: float = 1.0,
        f_A: float = 1.0,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        alpha: float = 1.0,
        gamma_bulk: float = 1.0,
        redshift: float = 0.0,
        pitch_average: bool = False,
    ):
        r"""
        Compute phenomenological SSA synchrotron parameters from physical inputs.

        This low-level method converts physical parameters describing a homogeneous,
        single-zone synchrotron source into a self-consistent set of
        phenomenological SED parameters for a power-law electron population with
        synchrotron self-absorption and no radiative cooling.

        All calculations are performed internally in natural logarithmic CGS units,
        but the returned values are converted back to linear CGS values (without
        attaching units).

        Parameters
        ----------
        log_B : float
            Natural logarithm of the magnetic field strength.
        log_R : float
            Natural logarithm of the source radius.
        log_D_L : float
            Natural logarithm of the luminosity distance.
        log_D_A : float
            Natural logarithm of the angular-diameter distance.
        log_gamma_min : float
            Natural logarithm of the minimum electron Lorentz factor.
        log_gamma_max : float, optional
            Natural logarithm of the maximum electron Lorentz factor.
        p : float, optional
            Power-law index of the electron energy distribution.
        f_V : float, optional
            Volume filling factor.
        f_A : float, optional
            Area filling factor.
        epsilon_E : float, optional
            Fraction of post-shock internal energy carried by relativistic electrons.
        epsilon_B : float, optional
            Fraction of post-shock internal energy stored in magnetic fields.
        alpha : float, optional
            Electron pitch angle in radians.
        gamma_bulk : float, optional
            Bulk Lorentz factor of the emitting region.
        redshift : float, optional
            Source redshift.
        pitch_average : bool, optional
            If ``True``, use pitch-angle averaged synchrotron emissivity.

        Returns
        -------
        dict
            Dictionary containing linear CGS phenomenological parameters:

            - ``F_norm``
            - ``F_peak``
            - ``nu_m``
            - ``nu_a``
            - ``nu_max``
            - ``nu_peak``
            - ``regime``
        """
        log_results = _log_normalize_powerlaw_sbpl_sed_ssa(
            log_B=log_B,
            log_R=log_R,
            log_D_L=log_D_L,
            log_D_A=log_D_A,
            log_gamma_min=log_gamma_min,
            log_gamma_max=log_gamma_max,
            p=p,
            f_V=f_V,
            f_A=f_A,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            alpha=alpha,
            gamma_bulk=gamma_bulk,
            redshift=redshift,
            pitch_average=pitch_average,
        )

        return {
            "F_norm": np.exp(log_results["log_F_norm"]),
            "F_peak": np.exp(log_results["log_F_peak"]),
            "nu_m": np.exp(log_results["log_nu_m"]),
            "nu_a": np.exp(log_results["log_nu_a"]),
            "nu_max": np.exp(log_results["log_nu_max"]),
            "nu_peak": np.exp(log_results["log_nu_peak"]),
            "regime": log_results["regime"],
        }

    def from_physics_to_params(
        self,
        B: "_UnitBearingScalarLike",
        R: "_UnitBearingScalarLike",
        gamma_min: float,
        gamma_max: float = np.inf,
        p: float = 2.5,
        f_V: float = 1.0,
        f_A: float = 1.0,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        alpha: float = 1.0,
        gamma_bulk: float = 1.0,
        redshift: float = None,
        luminosity_distance: "_UnitBearingScalarLike" = None,
        angular_diameter_distance: "_UnitBearingScalarLike" = None,
        proper_distance: "_UnitBearingScalarLike" = None,
        cosmology: cosmo.Cosmology = None,
        pitch_average: bool = False,
    ):
        r"""
        Construct phenomenological SSA synchrotron SED parameters from physical inputs.

        This is the public, unit-aware interface to the non-cooling SSA closure
        relation. Cosmological distance information is resolved, inputs are
        converted to CGS units, and the resulting phenomenological parameters are
        returned with units attached.

        Parameters
        ----------
        B : float, array-like, or ~astropy.units.Quantity
            Magnetic field strength. Must be convertible to Gauss.
        R : float, array-like, or ~astropy.units.Quantity
            Characteristic source radius. Must be convertible to cm.
        gamma_min : float
            Minimum electron Lorentz factor :math:`\gamma_{\min}`.
        gamma_max : float, optional
            Maximum electron Lorentz factor :math:`\gamma_{\max}`.
        p : float, optional
            Electron power-law index.
        f_V : float, optional
            Volume filling factor.
        f_A : float, optional
            Area filling factor entering the effective solid angle.
        epsilon_E : float, optional
            Fraction of post-shock internal energy carried by relativistic electrons.
        epsilon_B : float, optional
            Fraction of post-shock internal energy stored in magnetic fields.
        alpha : float, optional
            Electron pitch angle in radians. Ignored if ``pitch_average=True``.
        gamma_bulk : float, optional
            Bulk Lorentz factor of the emitting region.
        redshift : float, optional
            Cosmological redshift of the source.
        luminosity_distance : float, array-like, or ~astropy.units.Quantity, optional
            Luminosity distance to the source.
        angular_diameter_distance : float, array-like, or ~astropy.units.Quantity, optional
            Angular-diameter distance to the source.
        proper_distance : float, array-like, or ~astropy.units.Quantity, optional
            Proper distance to the source.
        cosmology : ~astropy.cosmology.Cosmology, optional
            Cosmology used to resolve missing distance information.
        pitch_average : bool, optional
            If ``True``, use pitch-angle averaged synchrotron emissivity.

        Returns
        -------
        dict
            Dictionary containing unit-bearing phenomenological parameters:

            - ``F_norm``
            - ``F_peak``
            - ``nu_m``
            - ``nu_a``
            - ``nu_max``
            - ``nu_peak``
            - ``regime``

        Notes
        -----
        Cosmological inputs are resolved using
        :func:`resolve_cosmological_distances`. If an angular-diameter distance is
        not explicitly supplied, it is computed from the resolved luminosity
        distance and redshift via

        .. math::

            D_A = \frac{D_L}{(1+z)^2}.

        This method assumes a homogeneous single-zone source and does not include
        radiative cooling.
        """
        # ------------------------------------------------------------------ #
        # Resolve cosmological information.
        # ------------------------------------------------------------------ #
        cosmology = get_cosmology(cosmology=cosmology)
        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        D_L = distances["luminosity_distance"]
        D_A = distances["angular_diameter_distance"]
        redshift = distances["redshift"]

        # ------------------------------------------------------------------ #
        # Unit enforcement.
        # ------------------------------------------------------------------ #
        B = ensure_in_units(B, "G")
        R = ensure_in_units(R, "cm")
        D_L = ensure_in_units(D_L, "cm")
        D_A = ensure_in_units(D_A, "cm")

        # ------------------------------------------------------------------ #
        # Convert to logarithmic CGS values.
        # ------------------------------------------------------------------ #
        log_B = np.log(B)
        log_R = np.log(R)
        log_D_L = np.log(D_L)
        log_D_A = np.log(D_A)
        log_gamma_min = np.log(gamma_min)
        log_gamma_max = np.log(gamma_max)

        # ------------------------------------------------------------------ #
        # Dispatch to the low-level closure.
        # ------------------------------------------------------------------ #
        params = self._opt_from_physics_to_params(
            log_B=log_B,
            log_R=log_R,
            log_D_L=log_D_L,
            log_D_A=log_D_A,
            log_gamma_min=log_gamma_min,
            log_gamma_max=log_gamma_max,
            p=p,
            f_V=f_V,
            f_A=f_A,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            alpha=alpha,
            gamma_bulk=gamma_bulk,
            redshift=redshift,
            pitch_average=pitch_average,
        )

        # ------------------------------------------------------------------ #
        # Attach units and return.
        # ------------------------------------------------------------------ #
        return {
            "F_norm": params["F_norm"] * u.erg / (u.cm**2 * u.s * u.Hz),
            "F_peak": params["F_peak"] * u.erg / (u.cm**2 * u.s * u.Hz),
            "nu_m": params["nu_m"] * u.Hz,
            "nu_a": params["nu_a"] * u.Hz,
            "nu_max": params["nu_max"] * u.Hz,
            "nu_peak": params["nu_peak"] * u.Hz,
            "regime": params["regime"],
        }

    def _opt_from_params_to_physics(
        self,
        regime: str,
        log_F_peak: "_ArrayLike",
        log_nu_peak: "_ArrayLike",
        log_D_L: "_ArrayLike",
        log_D_A: "_ArrayLike" = None,
        gamma_min: float = 1.0,
        gamma_max: float = np.inf,
        p: float = 3.0,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        f_V: float = 0.5,
        f_A: float = 0.5,
        gamma_bulk: float = 1.0,
        redshift: float = 0.0,
        alpha: float = None,
        pitch_average: bool = False,
    ):
        r"""
        Low-level inversion of synchrotron peak observables to physical parameters.

        This method implements the **log-space inverse closure relation** for the
        synchrotron SED including **radiative cooling and synchrotron
        self-absorption (SSA)**.

        It maps observed spectral peak quantities

        - :math:`F_{\nu,\mathrm{pk}}`
        - :math:`\nu_{\mathrm{pk}}`

        to the corresponding **source radius** :math:`R` and **magnetic field**
        :math:`B` under a specific synchrotron spectral regime.

        This routine is the optimized internal implementation used by
        :meth:`from_params_to_physics`.  All inputs must already be expressed in
        **natural logarithmic CGS units**, and no unit validation is performed.

        .. important::

            Unlike the simpler optically-thin synchrotron SED, **the inversion is not
            unique without specifying the spectral regime**.  Different SSA and
            cooling regimes correspond to different relationships between
            observables and physical parameters.

            For this reason the caller must explicitly provide the **regime
            identifier**.

        .. warning::

            This method implements a **closure relation**, not a direct physical
            measurement.  The inferred values of :math:`R` and :math:`B` are those
            required to reproduce the supplied observables **under the assumptions
            of the chosen regime and closure prescription**.

            If the supplied regime does not match the true spectral configuration,
            the inferred parameters may be significantly incorrect.

        Parameters
        ----------
        regime : str
            Identifier for the synchrotron spectral regime.  Must correspond to one
            of the regimes defined in :attr:`SPECTRUM_INVERSION_FUNCTIONS`.

            Each regime encodes a specific ordering of the characteristic
            synchrotron frequencies (e.g. :math:`\nu_a`, :math:`\nu_m`,
            :math:`\nu_c`, :math:`\nu_{\max}`) and therefore determines the
            analytic inversion formula used.

        log_F_peak : array-like
            Natural logarithm of the observed peak flux density
            :math:`F_{\nu,\mathrm{pk}}` in CGS units
            (:math:`\mathrm{erg\,cm^{-2}\,s^{-1}\,Hz^{-1}}`).

        log_nu_peak : array-like
            Natural logarithm of the observed peak frequency (Hz).

        log_D_L : array-like
            Natural logarithm of the luminosity distance (cm).

        log_D_A : array-like, optional
            Natural logarithm of the angular-diameter distance (cm).

            This quantity is required for regimes where the spectral peak occurs
            in the **optically thick SSA portion of the spectrum**, because those
            inversions depend explicitly on the **angular size of the emitting
            region**.

        gamma_min : float, optional
            Minimum Lorentz factor of the accelerated electron distribution.

        gamma_max : float, optional
            Maximum electron Lorentz factor.

        p : float, optional
            Power-law index of the electron energy distribution.

        epsilon_E : float, optional
            Fraction of post-shock internal energy carried by relativistic
            electrons.

        epsilon_B : float, optional
            Fraction of post-shock internal energy stored in magnetic fields.

        f_V : float, optional
            Volume filling factor of the emitting region.

        f_A : float, optional
            Area filling factor used in SSA inversions where the emitting surface
            area must be specified.

        gamma_bulk : float, optional
            Bulk Lorentz factor of the emitting region.

            A simple on-axis Doppler transformation is applied to convert the
            observed peak quantities into the comoving frame prior to inversion.

        redshift : float, optional
            Cosmological redshift of the source.

        alpha : float, optional
            Electron pitch angle in radians.

            Required if ``pitch_average=False``.

        pitch_average : bool, optional
            If ``True``, the pitch-angle-averaged synchrotron emissivity is used.

        Returns
        -------
        dict
            Dictionary containing inferred physical parameters

            - ``R`` : source radius (cm)
            - ``B`` : magnetic field strength (G)

            Values are returned in **linear CGS units**.

        Notes
        -----
        **Relativistic corrections**

        Observed quantities are transformed into the comoving frame using

        .. math::

            \nu' = \nu \frac{1+z}{\delta}

            F' = F \frac{1+z}{\delta^3}

        assuming on-axis emission with Doppler factor
        :math:`\delta(\Gamma, \theta=0)`.

        **Closure assumptions**

        The inversion assumes

        - a single homogeneous emission zone,
        - synchrotron emission from a power-law electron population,
        - equipartition-style microphysical closure,
        - a spherical or quasi-spherical emitting geometry,
        - on-axis relativistic boosting if bulk motion is present.

        **Why regimes matter**

        Different regimes correspond to different peak-forming mechanisms:

        - optically thin emission near :math:`\nu_m`
        - optically thick SSA peaks
        - cooling-dominated peaks

        Each of these produces **different scalings of**
        :math:`R` and :math:`B` with the observables.

        See Also
        --------
        from_params_to_physics
            Public, unit-aware wrapper for this inversion.
        determine_sed_regime
            Utility for determining the appropriate spectral regime.
        """
        # ------------------------------------------------------------------ #
        # Pitch-angle treatment
        # ------------------------------------------------------------------ #
        if pitch_average:
            if alpha is not None:
                warnings.warn(
                    "Pitch averaging is enabled, so the provided `alpha` is ignored. "
                    "(Set `alpha=None` to suppress this warning.)",
                    stacklevel=2,
                )
            sin_alpha = None
        else:
            if alpha is None:
                raise ValueError("Pitch averaging is disabled, so a fixed pitch angle must be provided via `alpha`.")
            sin_alpha = np.sin(alpha)

        # ------------------------------------------------------------------ #
        # Handle the regime-specification
        # ------------------------------------------------------------------ #
        if regime not in self.SPECTRUM_INVERSION_FUNCTIONS:
            raise ValueError(
                f"Unrecognized SED regime: {regime}."
                f" Available regimes: {list(self.SPECTRUM_INVERSION_FUNCTIONS.keys())}"
            )

        inversion_function = self.SPECTRUM_INVERSION_FUNCTIONS[regime]
        # ------------------------------------------------------------------ #
        # Transform observed peak quantities into the comoving frame.
        #
        # This applies the same minimal relativistic treatment used elsewhere
        # in the default closure: an on-axis Doppler correction plus the usual
        # cosmological redshift correction.
        #
        # IMPORTANT: this is intentionally simple and is not a substitute for a
        # full relativistic radiative-transfer treatment.
        # ------------------------------------------------------------------ #
        doppler_factor = compute_doppler_factor(gamma_bulk, 0.0)

        log_nu_peak_comoving = log_nu_peak - np.log(doppler_factor) + np.log1p(redshift)
        log_F_peak_comoving = log_F_peak - 3.0 * np.log(doppler_factor) + np.log1p(redshift)

        # ------------------------------------------------------------------ #
        # Invert the default closure in the comoving frame.
        # ------------------------------------------------------------------ #
        if regime == "optically_thick":  # nu_a < nu_m < nu_max < nu_c.
            if log_D_A is None:
                raise ValueError("Angular diameter distance `D_A` must be provided for Spectrum2 regime.")

            log_R, log_B = inversion_function(
                log_nu_peak_comoving,
                log_F_peak_comoving,
                log_D_L,
                log_D_A,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                p=p,
                epsilon_e=epsilon_E,
                epsilon_B=epsilon_B,
                f_V=f_V,
                f_A=f_A,
                sin_alpha=sin_alpha,
            )
        elif regime == "optically_thin":  # nu_m < nu_a < nu_max < nu_c.
            log_R, log_B = inversion_function(
                log_nu_peak_comoving,
                log_F_peak_comoving,
                log_D_L,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                p=p,
                epsilon_e=epsilon_E,
                epsilon_B=epsilon_B,
                f_V=f_V,
                sin_alpha=sin_alpha,
            )
        else:
            raise ValueError(
                f"Unrecognized SED regime: {regime}."
                f" Available regimes: {list(self.SPECTRUM_INVERSION_FUNCTIONS.keys())}"
            )

        return {
            "R": np.exp(log_R),
            "B": np.exp(log_B),
        }

    def from_params_to_physics(
        self,
        regime: str,
        F_peak: "_UnitBearingScalarLike",
        nu_peak: "_UnitBearingScalarLike",
        gamma_min: float = 1.0,
        gamma_max: float = np.inf,
        p: float = 3.0,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        f_V: float = 0.5,
        f_A: float = 0.5,
        gamma_bulk: float = 1.0,
        redshift: float = None,
        luminosity_distance: "_UnitBearingScalarLike" = None,
        angular_diameter_distance: "_UnitBearingScalarLike" = None,
        proper_distance: "_UnitBearingScalarLike" = None,
        cosmology: cosmo.Cosmology = None,
        alpha: float = None,
        pitch_average: bool = False,
    ):
        r"""
        Infer physical source parameters from observed synchrotron peak quantities.

        This method provides the **public, unit-aware interface** for the inverse
        synchrotron closure relation implemented by
        :meth:`_opt_from_params_to_physics`.

        It converts observed peak quantities

        - peak flux density :math:`F_{\nu,\mathrm{pk}}`
        - peak frequency :math:`\nu_{\mathrm{pk}}`

        into estimates of the **source radius** :math:`R` and **magnetic field**
        :math:`B`, assuming the emission follows one of the spectral regimes
        described by :class:`PowerLaw_Cooling_SSA_SynchrotronSED`.

        The calculation proceeds by

        1. resolving cosmological distance information,
        2. converting all quantities into CGS units,
        3. transforming observables into logarithmic form,
        4. dispatching to the optimized inversion routine.

        .. important::

            The caller **must specify the synchrotron spectral regime**.  This
            inversion cannot determine the regime automatically because multiple
            spectral configurations may produce identical peak observables.

        .. warning::

            The inferred parameters depend strongly on the assumed regime and
            closure relation.  Incorrect regime identification will generally lead
            to incorrect physical parameters.

        Parameters
        ----------
        regime : str
            Identifier of the synchrotron spectral regime.

            The spectrum is identified as ``"SpectrumN"``, where ``N`` is an
            integer corresponding to the regimes defined in
            :ref:`synch_sed_theory` and implemented internally in
            :attr:`SPECTRUM_INVERSION_FUNCTIONS`.

            Each regime corresponds to a specific ordering of the characteristic
            synchrotron frequencies (e.g. :math:`\nu_a`, :math:`\nu_m`,
            :math:`\nu_c`, :math:`\nu_{\max}`) and therefore determines the
            analytic inversion formula used to infer the physical parameters.
            The user must ensure that the supplied regime is consistent with the
            observed spectral configuration.

        F_peak : ~numpy.ndarray or float or ~astropy.units.Quantity
            Observed (lab-frame) peak flux density.

            Must be convertible to
            :math:`\mathrm{erg\,cm^{-2}\,s^{-1}\,Hz^{-1}}`.

            If an array is provided, the inversion is performed **element-wise**
            and the returned physical parameters will have the same shape.
            This makes it possible to evaluate the closure relation across
            posterior samples, Monte Carlo realizations, or time-series data.

        nu_peak : ~numpy.ndarray or float or ~astropy.units.Quantity
            Observed (lab-frame) peak frequency.

            Must be convertible to Hz.

            As with ``F_peak``, array inputs are supported and will produce
            element-wise solutions for the inferred parameters.

        gamma_min : float, optional
            Minimum Lorentz factor of the accelerated electron distribution.

            By default this is ``1.0``, corresponding to a non-relativistic
            minimum electron energy. In many astrophysical applications
            (e.g. shock acceleration), values of
            :math:`\gamma_{\min} \sim 10-10^3` may be more physically
            appropriate.

        gamma_max : float, optional
            Maximum Lorentz factor of the accelerated electron distribution.

            The default value of ``np.inf`` corresponds to no high-energy
            cutoff. This assumption is typically valid provided that the
            synchrotron cutoff frequency lies well above the observed band.

        p : float, optional
            Power-law index of the electron Lorentz factor distribution,

            .. math::

                N(\gamma) \propto \gamma^{-p}.

            The default value ``p = 3.0`` is commonly adopted for
            shock-accelerated electron populations.

        epsilon_E : float, optional
            Fraction of the post-shock internal energy carried by relativistic
            electrons.

            This parameter enters the equipartition-style closure relation used
            to normalize the electron distribution.

        epsilon_B : float, optional
            Fraction of the post-shock internal energy stored in magnetic
            fields.

            Together with ``epsilon_E``, this parameter determines the relative
            partition of energy between particles and magnetic fields in the
            emitting region.

        f_V : float, optional
            Effective **volume filling factor** of the emitting region.

            In optically thin emission scenarios, this parameter represents the
            fraction of the spherical volume that actively emits synchrotron
            radiation.

            Physically, this accounts for situations where the emission arises
            from a thin shell or clumpy medium rather than uniformly filling
            the entire volume.

            For example:

            - ``f_V = 1`` corresponds to a uniformly filled sphere.
            - ``f_V < 1`` may represent emission from a thin shocked shell or
              fragmented ejecta.

        f_A : float, optional
            Effective **area filling factor** of the emitting region.

            This parameter becomes relevant in regimes where the spectral peak
            is produced by **synchrotron self-absorption (SSA)**. In such cases,
            the inversion depends on the projected emitting area rather than
            the full volume.

            Physically, ``f_A`` represents the fraction of the projected
            spherical surface that contributes to the optically thick emission.

            For example:

            - ``f_A = 1`` corresponds to a uniformly emitting spherical surface.
            - ``f_A < 1`` may arise from patchy or filamentary emission regions.

            In many simple models one may take ``f_A ≈ f_V``, but the two
            parameters are conceptually distinct and correspond to **different
            geometric aspects** of the emitting region.

        gamma_bulk : float, optional
            Bulk Lorentz factor of the emitting region.

            If non-unity, the observed peak quantities are transformed into the
            comoving frame using relativistic Doppler corrections prior to
            performing the inversion.

            The implementation assumes **on-axis emission**, corresponding to a
            Doppler factor

            .. math::

                \delta(\Gamma, \theta=0).

        redshift : float, optional
            Cosmological redshift of the source.

            This parameter enters the conversion between observed and comoving
            frequencies and fluxes and also determines the cosmological
            distances if they are not supplied explicitly.

        luminosity_distance : quantity, optional
            Luminosity distance to the source.

            Must be convertible to cm.

            If supplied, this value is used directly in the inversion.

        angular_diameter_distance : quantity, optional
            Angular-diameter distance to the source.

            Must be convertible to cm.

            This quantity is required in some SSA regimes where the inversion
            depends explicitly on the **angular size of the emitting region**.

        proper_distance : quantity, optional
            Proper distance to the source.

            This can be used as an alternative specification of the cosmological
            distance.

        cosmology : ~astropy.cosmology.Cosmology, optional
            Cosmology used to compute missing distance quantities.

            The distances required by the inversion are resolved using the
            helper function :func:`resolve_cosmological_distances`, which
            processes the supplied cosmological parameters as follows:

            1. If explicit distances are provided (e.g. ``luminosity_distance``),
               those values are used directly.

            2. If only a redshift is provided, the necessary distances are
               computed using the supplied ``cosmology``.

            3. If no cosmology is provided, the default cosmology returned by
               :func:`get_cosmology` is used.

            This mechanism ensures that the inversion has access to all
            required cosmological distance measures while allowing users to
            specify whichever subset of parameters is most convenient.

        alpha : float, optional
            Electron pitch angle in radians.

            This parameter is required when ``pitch_average=False`` and
            specifies the angle between the electron velocity and the magnetic
            field.

        pitch_average : bool, optional
            If ``True``, the synchrotron emissivity is averaged over an
            isotropic distribution of pitch angles.

            In this case the explicit pitch angle ``alpha`` is ignored and the
            appropriate angle-averaged emissivity coefficients are used.

            Pitch-angle averaging is often appropriate when the electron
            distribution is isotropic in momentum space.

        Returns
        -------
        dict
            Dictionary containing inferred source parameters

            - ``R`` : source radius (cm)
            - ``B`` : magnetic field strength (G)

        Notes
        -----
        **Interpretation**

        These values are **model-dependent estimates**, not direct measurements.

        They represent the values of :math:`R` and :math:`B` that reproduce the
        observed peak under the assumptions of

        - the selected synchrotron spectral regime,
        - a single-zone homogeneous emitting region,
        - a power-law electron population,
        - equipartition-style microphysical closure.

        **When this method may fail**

        The inversion may produce misleading results if

        - the true spectrum contains multiple emitting zones,
        - the peak is produced by a process not described by the selected regime,
        - strong radiative transfer effects modify the spectrum,
        - relativistic equal-arrival-time surfaces dominate the emission,
        - the geometry deviates strongly from the assumed configuration.

        In such cases, users should treat the inferred parameters as
        **order-of-magnitude estimates only**.

        See Also
        --------
        determine_sed_regime
            Identify the synchrotron spectral regime.
        from_physics_to_params
            Forward closure relation.
        sed
            Evaluate the SED for a given parameter set.
        """
        # ------------------------------------------------------------------ #
        # Resolve cosmological information.
        # ------------------------------------------------------------------ #
        cosmology = get_cosmology(cosmology=cosmology)
        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        D_L = distances["luminosity_distance"]
        D_A = distances["angular_diameter_distance"]
        redshift = distances["redshift"]

        # ------------------------------------------------------------------ #
        # Unit enforcement and conversion to internal log-space representation.
        # ------------------------------------------------------------------ #
        log_F_peak = np.log(ensure_in_units(F_peak, "erg cm^-2 s^-1 Hz^-1"))
        log_nu_peak = np.log(ensure_in_units(nu_peak, "Hz"))
        log_D_L = np.log(ensure_in_units(D_L, "cm"))
        log_D_A = np.log(ensure_in_units(D_A, "cm"))

        # ------------------------------------------------------------------ #
        # Dispatch to the low-level inverse closure.
        # ------------------------------------------------------------------ #
        results = self._opt_from_params_to_physics(
            regime=regime,
            log_F_peak=log_F_peak,
            log_nu_peak=log_nu_peak,
            log_D_L=log_D_L,
            log_D_A=log_D_A,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            p=p,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            f_A=f_A,
            gamma_bulk=gamma_bulk,
            redshift=redshift,
            alpha=alpha,
            pitch_average=pitch_average,
        )

        return {
            "R": results["R"] * u.cm,
            "B": results["B"] * u.G,
        }


class SSA_SED_PowerLaw(SynchrotronSED):
    r"""
    Synchrotron self-absorbed (SSA) broken power-law spectral energy distribution.

    This class implements a phenomenological synchrotron spectral energy
    distribution characterized by a single smooth spectral break arising from
    synchrotron self-absorption (SSA). Below the break frequency, the spectrum is
    optically thick and follows a power-law slope of :math:`+5/2`; above the
    break, the spectrum is optically thin and follows a power-law slope of
    :math:`-(p-1)/2`, where :math:`p` is the power-law index of the electron
    energy distribution.

    The SED is implemented as a **smoothly broken power law**, ensuring numerical
    stability and differentiability across the SSA turnover. This makes the
    class well-suited for use in parameter inference, optimization, and
    likelihood-based modeling.

    In addition to forward evaluation of the SED, this class implements
    **analytic closure relations** that allow inversion between phenomenological
    SED parameters (the SSA break frequency and peak flux) and underlying
    physical properties of the emitting region (magnetic field strength and
    radius). These closure relations follow the formalism presented in
    :footcite:t:`demarchiRadioAnalysisSN2004C2022`.

    This class is intended for modeling radio synchrotron emission from
    approximately homogeneous emitting regions, such as those encountered in:

    - supernova radio afterglows,
    - compact object-powered transients,
    - synchrotron-emitting shells with a single dominant SSA turnover.

    No assumptions are made about time evolution, shock dynamics, or cooling
    regimes beyond those implicit in the broken power-law description.

    Notes
    -----
    - The break frequency :math:`\nu_{\rm brk}` is defined as the frequency at
      which the optically thick and optically thin asymptotic power laws
      intersect.
    - The closure relations implemented here assume a power-law electron energy
      distribution with finite bounds
      :math:`\gamma_{\rm min} \le \gamma \le \gamma_{\rm max}`.
    - The special case :math:`p = 2` is not supported due to logarithmic
      divergences in the electron energy integral and must be handled
      separately.

    See Also
    --------
    SynchrotronSED
        Abstract base class defining the interface for all synchrotron SED
        implementations.

    References
    ----------
    .. footbibliography::
    """

    # ================================================= #
    # Instantiation                                     #
    # ================================================= #
    def __init__(self):
        r"""Instantiate the SSA SED object."""
        # There are no class-wide constants to pre-compute for this SED.
        super().__init__()

    # ================================================ #
    # SED Function Implementation                      #
    # ================================================ #
    def _log_opt_sed(
        self,
        nu: "_ArrayLike",
        nu_brk: float,
        F_nu_brk: float,
        p: float,
        s: float,
    ) -> "_ArrayLike":
        r"""
        Low-level optimized SSA broken power-law SED.

        This method implements the synchrotron self-absorbed (SSA) broken power-law SED
        in a performance-optimized manner. It assumes that all inputs are provided as
        dimensionless scalars or NumPy arrays in CGS units. No unit validation or safety
        checks are performed.

        Parameters
        ----------
        nu: float or array-like
            Frequency at which to evaluate the SED (in Hz-equivalent CGS).
        nu_brk: float
            Break frequency (Hz-equivalent CGS).
        F_nu_brk: float
            Flux density at the break frequency (CGS units).
        p: float
            Power-law index of the electron energy distribution.
        s: float
            Smoothing parameter for the break.

        Returns
        -------
        float or array-like
            The computed SED value at the specified frequency. The output of this SED
            is in CGS units of erg s^-1 cm^-2 Hz^-1.
        """
        return smoothed_BPL(
            nu,
            F_nu_brk,
            nu_brk,
            -(p - 1) / 2,  # optically thin index
            (5 / 2),  # optically thick index
            s,  # smoothing parameter
        )

    def sed(
        self,
        nu: "_UnitBearingArrayLike",
        *,
        nu_brk: "_UnitBearingArrayLike",
        F_nu_brk: "_UnitBearingArrayLike",
        p: float,
        s: float,
    ):
        r"""
        Compute the synchrotron self-absorbed (SSA) broken power-law SED.

        This is the user-facing interface for the SSA broken power-law spectral
        energy distribution. It validates units, coerces inputs to CGS, and
        dispatches to the optimized low-level backend implementation
        :meth:`_opt_sed`.

        Parameters
        ----------
        nu : float, array-like, or astropy.units.Quantity
            Frequency at which to evaluate the SED. Default units are Hz.
        nu_brk : float or astropy.units.Quantity
            Break frequency of the SED. Default units are Hz.
        F_nu_brk : float or astropy.units.Quantity
            Flux density at the break frequency. Default units are
            erg s^-1 cm^-2 Hz^-1.
        p : float
            Power-law index of the electron energy distribution.
        s : float
            Smoothing parameter controlling the sharpness of the spectral break.

        Returns
        -------
        astropy.units.Quantity
            Flux density evaluated at ``nu`` with units of
            erg s^-1 cm^-2 Hz^-1.
        """
        # --- Unit validation and coercion --- #
        nu = ensure_in_units(nu, u.Hz)
        nu_brk = ensure_in_units(nu_brk, u.Hz)
        F_nu_brk = ensure_in_units(F_nu_brk, u.erg / u.s / u.cm**2 / u.Hz)

        # --- Call optimized backend --- #
        F_nu_cgs = self._log_opt_sed(
            nu=nu,
            nu_brk=nu_brk,
            F_nu_brk=F_nu_brk,
            p=p,
            s=s,
        )

        return F_nu_cgs * u.erg / u.s / u.cm**2 / u.Hz

    # ================================================ #
    # Closure Relations.                               #
    # ================================================ #
    # For this SED, we implement the closure relations to go from
    # the phenomenological parameters (nu_brk, F_nu_brk) to the physical
    # parameters (B, R) and vice versa. This is implemented exactly as
    # done in DeMarchi+22.
    def _opt_from_params_to_physics(
        self,
        nu_brk: Union[float, np.ndarray],
        F_nu_brk: Union[float, np.ndarray],
        distance: Union[float, np.ndarray],
        p: Union[float, np.ndarray] = 3.0,
        f: Union[float, np.ndarray] = 0.5,
        theta: Union[float, np.ndarray] = np.pi / 2,
        epsilon_B: Union[float, np.ndarray] = 0.1,
        epsilon_E: Union[float, np.ndarray] = 0.1,
        gamma_min: Union[float, np.ndarray] = 1.0,
        gamma_max: Union[float, np.ndarray] = 1e6,
    ) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        r"""
        Compute the magnetic field and radius of the emitting region from a broken power-law synchrotron SED.

        Parameters
        ----------
        nu_brk: float or array-like
            The break frequency :math:`\nu_{\rm brk}` where the synchrotron SED transitions from
            optically thick to optically thin. This should be provided in GHz. May be provided either
            as a scalar float (for a single SED) or as a 1-D array (for multiple SEDs).
        F_nu_brk: float or array-like
            The flux density :math:`F_{\nu_{\rm brk}}` at the break frequency. This should be provided
            in Jansky (Jy). May be provided either as a scalar float (for a single SED) or as a 1-D array (for multiple
            SEDs).
        distance: float or array-like
            The distance to the source in Megaparsecs (Mpc). May be provided either as a scalar float (for a single SED)
            or as a 1-D array (for multiple SEDs).
        p: float or array-like
            The power-law index :math:`p` of the electron energy distribution. By default, this is 3.0. If provided as a
            float, the value is used for all SEDs. If provided as an array, its shape must be compatible with that of
            ``nu_brk``.

            .. warning::

                This function does not handle the case where :math:`p = 2` due to singularities in the underlying
                equations.
                Users must ensure that :math:`p` is not equal to 2 when calling this function.

        f: float or array-like
            The filling factor :math:`f` of the emitting region. Default is ``0.5``. If provided as a float, the value
            is
            used for all SEDs. If provided as an array, its shape must be compatible with that of ``nu_brk``.
        theta: float or array-like
            The pitch angle :math:`\theta` between the magnetic field and the line of sight, in radians.
            Default is ``pi/2`` (i.e., perpendicular). If provided as a float, the value is used for all SEDs.
            If provided as an array, its shape must be compatible with that of ``nu_brk``.
        epsilon_B: float or array-like
            The fraction of post-shock energy in magnetic fields, :math:`\epsilon_B`.
        epsilon_E: float or array-like
            The fraction of post-shock energy in relativistic electrons, :math:`\epsilon_E`.
        gamma_min: float or array-like
            The minimum Lorentz factor :math:`\gamma_{\rm min}` of the electron energy.
        gamma_max: float or array-like
            The maximum Lorentz factor :math:`\gamma_{\rm max}` of the electron energy.

        Returns
        -------
        B: float or array-like
            The computed magnetic field strength :math:`B` in Gauss. The shape matches that of the input parameters.
        R: float or array-like
            The computed radius of the emitting region :math:`R` in cm. The shape matches that of the input parameters.

        Notes
        -----
        See notes in the user-facing wrapper function `compute_BR_from_BPL` for details on the
        underlying physics and assumptions.

        Effective 1/19/26: We have modified this to work exclusively in log space due to issues with floating point
        truncation errors. Catastrophic cancellation caused significant loss of accuracy.
        """
        return _compute_ssa_BR_from_spectrum_dm22(
            nu_brk=nu_brk,
            F_nu_brk=F_nu_brk,
            distance=distance,
            p=p,
            f=f,
            theta=theta,
            epsilon_B=epsilon_B,
            epsilon_E=epsilon_E,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )

    def from_params_to_physics(
        self,
        nu_brk: Union[float, np.ndarray, u.Quantity],
        F_nu_brk: Union[float, np.ndarray, u.Quantity],
        distance: Union[float, np.ndarray, u.Quantity],
        *,
        p: Union[float, np.ndarray] = 3.0,
        f: Union[float, np.ndarray] = 0.5,
        theta: Union[float, np.ndarray] = np.pi / 2,
        epsilon_B: Union[float, np.ndarray] = 0.1,
        epsilon_E: Union[float, np.ndarray] = 0.1,
        gamma_min: Union[float, np.ndarray] = 1.0,
        gamma_max: Union[float, np.ndarray] = 1e6,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the magnetic field strength and emitting radius from a broken power-law SED.

        This function provides a **user-facing, unit-aware interface** for computing
        the physical properties of a synchrotron-emitting region whose radio spectrum
        exhibits a turnover due to synchrotron self-absorption (SSA). Internally, it
        wraps a low-level optimized routine that implements the analytic inversion
        described in :footcite:t:`demarchiRadioAnalysisSN2004C2022` (DM22).

        The function accepts scalar values, NumPy arrays, or Astropy ``Quantity`` objects
        and performs the necessary unit coercion, validation, and shape checking before
        dispatching to the optimized backend.

        Parameters
        ----------
        nu_brk : float, array-like, or astropy.units.Quantity
            The SSA break (turnover) frequency :math:`\nu_{\rm brk}` separating the
            optically thick and optically thin synchrotron regimes. Default units are GHz, but may
            be overridden by providing ``nu_brk`` as a :class:`astropy.units.Quantity` object. May be
            provided as either a scalar (for a single SED) or a 1-D array (for multiple SEDs).

        F_nu_brk : float, array-like, or astropy.units.Quantity
            The flux density :math:`F_{\nu_{\rm brk}}` at the break frequency. Default units
            are Jansky (Jy), but may be overridden by providing ``F_nu_brk`` as a
            :class:`astropy.units.Quantity` object. May be provided as either a scalar
            (for a single SED) or a 1-D array (for multiple SEDs). If provided as an array, shape
            must be compatible with that of ``nu_brk``.

        distance : float, array-like, or astropy.units.Quantity
            The (luminosity) distance to the source. By default, units are Megaparsecs (Mpc),
            but may be overridden by providing ``distance`` as a :class:`astropy.units.Quantity` object.
            May be provided as either a scalar (for a single SED) or a 1-D array (for multiple SEDs). If provided
            as an array, shape must be compatible with that of ``nu_brk``.

        p : float or array-like, optional
            The power-law index :math:`p` of the electron Lorentz factor distribution,
            defined by

            .. math::

                N(\Gamma)\, d\Gamma = K_e \Gamma^{-p}\, d\Gamma.

            Default is ``3.0``.

            .. warning::

                This function **does not support** the case :math:`p = 2`, for which
                the electron energy integral is logarithmically divergent and requires
                a separate analytic treatment. If any value of ``p`` is exactly equal
                to 2, a ``ValueError`` is raised.

        f : float or array-like, optional
            Volume filling factor of the synchrotron-emitting region. Default is ``0.5``.

        theta : float or array-like, optional
            Pitch angle :math:`\theta` between the magnetic field and the electron
            velocity, in radians. Default is ``\pi/2`` (isotropic average).

        epsilon_B : float or array-like, optional
            Fraction of post-shock internal energy in magnetic fields,
            :math:`\epsilon_B`. Default is ``0.1``.

        epsilon_E : float or array-like, optional
            Fraction of post-shock internal energy in relativistic electrons,
            :math:`\epsilon_E`. Default is ``0.1``.

        gamma_min : float or array-like, optional
            Minimum electron Lorentz factor :math:`\gamma_{\rm min}`. Default is ``1``.

        gamma_max : float or array-like, optional
            Maximum electron Lorentz factor :math:`\gamma_{\rm max}`. Default is ``1e6``.

            This parameter only affects the calculation when :math:`p < 2`.

        Returns
        -------
        B : astropy.units.Quantity
            The inferred magnetic field strength :math:`B` in **Gauss**.

        R : astropy.units.Quantity
            The inferred radius of the synchrotron-emitting region :math:`R`
            in **cm**.

        Notes
        -----
        This function follows the formalism laid out in :footcite:t:`demarchiRadioAnalysisSN2004C2022` (DM22) to compute
        the magnetic field strength and radius of the synchrotron-emitting region from the observed break frequency and
        flux density of a broken power-law SED. The calculations assume a power-law distribution of electron energies
        characterized by the index :math:`p`.

        Letting :math:`\nu_{\rm brk}` be the break frequency (in GHz) between the SSA-thick and SSA-thin regimes, and
        :math:`F_{\nu_{\rm brk}}` be the flux density (in Jy) at that frequency, the magnetic field strength :math:`B`
        (in Gauss) and radius :math:`R` (in cm) of the emitting region can be computed
        by requiring that the asymptotic behavior of
        the optically thick and thin synchrotron spectra match at :math:`\nu_{\rm brk}`. The equations used are
        equations
        (16) and (17) from DM22 with minor alterations.

        In treating the electron energy distribution, some additional care is taken based on the value of :math:`p`. In
        particular, we assume a power-law distribution of electron Lorentz factors :math:`\Gamma` such that

        .. math::

            N(\Gamma) d\Gamma = K_e \Gamma^{-p} d\Gamma,\;\; \Gamma_{\rm min} \leq \Gamma \leq \Gamma_{\rm max},

        where :math:`K_e` is the normalization constant, :math:`\Gamma_{\rm min}` is the minimum Lorentz factor,
        and :math:`\Gamma_{\rm max}` is the maximum Lorentz factor. For values of :math:`p > 2`, the total energy is
        dominated by electrons near :math:`\Gamma_{\rm min}`, while for :math:`p < 2`, it is dominated by those near
        :math:`\Gamma_{\rm max}`.

        To account for this, when :math:`p > 2`, we enforce :math:`\Gamma_{\rm max} = \infty` in the energy integral,
        while
        for :math:`p < 2`, we enforce the upper limit on the energy integral to be :math:`\Gamma_{\rm max}`. This leads
        to
        a correction factor :math:`\delta` defined as:

        .. math::

            \delta = \begin{cases} 1, & p > 2 \[6pt]
            \left(\frac{\Gamma_{\rm max}}{\Gamma_{\rm min}}\right)^{2 - p} - 1, & p < 2 \end{cases}

        which modifies the expressions for :math:`B` and :math:`R` accordingly.

        References
        ----------

        .. footbibliography::

        """
        # Validate units of all unit bearing quantities and coerce them to the expected
        # units for the optimized backend.
        nu_brk = ensure_in_units(nu_brk, u.GHz)
        F_nu_brk = ensure_in_units(F_nu_brk, u.Jy)
        distance = ensure_in_units(distance, u.Mpc)

        # Check the validity of p values. We need to ensure that ``p`` behaves as an array at
        # this point, so we cast it explicitly.
        p = np.asarray(p)
        if np.any(p == 2):
            raise ValueError(
                "compute_BR_from_BPL does not support p = 2. "
                "Use p slightly above or below 2, or implement a dedicated "
                "logarithmic normalization for this case."
            )

        # Dispatch to the optimized backend.
        B, R = self._opt_from_params_to_physics(
            nu_brk=nu_brk,
            F_nu_brk=F_nu_brk,
            distance=distance,
            p=p,
            f=f,
            theta=theta,
            epsilon_B=epsilon_B,
            epsilon_E=epsilon_E,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        return B * u.Gauss, R * u.cm

    def _opt_from_physics_to_params(
        self,
        B: Union[float, np.ndarray],
        R: Union[float, np.ndarray],
        distance: Union[float, np.ndarray],
        p: Union[float, np.ndarray] = 3.0,
        f: Union[float, np.ndarray] = 0.5,
        theta: Union[float, np.ndarray] = np.pi / 2,
        epsilon_B: Union[float, np.ndarray] = 0.1,
        epsilon_E: Union[float, np.ndarray] = 0.1,
        gamma_min: Union[float, np.ndarray] = 1.0,
        gamma_max: Union[float, np.ndarray] = 1e6,
    ):
        r"""
        Compute the synchrotron self-absorption frequency and peak flux.

        This is a **low-level, performance-optimized routine** that assumes all
        inputs are provided as dimensionless scalars or NumPy arrays in CGS units.
        No unit validation or safety checks are performed.

        The function analytically eliminates the normalization of the electron
        energy distribution using microphysical energy-partition assumptions,
        introducing a correction factor ``delta`` to account for the convergence
        of the electron energy integral when :math:`p < 2`.

        Parameters
        ----------
        B : float or array-like
            Magnetic field strength in Gauss.

        R : float or array-like
            Radius of the emitting region in cm.

        distance : float or array-like
            Distance to the source in cm.

        p, f, theta, epsilon_B, epsilon_E, gamma_min, gamma_max
            See the user-facing function ``compute_BPL_SED_from_BR``.

        Returns
        -------
        nu_brk : float or array-like
            Synchrotron self-absorption break frequency (Hz-equivalent CGS).

        F_nu_brk : float or array-like
            Peak flux density at the break frequency (CGS-equivalent).

        Notes
        -----
        See notes in the user-facing wrapper function `compute_BR_from_BPL` for details on the
        underlying physics and assumptions.
        """
        # Validate input parameters. We do NOT check explicitly for shape correctness of the arrays, instead opting
        # to allow an error to rise naturally if the shapes are incompatible during computation. We do want to ensure
        # that all of the constants are cast properly for array operations and masking.
        p, gamma_min, gamma_max = (
            np.asarray(p, dtype="f8"),
            np.asarray(gamma_min, dtype="f8"),
            np.asarray(gamma_max, dtype="f8"),
        )

        # Obtain the synchrotron coefficients relevant for this scenario. We need c_1,c_5, and c_6.
        c_5 = compute_c5_parameter(p)  # Should match shape of p.
        c_6 = compute_c6_parameter(p)  # Should match shape of p.
        c_1 = c_1_cgs  # Scalar constant.

        # Construct the array for delta. See the documentation notes for details on the procedure here / physical
        # motivation. To do this efficiently, we pre-allocate the array and then fill in values based on the conditions.
        # Because we pre-allocate with ones, we only need to fill in values where p < 2. In THIS IMPLEMENTATION, we
        # ignore
        # cases where p == 2 for efficiency; these should be screened for upstream if needed.
        delta = np.ones_like(p)
        _p_lt_2_mask = p < 2
        delta[_p_lt_2_mask] = (gamma_max[_p_lt_2_mask] / gamma_min[_p_lt_2_mask]) ** (
            2 - p[_p_lt_2_mask]
        ) - 1  # Fill in values where p < 2.

        # Construct the "p_norm" term used in the B and R calculations. This allows us to handle the two branches
        # of the solution (p < 2 and p > 2) in a unified way.
        # Note that we take the absolute value here to avoid issues with negative bases and fractional exponents.
        # The p == 2 case is handled upstream.
        p_norm = np.abs(p - 2.0)

        # Compute the electron energy floor using the gamma_min parameter and the standard formula.
        E_l = electron_rest_energy_cgs * gamma_min

        # Calculate nu_brk. We break this into parts for clarity. See the notes on this function for an explanation of
        # where this formula comes from.
        _nu_brk_coeff = 2 * c_1
        _nu_brk_t1 = R * f * (epsilon_E / (delta * epsilon_B)) * p_norm / (6 * np.pi)
        _nu_brk_t2 = c_6 ** (2 / (p + 4))
        _nu_brk_t3 = np.sin(theta) ** ((p + 2) / (p + 4))
        _nu_brk_E_l_exp = 2 * (p - 2) / (p + 4)
        _nu_brk_B_exp = (p + 6) / (p + 4)

        nu_brk = (
            _nu_brk_coeff
            * (_nu_brk_t1 ** (2 / (p + 4)))
            * (E_l**_nu_brk_E_l_exp)
            * (B**_nu_brk_B_exp)
            * _nu_brk_t2
            * _nu_brk_t3
        )

        # Calculate F_nu_brk by inserting nu_brk into either the optically thick or thin formula.
        # We choose the optically thick formula here (equation 14 of DeMarchi+22) for consistency.
        _F_nu_brk_coeff = (c_5 / c_6) * np.pi * (R / distance) ** 2
        _F_nu_brk_t1 = (B * np.sin(theta)) ** (-1 / 2)
        _F_nu_brk_t2 = (nu_brk / (2 * c_1)) ** (5 / 2)

        F_nu_brk = _F_nu_brk_coeff * _F_nu_brk_t1 * _F_nu_brk_t2

        return nu_brk, F_nu_brk

    def from_physics_to_params(
        self,
        B: Union[float, np.ndarray, u.Quantity],
        R: Union[float, np.ndarray, u.Quantity],
        distance: Union[float, np.ndarray, u.Quantity],
        *,
        p: Union[float, np.ndarray] = 3.0,
        f: Union[float, np.ndarray] = 0.5,
        theta: Union[float, np.ndarray] = np.pi / 2,
        epsilon_B: Union[float, np.ndarray] = 0.1,
        epsilon_E: Union[float, np.ndarray] = 0.1,
        gamma_min: Union[float, np.ndarray] = 1.0,
        gamma_max: Union[float, np.ndarray] = 1e6,
    ):
        r"""
        Compute the synchrotron self-absorption break frequency and peak flux.

        This is a **low-level, performance-optimized routine** that assumes all
        inputs are provided as dimensionless scalars or NumPy arrays in CGS units.
        No unit validation or safety checks are performed.

        The function analytically eliminates the normalization of the electron
        energy distribution using microphysical energy-partition assumptions,
        introducing a correction factor ``delta`` to account for the convergence
        of the electron energy integral when :math:`p < 2`.

        Parameters
        ----------
        B : float or array-like
            Magnetic field strength in Gauss.

        R : float or array-like
            Radius of the emitting region in cm.

        distance : float or array-like
            Distance to the source in Mpc.

        p, f, theta, epsilon_B, epsilon_E, gamma_min, gamma_max
            See the user-facing function ``compute_BPL_SED_from_BR``.

        Returns
        -------
        nu_brk : float or array-like
            Synchrotron self-absorption break frequency (Hz-equivalent CGS).

        F_nu_brk : float or array-like
            Peak flux density at the break frequency (CGS-equivalent).

        Notes
        -----
        In the optically thick regime, the synchrotron SED follows a power law

        .. math::

            F_\nu = \frac{c_5}{c_6} \left(B\sin\theta\right)^{-1/2} \left(\frac{\nu}{2c_1}\right)^{5/2}
            \frac{\pi R^2}{d^2}.

        In the optically thin regime, the SED follows a different power law:

        .. math::

            F_\nu = \frac{4\pi f R^3}{3 d^2} c_5 N_0 \left(B\sin \theta\right)^{(p+1)/2}
            \left(\frac{\nu}{2c_1}\right)^{-(p-1)/2}.

        We define the break frequency :math:`\nu_{\rm brk}` as the frequency where these two power laws intersect and
        the
        corresponding peak flux density :math:`F_{\nu_{\rm brk}}`. By equating the two expressions for :math:`F_\nu` at
        :math:`\nu = \nu_{\rm brk}`, we can solve for :math:`\nu_{\rm brk}` and :math:`F_{\nu_{\rm brk}}` in terms of
        the physical parameters :math:`B`, :math:`R`, and :math:`d`.

        Equation these two equations yields

        .. math::

            \nu_{\rm brk} = 2 c_1 \left[\frac{4}{3} c_6 f N_0\right]^{2/(p+4)} R^{2/(p+4)}
            \left(B\sin\theta\right)^{(p+2)/(p+4)}

        Inserting :math:`\nu_{\rm brk}` back into either expression for :math:`F_\nu` gives the peak flux density
        :math:`F_{\nu_{\rm brk}}`.

        The normalization :math:`N_0` of the electron distribution is eliminated
        analytically by equating the total electron energy density to a fraction
        :math:`\epsilon_E` of the post-shock internal energy density, with magnetic
        energy fraction :math:`\epsilon_B`.

        For :math:`p \neq 2`, this introduces a correction factor

        .. math::

            \delta =
            \begin{cases}
            1, & p > 2 \
            (\gamma_{\max}/\gamma_{\min})^{2-p} - 1, & p < 2
            \end{cases}

        which accounts for the convergence properties of the electron energy integral.
        """
        # Validate units of all unit bearing quantities and coerce them to the expected
        # units for the optimized backend.
        if hasattr(B, "units"):
            B = B.to_value(u.Gauss)
        if hasattr(R, "units"):
            R = R.to_value(u.cm)
        if hasattr(distance, "units"):
            distance = distance.to_value(u.cm)

        # Check the validity of p values. We need to ensure that ``p`` behaves as an array at
        # this point, so we cast it explicitly.
        p = np.asarray(p)
        if np.any(p == 2):
            raise ValueError(
                "compute_BR_from_BPL does not support p = 2. "
                "Use p slightly above or below 2, or implement a dedicated "
                "logarithmic normalization for this case."
            )

        # Dispatch to the optimized backend.
        nu, F_nu = self._opt_from_physics_to_params(
            B=B,
            R=R,
            distance=distance,
            p=p,
            f=f,
            theta=theta,
            epsilon_B=epsilon_B,
            epsilon_E=epsilon_E,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        return nu * u.Hz, F_nu * u.erg / (u.s * u.cm**2 * u.Hz)
