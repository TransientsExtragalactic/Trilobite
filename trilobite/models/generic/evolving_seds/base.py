r"""
Base classes for evolving SED models.

This module defines :class:`EvolvingSEDModel`, an abstract base class for
phenomenological, time-evolving spectral energy distribution (SED) models
of the separable form:

.. math::

    F_{\nu}(\nu, t) = N(t) \Phi(\nu, b(t)),

where ``N(t)`` is a flux-density normalization and ``Phi`` is a dimensionless
spectral shape function controlled by one or more evolving break parameters.

All internal computations are performed in natural log-space for numerical
stability across many decades in frequency and time.
"""

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

from ...core.base import Model, ModelVariable

if TYPE_CHECKING:
    from ..._typing import _ModelParametersInputRaw, _ModelVariablesInputRaw


__all__ = ["EvolvingSEDModel"]
# ============================================================
# Output Definition
# ============================================================
_EvolvingSEDOutputs = namedtuple("EvolvingSEDOutputs", ["flux_density"])


# =============================================== #
# Base Class Definition
# =============================================== #
class EvolvingSEDModel(Model, ABC):
    r"""
    Abstract base class for separable, time-evolving spectral energy distribution (SED) models evaluated in log-space.

    This class defines models of the form

    .. math::

        F_\nu(\nu, t) = N(t) \, \Phi(\nu, t),

    where

    - :math:`N(t)` is a time-dependent normalization with units of flux density,
    - :math:`\Phi(\nu,t)` is a dimensionless spectral shape function,
    - the spectral shape may depend on one or more evolving break parameters.

    All internal calculations are performed using natural logarithms
    for numerical stability.

    Notes
    -----
    Subclasses must implement three abstract methods:

    ``_compute_log_norm(log_t, parameters)``
        Returns ``ln(N(t))`` evaluated on the broadcasted time grid.

    ``_compute_log_breaks(log_t, parameters)``
        Returns a dictionary mapping break names to their natural logarithms. These are then used
        in the log shape calculation. Thus, if the SED requires a break at some ``nu_brk(t)``, this
        method can return ``{"nu_brk": ln(nu_brk(t))}``.

    ``_compute_log_shape(log_nu, log_t, log_breaks, parameters)``
        Returns ``ln(Phi(nu,t))`` evaluated on the broadcasted
        frequency-time grid.

    These methods must:

    - Accept *already coerced*, unitless numerical inputs.
    - Respect NumPy broadcasting rules.
    - Return arrays compatible with the broadcasted shape of
      ``frequency`` and ``time``.

    The base class handles:

    - Unit coercion and validation
    - Bounds checking
    - Broadcasting of frequency and time
    - Reconstruction of linear-space outputs
    - Attachment of physical units
    """

    # ============================================================
    # Variable and Output Definitions
    # ============================================================
    VARIABLES = (
        ModelVariable("frequency", base_units=u.Hz, description="Observing frequency."),
        ModelVariable("time", base_units=u.s, description="Time since explosion."),
    )
    OUTPUTS = _EvolvingSEDOutputs
    UNITS = OUTPUTS(flux_density=u.Jy)

    # ============================================================
    # Abstract procedural API (log-space)
    # ============================================================
    @abstractmethod
    def _compute_log_norm(
        self,
        log_t: np.ndarray,
        parameters: "_ModelParametersInputRaw",
    ) -> np.ndarray:
        """
        Compute the natural log of the normalization ``N(t)``.

        Parameters
        ----------
        log_t : ndarray
            Natural logarithm of time, ``ln(t)``.
        parameters : dict-like
            Coerced, unitless model parameters.

        Returns
        -------
        ndarray
            Natural logarithm of the normalization, ``ln(N(t))``.
            Must broadcast with ``log_t``.
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_log_breaks(
        self,
        log_t: np.ndarray,
        parameters: "_ModelParametersInputRaw",
    ) -> dict[str, np.ndarray]:
        """
        Compute evolving spectral breaks in log-space.

        Parameters
        ----------
        log_t : ndarray
            Natural logarithm of time, ``ln(t)``.
        parameters : dict-like
            Coerced, unitless model parameters.

        Returns
        -------
        dict[str, ndarray]
            Dictionary mapping break names to their natural logarithms,
            e.g. ``{"nu_brk": ln(nu_brk(t))}``. Each value must broadcast
            with ``log_t``.
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_log_shape(
        self,
        log_nu: np.ndarray,
        log_t: np.ndarray,
        log_breaks: dict[str, np.ndarray],
        parameters: "_ModelParametersInputRaw",
    ) -> np.ndarray:
        """
        Compute the natural log of the dimensionless spectral shape ``Phi(nu,t)``.

        Parameters
        ----------
        log_nu : ndarray
            Natural logarithm of frequency, ``ln(nu)``.
        log_t : ndarray
            Natural logarithm of time, ``ln(t)``.
        log_breaks : dict[str, ndarray]
            Output of :meth:`_compute_log_breaks`.
        parameters : dict-like
            Coerced, unitless model parameters.

        Returns
        -------
        ndarray
            Natural logarithm of the dimensionless shape, ``ln(Phi)``.
            Must broadcast with ``log_nu`` and ``log_t``.
        """
        raise NotImplementedError

    # ============================================================
    # Private (low-level) interface for evaluation.
    # ============================================================
    def _forward_model(
        self,
        variables: "_ModelVariablesInputRaw",
        parameters: "_ModelParametersInputRaw",
    ) -> "_EvolvingSEDOutputs":
        """
        Evaluate the full SED (unitless core).

        Notes
        -----
        This method assumes ``variables`` and ``parameters`` have already been
        coerced to raw numerical values and validated.
        """
        # Broadcast frequency and time to the same shape for evaluation.
        log_nu, log_t = np.broadcast_arrays(
            np.log(variables["frequency"]),
            np.log(variables["time"]),
        )

        # Extract the log normalization and log breaks, then compute the log shape.
        log_norm = self._compute_log_norm(log_t, parameters)
        log_breaks = self._compute_log_breaks(log_t, parameters)
        log_shape = self._compute_log_shape(log_nu, log_t, log_breaks, parameters)

        log_flux = log_norm + log_shape

        return self.OUTPUTS(flux_density=np.exp(log_flux))

    def _forward_model_sed_shape(
        self,
        variables: "_ModelVariablesInputRaw",
        parameters: "_ModelParametersInputRaw",
    ) -> np.ndarray:
        """
        Evaluate only the dimensionless shape Phi(nu,t) (unitless core).

        Returns
        -------
        ndarray
            Dimensionless shape Phi(nu,t), with broadcasted shape of
            ``frequency`` and ``time``.
        """
        log_nu, log_t = np.broadcast_arrays(
            np.log(variables["frequency"]),
            np.log(variables["time"]),
        )

        log_breaks = self._compute_log_breaks(log_t, parameters)
        log_shape = self._compute_log_shape(log_nu, log_t, log_breaks, parameters)

        return np.exp(log_shape)

    def _forward_model_breaks(
        self,
        variables: "_ModelVariablesInputRaw",
        parameters: "_ModelParametersInputRaw",
    ) -> dict[str, np.ndarray]:
        """
        Evaluate breaks(t) in *linear* space (unitless core).

        Returns
        -------
        dict[str, ndarray]
            Dictionary mapping break names to break values (not logs),
            as raw numerical values.
        """
        log_t = np.log(variables["time"])
        log_breaks = self._compute_log_breaks(log_t, parameters)
        return {name: np.exp(val) for name, val in log_breaks.items()}

    def _forward_model_norm(
        self,
        variables: "_ModelVariablesInputRaw",
        parameters: "_ModelParametersInputRaw",
    ) -> np.ndarray:
        """
        Evaluate normalization N(t) in linear space (unitless core).

        Returns
        -------
        ndarray
            Normalization values as raw numerical values (compatible with
            ``UNITS.flux_density``).
        """
        log_t = np.log(variables["time"])
        log_norm = self._compute_log_norm(log_t, parameters)
        return np.exp(log_norm)

    # ============================================================
    # Shared coercion/validation helper (mirrors Model.forward_model preflight)
    # ============================================================

    def _prepare_inputs(self, variables, parameters):
        """
        Coerce inputs and validate bounds.

        Parameters
        ----------
        variables : dict
            Input variables (may be unit-bearing).
        parameters : dict
            Input parameters (may be unit-bearing).

        Returns
        -------
        tuple(dict, dict)
            Coerced, unitless variables and parameters.
        """
        vars_ = self.coerce_model_variables(variables)
        pars_ = self.coerce_model_parameters(parameters)

        self._check_parameter_bounds_base(pars_)
        self._check_model_bounds_base(vars_, pars_)

        return vars_, pars_

    # ============================================================
    # Public helper methods (coercion + bounds + units)
    # ============================================================

    def compute_sed_shape(self, variables, parameters) -> np.ndarray:
        r"""
        Evaluate the dimensionless spectral shape :math:`\Phi(\nu,t)`.

        This method computes only the dimensionless shape component of the
        evolving SED model, without applying the time-dependent normalization.
        It performs full variable/parameter coercion, bounds checking, and
        broadcasting consistent with :meth:`~trilobite.models.core.base.Model.forward_model`.

        Parameters
        ----------
        variables : dict
            Model variables. Must include:

            - ``"frequency"`` : array-like or `~astropy.units.Quantity`
              Observing frequency.
            - ``"time"`` : array-like or `~astropy.units.Quantity`
              Time since explosion.

            The arrays will be broadcast following
            :func:`numpy.broadcast_arrays`.

        parameters : dict
            Model parameters. May include units. Values are coerced to the
            model's base units before evaluation.

        Returns
        -------
        ndarray
            Dimensionless spectral shape evaluated on the broadcasted grid
            defined by ``frequency`` and ``time``.

            The returned array has the broadcasted shape of the inputs.

        Raises
        ------
        ValueError
            If required variables or parameters are missing, or if any
            bounds checks fail.

        Notes
        -----
        The full SED is defined as

        .. math::

            F_\nu(\nu,t) = N(t) \, \Phi(\nu,t).

        This method returns only :math:`\Phi(\nu,t)`.

        All computations are performed in natural log-space internally for
        numerical stability.

        Examples
        --------
        .. code-block:: python

            import numpy as np
            import astropy.units as u

            model = MyEvolvingSEDModel()

            nu = np.logspace(8, 11, 100) * u.Hz
            t = 10.0 * u.day

            shape = model.compute_sed_shape(
                {"frequency": nu, "time": t}, parameters
            )

        See Also
        --------
        compute_sed_norm
        compute_sed_breaks
        ~trilobite.models.core.base.Model.forward_model
        """
        vars_, pars_ = self._prepare_inputs(variables, parameters)
        return self._forward_model_sed_shape(vars_, pars_)

    def compute_sed_norm(self, variables, parameters) -> u.Quantity:
        r"""
        Evaluate the time-dependent normalization :math:`N(t)`.

        This method computes only the normalization component of the SED,
        with units attached. It performs full input coercion and bounds
        validation consistent with :meth:`~trilobite.models.core.base.Model.forward_model`.

        Parameters
        ----------
        variables : dict
            Model variables. Must include:

            - ``"time"`` : array-like or `~astropy.units.Quantity`
              Time since explosion.

            If ``"frequency"`` is provided, it is ignored.

        parameters : dict
            Model parameters. May include units. Values are coerced to base
            units prior to evaluation.

        Returns
        -------
        astropy.units.Quantity
            Normalization :math:`N(t)` with units of flux density
            (typically Jy).

            The returned quantity follows the broadcasted shape of ``time``.

        Raises
        ------
        ValueError
            If required variables or parameters are missing, or if bounds
            validation fails.

        Notes
        -----
        The full SED is given by

        .. math::

            F_\nu(\nu,t) = N(t) \, \Phi(\nu,t).

        This method returns only :math:`N(t)`.

        Internally, normalization is computed in log-space and exponentiated
        before units are attached.

        Examples
        --------
        .. code-block:: python

            import astropy.units as u

            model = MyEvolvingSEDModel()

            norm = model.compute_sed_norm(
                {"time": 5.0 * u.day}, parameters
            )

        See Also
        --------
        compute_sed_shape
        ~trilobite.models.core.base.Model.forward_model
        """
        vars_, pars_ = self._prepare_inputs(variables, parameters)
        raw = self._forward_model_norm(vars_, pars_)

        unit = self.UNITS.flux_density
        return raw * unit

    def compute_sed_breaks(self, variables, parameters) -> dict[str, u.Quantity]:
        """
        Evaluate evolving spectral break parameters.

        This method computes the break parameters in linear space and
        attaches physical units. Breaks are returned as a dictionary
        mapping break names to `~astropy.units.Quantity`.

        Parameters
        ----------
        variables : dict
            Model variables. Must include:

            - ``"time"`` : array-like or `~astropy.units.Quantity`
              Time since explosion.

        parameters : dict
            Model parameters. May include units. Values are coerced to base
            units before evaluation.

        Returns
        -------
        dict[str, astropy.units.Quantity]
            Dictionary mapping break names to physical values.

            If a break name is not found in
            :attr:`~trilobite.models.generic.evolving_seds.base.EvolvingSEDModel.BREAK_UNITS`,
            it is assumed to have units of Hz.

        Raises
        ------
        ValueError
            If required variables or parameters are missing, or if bounds
            validation fails.

        Notes
        -----
        Break values are internally computed in log-space and exponentiated
        before units are attached.

        Subclasses may define

        .. code-block:: python

            BREAK_UNITS = {
                "nu_brk": u.Hz,
                "t_0": u.s,
            }

        to control the physical units assigned to each break.

        Examples
        --------
        .. code-block:: python

            import astropy.units as u

            model = MyEvolvingSEDModel()

            breaks = model.compute_sed_breaks(
                {"time": 10.0 * u.day}, parameters
            )

            nu_brk = breaks["nu_brk"]

        See Also
        --------
        compute_sed_norm
        compute_sed_shape
        """
        vars_, pars_ = self._prepare_inputs(variables, parameters)
        raw = self._forward_model_breaks(vars_, pars_)

        return raw
