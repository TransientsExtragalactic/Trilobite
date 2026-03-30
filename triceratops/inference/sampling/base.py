"""Base classes for sampling strategies in Triceratops inference."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .result import SamplingResult

if TYPE_CHECKING:
    from ..problem import InferenceProblem
    from .result import SamplingResult


class Sampler(ABC):
    """
    Abstract base class for sampling strategies in Triceratops inference.

    This class defines the interface and common functionality for all sampling algorithms
    used in the Triceratops inference framework. Specific sampling methods should inherit
    from this class and implement the required abstract methods. The :class:`Sampler` class
    is designed to be very general, allowing for a wide range of sampling techniques to be
    integrated into the Triceratops ecosystem.
    """

    # -------------------------------------- #
    # Initialization                         #
    # -------------------------------------- #
    def __init__(self, inference_problem: "InferenceProblem", **kwargs):
        """
        Initialize the sampler from a given inference problem.

        This sets up the sampler with the necessary configuration to perform sampling
        on the provided inference problem. Depending on the specific sampler implementation,
        different additional arguments may be required through ``kwargs``. The ``__init__`` call
        should handle all necessary setup for the sampler to be ready to run.

        Parameters
        ----------
        inference_problem: ~triceratops.inference.problem.InferenceProblem
            The inference problem to sample from.
        **kwargs:
            Additional arguments specific to the sampler implementation.
        """
        # Declare the inference problem
        self._problem: InferenceProblem = inference_problem

        # Perform validation on the inference problem. By default,
        # this does nothing, but specific samplers may override this method.
        self._validate_inference_problem()

    # -------------------------------------- #
    # Properties                             #
    # -------------------------------------- #
    @property
    def problem(self) -> "InferenceProblem":
        """
        The inference problem associated with this sampler.

        Returns
        -------
        ~triceratops.inference.problem.InferenceProblem
            The inference problem configured for this sampler.
        """
        return self._problem

    # -------------------------------------- #
    # Abstract Methods                       #
    # -------------------------------------- #
    @abstractmethod
    def run(self, **kwargs) -> SamplingResult:
        """
        Run the sampling procedure.

        This method executes the sampling algorithm on the configured inference problem.
        It should handle all aspects of the sampling process, including initialization,
        execution, and result collection. The specific arguments required for running
        the sampler may vary depending on the implementation and should be provided
        through ``kwargs``.

        Returns
        -------
        ~triceratops.inference.sampling.result.SamplingResult
            The result of the sampling procedure, encapsulated in a SamplingResult object.
        """
        pass

    @abstractmethod
    def _validate_inference_problem(self):
        """
        Validate the inference problem for compatibility with this sampler.

        This method checks whether the provided inference problem meets the requirements
        for the specific sampling algorithm. If the problem is not compatible, this method
        should raise an appropriate exception. By default, this method does nothing,
        but specific samplers may override it to implement their own validation logic.
        """
        pass
