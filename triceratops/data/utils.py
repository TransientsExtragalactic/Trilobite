"""
Utility module for various data related manipulations in Triceratops.

This module contains various utility functions for handling data in Triceratops. This includes
functions for loading and processing observational data, as well as tools for managing and
organizing data within the Triceratops framework. These utilities are designed to facilitate
the integration of observational data into Triceratops models and analyses, making it easier for
users to work with their data in a consistent and efficient manner.
"""


def infer_err_from_non_detection(flux_limit: float, sigma: float = 3.0) -> float:
    """
    Infer an error from a non-detection flux limit.

    This function takes a flux limit from a non-detection and infers an error value that can be
    used in likelihood calculations. The inferred error is typically set to the flux limit divided
    by the specified sigma level, which represents the confidence level of the non-detection.

    Parameters
    ----------
    flux_limit : float
        The flux limit from the non-detection.
    sigma : float, optional
        The sigma level to use for inferring the error (default is 3.0).

    Returns
    -------
    float
        The inferred error value based on the non-detection flux limit.
    """
    return flux_limit / sigma
