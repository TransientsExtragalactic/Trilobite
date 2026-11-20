"""
Type aliases for the inputs and outputs of the model interfaces.

The ``Raw`` aliases describe the unitless dictionaries of floats and arrays handed to the low-level model backends. The
remaining aliases additionally accept :class:`~astropy.units.Quantity` values.
"""

from typing import Union

import numpy as np
from astropy import units as u

_ModelOutputRaw = dict[str, Union[float, np.ndarray]]
_ModelOutput = dict[str, Union[float, np.ndarray, u.Quantity]]
_ModelVariablesInputRaw = dict[str, Union[float, np.ndarray]]
_ModelVariablesInput = dict[str, Union[float, np.ndarray, u.Quantity]]
_ModelParametersInputRaw = dict[str, Union[float, np.ndarray]]
_ModelParametersInput = dict[str, Union[float, np.ndarray, u.Quantity]]
