from typing import Union

import numpy as np
from astropy import units as u

_ModelOutputRaw = dict[str, Union[float, np.ndarray]]
_ModelOutput = dict[str, Union[float, np.ndarray, u.Quantity]]
_ModelVariablesInputRaw = dict[str, Union[float, np.ndarray]]
_ModelVariablesInput = dict[str, Union[float, np.ndarray, u.Quantity]]
_ModelParametersInputRaw = dict[str, Union[float, np.ndarray]]
_ModelParametersInput = dict[str, Union[float, np.ndarray, u.Quantity]]
