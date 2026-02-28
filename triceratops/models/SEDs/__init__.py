"""
Spectral Energy Distribution (SED) models.

This subpackage provides analytic and semi-analytic models for
spectral energy distributions (SEDs).

The models included here fall into two broad categories:

1. **Phenomenological SED models**
   These models describe observed SED shapes using flexible
   parameterizations (e.g., broken power laws), without enforcing
   physical consistency conditions. They are intended for:

   - Empirical SED fitting
   - Multi-epoch spectral evolution studies
   - Exploratory transient modeling
   - Situations where physical interpretation is secondary

2. **Physics-based synchrotron models**
   These models enforce microphysical closure relations and
   self-consistent spectral break calculations (e.g., cooling,
   synchrotron self-absorption). They are appropriate when
   physical interpretation of parameters is required.

The key distinction is that phenomenological models do **not**
enforce:

- Energy conservation
- Shock dynamics
- Equipartition relations
- Self-consistent spectral break ordering

They are designed purely for flexible data modeling.

For physically coupled shock and radiation models,
see the shock and synchrotron modules.
"""

__all__ = ["synchrotron"]

from . import synchrotron
from .synchrotron import *

__all__.extend(synchrotron.__all__)
