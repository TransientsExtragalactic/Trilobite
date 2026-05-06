"""
Core shock-physics functions and classes.

These modules provide the backbone of Triceratops' shock-physics capabilities,
including the Rankine-Hugoniot jump conditions and related calculations.

For detailed documentation regarding the shock physics modules and their structure,
please read the corresponding section in the user guide: :ref:`shock_overview`.
"""
# Set the __all__.\

__all__ = [
    "rankine_hugoniot",
    "relativistic_jump_conditions",
    "JumpConditions",
    "StrongShockConditions",
    "StrongColdShockConditions",
    "WeakShockConditions",
    "WeakColdShockConditions",
    "RelativisticJumpConditions",
    "RelativisticShockConditions",
    "RelativisticColdShockConditions",
    "UltraRelativisticShockConditions",
    "UltraRelativisticColdShockConditions",
]

# Import the constituents.
from triceratops.dynamics.shocks.core import rankine_hugoniot, relativistic_jump_conditions
from triceratops.dynamics.shocks.core.rankine_hugoniot import (
    JumpConditions,
    StrongColdShockConditions,
    StrongShockConditions,
    WeakColdShockConditions,
    WeakShockConditions,
)
from triceratops.dynamics.shocks.core.relativistic_jump_conditions import (
    RelativisticColdShockConditions,
    RelativisticJumpConditions,
    RelativisticShockConditions,
    UltraRelativisticColdShockConditions,
    UltraRelativisticShockConditions,
)
