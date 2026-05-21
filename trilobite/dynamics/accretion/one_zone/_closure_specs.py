"""
Specifiers for various disk models.

These specifiers are used in the headings of the disk models to specify
things like the ordering of the result fields, the expected runtime parameters,
the expect context parameters, etc.

Now models should have their specifiers added here or (if possible), they should
co-opt an existing specifier. This allows us to have a single source of truth for the
specifiers, and it allows us to easily change the specifiers for multiple models at once if needed.
"""

# =============================================== #
# RUNTIME PARAMETER SPECIFIERS                    #
# =============================================== #
# The runtime parameters are passed through the C-level to the
# closure functions and the integrator. They can be accessed there
# in their CGS / post-processed form to perform whatever relevant
# computations are necessary.

BASE_RUNTIME_PARAMETERS: dict = {
    "M_BH": {
        "description": "Black hole mass",
        "base_units": "g",
        "default": None,
        "log_transform": True,
    },
    "R_in": {
        "description": "Inner truncation radius",
        "base_units": "cm",
        "default": None,
        "log_transform": True,
    },
    "alpha": {
        "description": "Shakura-Sunyaev alpha",
        "base_units": None,
        "default": None,
        "log_transform": False,
    },
}
FALLBACK_RUNTIME_PARAMETERS: dict = {
    **BASE_RUNTIME_PARAMETERS,
    "M_fb_0": {
        "description": "Fallback mass supply rate at t_fb",
        "base_units": "g/s",
        "default": 1.0,  # dummy; user must override when fallback=True
        "log_transform": True,
    },
    "R_c": {
        "description": "Fallback circularization radius",
        "base_units": "cm",
        "default": 1.0,  # dummy; user must override when fallback=True
        "log_transform": True,
    },
    "t_fb": {
        "description": "Fallback reference time",
        "base_units": "s",
        "default": 1.0,  # dummy; user must override when fallback=True
        "log_transform": True,
    },
    "beta_fb": {
        "description": "Fallback power-law index",
        "base_units": None,
        "default": 5.0 / 3.0,
        "log_transform": False,
    },
}

# =============================================== #
# INITIAL CONDITIONS SPECIFIERS                   #
# =============================================== #
# These set the initial state of the disk. These should generally not
# need to be modified unless the entire integrator is being modified to
# allow for substantially different physics.
#
# IMPORTANT: Only M_D and J_D are read in the default integrator.
BASE_INITIAL_CONDITIONS: dict = {
    "M_D_0": {
        "description": "Initial disk mass",
        "base_units": "g",
        "default": None,
        "log_transform": True,
    },
    "J_D_0": {
        "description": "Initial disk angular momentum",
        "base_units": "g*cm**2/s",
        "default": None,
        "log_transform": True,
    },
}

# =============================================== #
# RESULT MAPPING                                  #
# =============================================== #
# During the writing phase of each timestep, the integrator and
# corresponding closure will write results to an array. The ordering
# of the results must be provided to the python level for interpretation.
BASE_RESULT_FIELDS: dict = {
    "R_D": {"description": "Disk outer radius", "units": "cm"},
    "Sigma": {"description": "Surface density", "units": "g/cm**2"},
    "Omega": {"description": "Angular velocity", "units": "rad/s"},
    "T_eff": {"description": "Effective temperature", "units": "K"},
    "T_c": {"description": "Central temperature", "units": "K"},
    "tau": {"description": "Optical depth", "units": None},
    "c_s": {"description": "Sound speed", "units": "cm/s"},
    "nu": {"description": "Kinematic viscosity", "units": "cm**2/s"},
    "t_visc": {"description": "Viscous timescale", "units": "s"},
    "Q_visc": {"description": "Viscous dissipation rate", "units": "erg cm-2 s-1"},
    "mdot": {"description": "Accretion rate", "units": "g/s"},
    "H": {"description": "Scale height", "units": "cm"},
    "H_over_R": {"description": "Aspect ratio", "units": None},
    "rho": {"description": "Midplane density", "units": "g/cm**3"},
}
FALLBACK_RESULT_FIELDS: dict = {
    **BASE_RESULT_FIELDS,
    "mdot_fb": {"description": "Fallback mass supply rate", "units": "g/s"},
}
ADVECTION_RESULT_FIELDS: dict = {
    **BASE_RESULT_FIELDS,
    "Q_adv": {"description": "Advective cooling rate", "units": "erg cm-2 s-1"},
}
ADV_FB_RESULT_FIELDS: dict = {
    **ADVECTION_RESULT_FIELDS,
    "mdot_fb": {"description": "Fallback mass supply rate", "units": "g/s"},
}

# Cython field mappings
BASE_CYTHON_FIELD_MAP: dict = {
    "R_D": 4,
    "Sigma": 5,
    "Omega": 6,
    "T_eff": 7,
    "T_c": 8,
    "tau": 9,
    "c_s": 10,
    "nu": 11,
    "t_visc": 16,
    "Q_visc": 12,
    "mdot": 13,
    "H": 17,
    "H_over_R": 18,
    "rho": 19,
}
FALLBACK_CYTHON_FIELD_MAP: dict = {
    **BASE_CYTHON_FIELD_MAP,
    "mdot_fb": None,
}
ADV_FB_CYTHON_FIELD_MAP: dict = {
    "R_D": 4,
    "Sigma": 5,
    "Omega": 6,
    "T_eff": 7,
    "T_c": 8,
    "tau": 9,
    "c_s": 10,
    "nu": 11,
    "t_visc": 16,
    "Q_visc": 12,
    "Q_adv": 13,
    "mdot": 14,
    "H": 18,
    "H_over_R": 19,
    "rho": 20,
    "mdot_fb": None,
}
