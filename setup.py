"""
Build script for Cython extensions.

All other project metadata lives in ``pyproject.toml``.  This file exists
solely to declare the ``ext_modules`` that setuptools cannot express in TOML.
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import setup

_PYX_SOURCES = [
    # architecture surface
    "triceratops/dynamics/accretion/one_zone/closure.pyx",
    "triceratops/dynamics/accretion/one_zone/integrator.pyx",
    "triceratops/dynamics/accretion/one_zone/_writer.pyx",
    "triceratops/dynamics/accretion/one_zone/_sources.pyx",
    # physics implementations
    "triceratops/dynamics/accretion/one_zone/physics/_eos.pyx",
    "triceratops/dynamics/accretion/one_zone/physics/_viscous.pyx",
    "triceratops/dynamics/accretion/one_zone/physics/_fallback.pyx",
    # assembled disk closures
    "triceratops/dynamics/accretion/one_zone/models/_gP_es.pyx",
    "triceratops/dynamics/accretion/one_zone/models/_igP_es.pyx",
    "triceratops/dynamics/accretion/one_zone/models/_gP_es_fb.pyx",
    "triceratops/dynamics/accretion/one_zone/models/_igP_es_fb.pyx",
    "triceratops/dynamics/accretion/one_zone/models/_igP_es_adv.pyx",
    "triceratops/dynamics/accretion/one_zone/models/_igP_es_adv_fb.pyx",
    # math utilities
    "triceratops/math_utils/_bracket_root_finder.pyx",
]

setup(
    ext_modules=cythonize(
        _PYX_SOURCES,
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
        },
    ),
    include_dirs=[np.get_include()],
)
