# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import matplotlib  # noqa: F401

sys.path.insert(0, os.path.abspath("../../triceratops"))
sys.path.insert(0, os.path.abspath("../../"))
# -- Project information -----------------------------------------------------

project = "Triceratops"
copyright = "2025, Eliza Diggins"
author = "Eliza Diggins"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.graphviz",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx.ext.doctest",
    "matplotlib.sphinxext.plot_directive",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["../_templates"]
bibtex_bibfiles = ["docs_bib.bib"]
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# Get the current git tag or branch name from the environment variable set by GitHub Actions.
ref = os.environ.get("GITHUB_REF_NAME", "")

if ref.startswith("v"):
    version = release = ref
else:
    version = release = "dev"

# Set the HTML theme to "pydata_sphinx_theme" and configure theme options for the project.
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "text": "Triceratops",
        "image_light": "images/logo.png",
        "image_dark": "images/logo.png",
        "alt_text": "Triceratops logo",
    },
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/eliza-diggins/Triceratops",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ],
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
    "switcher": {
        "json_url": "https://eliza-diggins.github.io/Triceratops/versions.json",
        "version_match": version,
    },
}

html_favicon = "images/logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "yt": ("https://yt-project.org/doc/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "unyt": ("https://unyt.readthedocs.io/en/latest/", None),
    "h5py": ("https://docs.h5py.org/en/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "emcee": ("https://emcee.readthedocs.io/en/stable/", None),
}

napoleon_use_param = True
napoleon_preprocess_types = True

# Suppress toc.not_included warnings for autosummary-generated attribute pages
# that are referenced by the class pages but not explicitly in a toctree.
suppress_warnings = ["toc.not_included"]

# Suppress nitpick warnings for informal type annotations used in NumPy-style
# docstrings that are not resolvable as Python objects.  These are intentional
# documentation conventions, not errors.
nitpick_ignore_regex = [
    # Informal type names used in NumPy-style docstrings
    (r"py:.*", r"array-like"),
    (r"py:.*", r"float-like"),
    (r"py:.*", r"callable"),
    (r"py:.*", r"quantity"),
    (r"py:.*", r"Enum-like"),
    (r"py:.*", r"enum-like"),
    # Private type aliases not exposed in the public API
    (r"py:class", r"_UnitBearingScalarLike"),
    (r"py:class", r"_UnitBearingArrayLike"),
    (r"py:class", r"_ArrayLike"),
    (r"py:class", r"_OneZoneMeta"),
    (r"py:class", r"_ModelParametersInput"),
    # Broken generic annotations produced by napoleon for Python 3.10+ syntax
    (r"py:class", r"dict\[.*"),
    (r"py:class", r"tuple\[.*"),
    (r"py:class", r"list\]"),
    (r"py:class", r"float\]"),
    (r"py:class", r"str$"),
    (r"py:class", r".*\]$"),
    # Private methods, functions, attributes, and modules — match both short names
    # (e.g. _log_opacity) and fully-qualified paths (e.g. triceratops.foo.Bar._method)
    (r"py:meth", r"_.*|.*\._.*"),
    (r"py:func", r"_.*|.*\._.*"),
    (r"py:attr", r"_.*|.*\._.*"),
    (r"py:obj", r"_.*|.*\._.*"),
    (r"py:mod", r".*\._.*|triceratops\..*\._.*"),
    # Short-form class names used consistently throughout the codebase
    (r"py:class", r"ModelParameter"),
    (r"py:class", r"ModelVariable"),
    (r"py:class", r"InferenceProblem"),
    (r"py:class", r"InferenceData"),
    (r"py:class", r"OneZoneAccretionResult"),
    (r"py:class", r"OneZoneAccretionDiskBase"),
    (r"py:class", r"SynchrotronSEDRegime"),
    (r"py:class", r"Quantity"),
    (r"py:class", r"u\.Quantity"),
    (r"py:class", r"np\.ndarray"),
    (r"py:class", r"ndarray"),
    (r"py:class", r"Path"),
    (r"py:class", r"cosmo\.Cosmology"),
    (r"py:class", r"tqdm"),
    (r"py:class", r"dtype=float64"),
    (r"py:class", r"ruamel\.yaml\.nodes\.Node"),
    (r"py:class", r"triceratops\.parallel\.base\.[RT]"),
    # Informal type annotations: not Python objects
    (r"py:class", r"namedtuple"),
    (r"py:class", r"OUTPUTS"),
    (r"py:class", r"C-contiguous"),
    (r"py:class", r"float64"),
    (r"py:class", r"shape \(.*"),
    (r"py:class", r"ax"),
    (r"py:class", r"fig"),
    (r"py:class", r"multiprocessing pool"),
    (r"py:class", r"multiprocessing\.base\.Pool"),
    (r"py:class", r"BoundType"),
    (r"py:class", r"CallablePrior"),
    (r"py:class", r"BaseTestOneZoneDisk"),
    (r"py:class", r"Data"),
    (r"py:class", r"SamplingResult"),
    (r"py:class", r"MCMCSamplingResult"),
    (r"py:class", r"InverseComptonCoolingEngine"),
    (r"py:class", r"SynchrotronRadiativeCoolingEngine"),
    (r"py:class", r"PowerLaw_SynchrotronSED"),
    (r"py:class", r"PL_Evolving_SSA_SED_Model"),
    (r"py:class", r"KramersBFOpacity"),
    (r"py:class", r"KramersFFOpacity"),
    (r"py:class", r"OneZoneClosure"),
    (r"py:class", r"YAML"),
    # Dict-view description strings injected by Napoleon from inherited dict methods
    (r"py:class", r"a set-like object.*"),
    (r"py:class", r"an object providing.*"),
    (r"py:class", r"\(k, v\).*"),
    (r"py:class", r"D\[k\].*"),
    (r"py:class", r"D\.get.*"),
    (r"py:class", r"None\..*"),
    (r"py:class", r"v, remove.*"),
    (r"py:class", r"actual_steps\)"),
    (r"py:class", r"Additional keyword.*"),
    (r"py:class", r"the plotting function\."),
    (r"py:class", r"\*\*kwargs"),
    # Private class refs from private modules
    (r"py:class", r"_igP\..*"),
    (r"py:class", r"_igP_adv\..*"),
    (r"py:class", r"\.closure\..*"),
    (r"py:meth", r"\.closure\..*"),
    # Relative class refs that remain unqualified in some docstrings
    (r"py:class", r"inference\..*"),
    (r"py:class", r"models\..*"),
    (r"py:class", r"radiation\..*"),
    (r"py:class", r"data\..*"),
    (r"py:class", r"dynamics\..*"),
    (r"py:class", r"sampling\..*"),
    (r"py:mod", r"tqdm"),
    (r"py:mod", r"data"),
    (r"py:mod", r"radiation\.synchrotron\.cooling"),
    (r"py:func", r"astropy\.units\.Unit"),
    (r"py:func", r"resolve_cosmological_distances"),
    (r"py:func", r"get_cosmology"),
    (r"py:func", r"run_one_zone_model"),
    (r"py:func", r"compute_ME_spectrum_from_dist_function"),
    (r"py:func", r"compute_ME_spectrum_from_dist_grid"),
    (r"py:func", r"compute_chevalier_ejecta_parameters"),
    (r"py:func", r"compute_first_kernel_interp"),
    (r"py:func", r"compute_second_kernel_interp"),
    (r"py:func", r"compute_gyrofrequency"),
    (r"py:func", r"compute_nu_critical"),
    (r"py:func", r"compute_single_electron_power"),
    (r"py:func", r"first_synchrotron_kernel"),
    (r"py:func", r"second_synchrotron_kernel"),
    (r"py:func", r"get_first_kernel_interpolator"),
    (r"py:func", r"get_second_kernel_interpolator"),
    (r"py:func", r"scipy\.interpolate\.RegularGridInterpolator"),
    (r"py:func", r"scipy\.interpolate\.interp1d"),
    (r"py:meth", r"__init_subclass__"),
    (r"py:meth", r"OneZoneAccretionResult\..*"),
    (r"py:meth", r"EmceeSampler\..*"),
    (r"py:meth", r"from_physics_to_params"),
    (r"py:meth", r"run"),
    (r"py:meth", r"solve"),
    (r"py:meth", r"to_cgs_array"),
    (r"py:meth", r"to_inference_data"),
    (r"py:meth", r"to_model_spec"),
    (r"py:meth", r"compute_shock_properties.*"),
    (r"py:attr", r"OUTPUTS"),
    (r"py:attr", r"SPECTRUM_INVERSION_FUNCTIONS"),
    (r"py:attr", r"OneZoneAccretionResult\..*"),
    (r"py:attr", r"params"),
    (r"py:attr", r"frequency"),
    (r"py:attr", r"model\..*"),
    (r"py:attr", r"ModelParameter\..*"),
    (r"py:obj", r"OneZoneAccretionResult"),
    (r"py:obj", r"gP_esDisk"),
    (r"py:obj", r"RadioLightCurveContainer"),
    (r"py:obj", r"XYDataContainer"),
    (r"py:obj", r"InferenceData\..*"),
    # Quantity with units appended (e.g. "Quantity [temperature]")
    (r"py:class", r"Quantity \[.*\]"),
    (r"py:class", r"Quantity\]"),
]

# Configure the sphinx galleries. These are contained in the
# /examples gallery.
sphinx_gallery_conf = {
    "examples_dirs": [
        "./galleries/data",
        "./galleries/inference",
        "./galleries/modeling",
        "./galleries/dynamics",
        "./galleries/synchrotron",
        "./galleries/accretion",
        "./galleries/photometry",
        "./galleries/opacity",
    ],
    "gallery_dirs": [
        "auto_examples/data",
        "auto_examples/inference",
        "auto_examples/modeling",
        "auto_examples/dynamics",
        "auto_examples/synchrotron",
        "auto_examples/accretion",
        "auto_examples/photometry",
        "auto_examples/opacity",
    ],
    # Do not abort the build if an individual gallery example fails.
    # Computationally intensive examples (MCMC, long integrations) may
    # fail in CI or constrained doc-build environments without invalidating
    # the rest of the documentation.
    "abort_on_example_error": False,
}
