"""Build a JSON index of the sphinx-gallery examples for the gallery landing-page search.

At the end of an HTML build this scans every ``plot_*.py`` under the ``galleries/`` directory, reads the title from
the module docstring, and writes ``_static/gallery_index.json`` into the build output. ``_static/gallery_search.js``
loads that file on the landing page (``examples.rst``).

Examples are never listed by hand: a new example appears in the search as soon as it exists. Only the *grouping*
metadata below (topic per gallery directory, science-case tags per example) is curated. A gallery directory with no
entry falls back to the topic ``"Other"`` so it is still searchable.
"""

import ast
import json
import re
from pathlib import Path

# Landing-page topic for each top-level gallery directory.
TOPICS = {
    "data": "Data & Photometry",
    "photometry": "Data & Photometry",
    "synchrotron": "Radiation Physics",
    "free_free": "Radiation Physics",
    "opacity": "Radiation Physics",
    "dynamics": "Shocks & Dynamics",
    "accretion": "Accretion Disks",
    "modeling": "Forward Modeling",
    "inference": "Inference",
}

# Human-readable sub-gallery label, keyed by directory path relative to ``galleries/``.
SUBGALLERIES = {
    "data": "Observational Data",
    "photometry": "Filters & Photometry",
    "opacity": "Opacity",
    "accretion": "Accretion Disks",
    "modeling": "Forward Modeling",
    "inference": "Inference",
    "dynamics/a_fundamentals": "Shock basics",
    "dynamics/b_models": "Self-similar shocks",
    "dynamics/c_engines": "Numerical engines",
    "synchrotron/a_fundamentals": "Synchrotron fundamentals",
    "synchrotron/b_seds": "Synchrotron SEDs",
    "synchrotron/c_cooling": "Electron cooling",
    "synchrotron/d_closures": "Closures & equipartition",
    "free_free/a_gaunt_factors": "Free-free",
    "free_free/b_emission": "Free-free",
    "free_free/c_ffa_sed": "Free-free",
}

# Science-case tags, keyed by example path relative to ``galleries/`` (no ``.py``). An example may have several.
CASES = {
    "SNe & CSM": [
        "modeling/plot_chevalier_shock_synchrotron_cooling",
        "modeling/plot_type_IIn_SN_dense_CSM",
        "modeling/plot_ffa_fadeout_evolution",
        "dynamics/b_models/plot_chevalier_two_shock",
        "dynamics/c_engines/plot_numerical_shocks_csms",
        "inference/plot_shock_parameter_inference",
    ],
    "TDEs": [
        "modeling/plot_tde_radio_afterglow",
        "accretion/plot_tde_disk_observables",
        "accretion/plot_fallback_disk",
        "accretion/plot_disk_broadband_sed",
    ],
    "GRBs & relativistic": [
        "dynamics/b_models/plot_blandford_mcKee",
        "dynamics/c_engines/plot_relativistic_thin_shell_convergence",
        "synchrotron/b_seds/plot_off_axis_numerical",
        "synchrotron/b_seds/plot_on_axis_numerical",
    ],
    "Radio SED analysis": [
        "synchrotron/d_closures/plot_equipartition_analysis",
        "synchrotron/d_closures/plot_sed_to_source_size",
        "inference/plot_SED_fit",
        "inference/plot_evolving_sed",
        "inference/plot_censored_SED_fit",
        "inference/plot_multi_epoch_sed_evolution",
    ],
}

_RULE = re.compile(r"[=\-*~^#]{3,}")


def _title(path):
    """First non-empty, non-underline line of the module docstring (math roles stripped)."""
    doc = ast.get_docstring(ast.parse(path.read_text())) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line and not _RULE.fullmatch(line):
            return re.sub(r":math:`([^`]*)`", r"\1", line)
    return path.stem


def build_index(galleries_dir):
    """Return a list of example records for every ``plot_*.py`` under ``galleries_dir``."""
    case_of = {}
    for case, paths in CASES.items():
        for p in paths:
            case_of.setdefault(p, []).append(case)

    records = []
    for path in sorted(galleries_dir.rglob("plot_*.py")):
        rel = path.relative_to(galleries_dir)
        directory = rel.parent.as_posix()
        top = rel.parts[0]
        key = rel.with_suffix("").as_posix()
        records.append(
            {
                "title": _title(path),
                "topic": TOPICS.get(top, "Other"),
                "sub": SUBGALLERIES.get(directory, SUBGALLERIES.get(top, top)),
                "cases": case_of.get(key, []),
                # Output page, relative to the docs root (see gallery_dirs in conf.py).
                "url": f"auto_examples/{key}.html",
            }
        )
    return records


def _write_index(app, exception):
    if exception is not None or app.builder.format != "html":
        return
    records = build_index(Path(app.srcdir) / "galleries")
    out = Path(app.outdir) / "_static" / "gallery_index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records))


def setup(app):
    app.connect("build-finished", _write_index)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
