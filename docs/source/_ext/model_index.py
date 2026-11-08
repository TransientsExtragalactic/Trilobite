"""Model directory support: a searchable index, generated tables, and development-status badges.

The :class:`~trilobite.models.core.base.Model` subclasses are the single source of truth. Nothing about an
individual model is listed by hand in the docs; a new model appears in the directory, the search index and the
counts as soon as it exists. This extension provides

* ``_static/model_index.json`` (written at the end of an HTML build), consumed by ``_static/model_search.js`` on the
  model directory page;
* the ``model-search`` directive, which marks where the search box mounts;
* the ``model-table`` directive, which renders a table of models (link, status badge, summary) for the given
  module path(s);
* the ``model-status-legend`` directive, which renders the status vocabulary from
  :class:`~trilobite.models.core.status.ModelStatus`;
* the ``model-count`` role, which renders the number of models in a category;
* an ``autodoc-process-docstring`` hook that puts the status badge at the top of every model's API page.

Only the *grouping* metadata below (display group per module) is curated. A module with no entry falls back to its
top-level subpackage name so the model is still listed and searchable.
"""

import inspect
import json
import re
from functools import lru_cache
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList

# Display group (the section of the directory) for each module that defines models. Keys are module paths.
GROUPS = {
    "trilobite.models.generic.curves": "Curve fitting",
    "trilobite.models.generic.light_curve": "Light curves",
    "trilobite.models.generic.evolving_seds": "Evolving SEDs",
    "trilobite.models.SEDs.synchrotron": "Synchrotron SEDs",
    "trilobite.models.SEDs.blackbody_optical": "Blackbody photometry",
    "trilobite.models.supernovae": "Self-similar shocks",
    "trilobite.models.GRBs": "Prompt emission",
    "trilobite.models.LFBOTs": "Two-component synchrotron",
    "trilobite.models.TDEs": "TDE light curves",
}

# Top-level directory category for each subpackage of ``trilobite.models``.
CATEGORIES = {
    "generic": "Generic",
    "SEDs": "SEDs",
    "supernovae": "Supernovae",
    "LFBOTs": "LFBOTs",
    "GRBs": "GRBs",
    "TDEs": "TDEs",
}

_ROLE = re.compile(r":[a-zA-Z:_-]+:`~?([^`]*)`")
_MARKUP = re.compile(r"[`*]|\\")


def _plain(text):
    """Strip RST roles and inline markup from a one-line summary (for the search index)."""
    text = _ROLE.sub(lambda m: m.group(1).split(".")[-1] if "." in m.group(1) else m.group(1), text)
    return _MARKUP.sub("", text).strip()


def _summary(cls):
    """One-line summary: ``DESCRIPTION`` if set, else the first paragraph of the class docstring (RST kept)."""
    if cls.DESCRIPTION:
        return " ".join(cls.DESCRIPTION.split())
    doc = inspect.getdoc(cls) or ""
    return " ".join(doc.split("\n\n")[0].split())


def _category(cls):
    from trilobite.models.core.optical import OpticalEpochModel, OpticalModel

    if issubclass(cls, (OpticalModel, OpticalEpochModel)):
        return "Optical"
    parts = cls.__module__.split(".")
    return CATEGORIES.get(parts[2] if len(parts) > 2 else "", "Other")


def _group(cls):
    module = cls.__module__
    while module:
        if module in GROUPS:
            return GROUPS[module]
        module = module.rpartition(".")[0]
    return _category(cls)


@lru_cache(maxsize=1)
def collect_models():
    """Return a record for every concrete, in-library :class:`~trilobite.models.core.base.Model` subclass."""
    import trilobite.models  # noqa: F401  (imports every model module so subclasses are registered)
    from trilobite.models.core.base import Model

    def walk(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from walk(sub)

    records = {}
    for cls in walk(Model):
        if inspect.isabstract(cls) or not cls.__module__.startswith("trilobite.models"):
            continue
        # Documented under the package that re-exports it, the canonical path is where it is *defined*.
        full = f"{cls.__module__}.{cls.__name__}"
        records[full] = {
            "name": cls.__name__,
            "full": full,
            "module": cls.__module__,
            "category": _category(cls),
            "group": _group(cls),
            "status": cls.STATUS.value,
            "status_label": cls.STATUS.label,
            "status_note": cls.STATUS_NOTE,
            "summary_rst": _summary(cls),
            "description": _plain(_summary(cls)),
            "parameters": [p.name for p in cls.PARAMETERS],
            "variables": [v.name for v in cls.VARIABLES],
            "outputs": list(cls.OUTPUTS._fields),
            "reference": cls.REFERENCE,
            "url": f"_as_gen/{full}.html",
        }
    return sorted(records.values(), key=lambda r: (r["category"], r["group"], r["name"].lower()))


# --------------------------------------------------------------------------------------------------------------------
# Directives and roles
# --------------------------------------------------------------------------------------------------------------------
def _badge(status):
    return f":bdg-{status.badge_color}:`{status.label}`"


class ModelSearch(Directive):
    """Mount point for the model search box (see ``_static/model_search.js``)."""

    def run(self):
        return [nodes.raw("", '<div id="model-search"></div>', format="html")]


class ModelTable(Directive):
    """Table of the models defined in the given module path(s): link, status badge and one-line summary."""

    required_arguments = 1
    optional_arguments = 20
    has_content = False

    def run(self):
        prefixes = self.arguments
        rows = [
            r for r in collect_models() if any(r["module"] == p or r["module"].startswith(p + ".") for p in prefixes)
        ]
        if not rows:
            raise self.error(f"model-table: no models found under {', '.join(prefixes)}")

        lines = [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 30 12 58",
            "   :class: model-table",
            "",
            "   * - Model",
            "     - Status",
            "     - Description",
        ]
        from trilobite.models.core.status import ModelStatus

        for r in rows:
            note = f" *{r['status_note']}*" if r["status_note"] else ""
            lines += [
                f"   * - :py:class:`~{r['full']}`",
                f"     - {_badge(ModelStatus(r['status']))}",
                f"     - {r['summary_rst']}{note}",
            ]
        node = nodes.container()
        self.state.nested_parse(StringList(lines, source="model-table"), self.content_offset, node)
        return node.children


class ModelStatusLegend(Directive):
    """Definition list of every development status."""

    def run(self):
        from trilobite.models.core.status import ModelStatus

        lines = []
        for status in ModelStatus:
            lines += [f"{_badge(status)}", f"    {status.description}", ""]
        node = nodes.container(classes=["model-status-legend"])
        self.state.nested_parse(StringList(lines, source="model-status-legend"), self.content_offset, node)
        return node.children


def model_count_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """``:model-count:`Generic``` -> the number of models in that category (case-insensitive; ``all`` for every one)."""
    key = text.strip().lower()
    models = collect_models()
    n = len(models) if key == "all" else sum(1 for r in models if r["category"].lower() == key)
    if n == 0 and key != "all":
        msg = inliner.reporter.error(f"model-count: no models in category {text!r}", line=lineno)
        return [inliner.problematic(rawtext, rawtext, msg)], [msg]
    return [nodes.Text(f"{n} model" + ("" if n == 1 else "s"))], []


# --------------------------------------------------------------------------------------------------------------------
# API pages: status banner
# --------------------------------------------------------------------------------------------------------------------
def _add_status_banner(app, what, name, obj, options, lines):
    if what != "class" or not inspect.isclass(obj):
        return
    from trilobite.models.core.base import Model

    if obj is Model or not issubclass(obj, Model) or inspect.isabstract(obj):
        return
    note = f" {obj.STATUS_NOTE}" if obj.STATUS_NOTE else ""
    banner = f"{_badge(obj.STATUS)} {obj.STATUS.description}{note}"
    lines[0:0] = [banner, ""]


# --------------------------------------------------------------------------------------------------------------------
# Search index
# --------------------------------------------------------------------------------------------------------------------
def _write_index(app, exception):
    if exception is not None or app.builder.format != "html":
        return
    keep = ("name", "full", "category", "group", "status", "status_label", "status_note", "description")
    records = [
        {
            **{k: r[k] for k in keep},
            "params": r["parameters"],
            "outputs": r["outputs"],
            "ref": r["reference"],
            "url": r["url"],
        }
        for r in collect_models()
    ]
    out = Path(app.outdir) / "_static" / "model_index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records))


def setup(app):
    """Register the directives, role and build hooks."""
    app.add_directive("model-search", ModelSearch)
    app.add_directive("model-table", ModelTable)
    app.add_directive("model-status-legend", ModelStatusLegend)
    app.add_role("model-count", model_count_role)
    app.connect("autodoc-process-docstring", _add_status_banner)
    app.connect("build-finished", _write_index)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
