# _sources.pxd — source-term convention.
#
# Re-exports the source_func typedef from closure.pxd so that files that
# only need the source interface can cimport a single module.

from .closure cimport source_func  # noqa: F401 — re-export
