"""
Compatibility shim for unpickling case-study graphs from pre-Phase-3f runs.

Old pickles stored the `solver_construction` class as a node attribute
(`forward_coupling_solver` / `backward_coupling_solver`).  After the casadi
backend was removed in Phase 3f those attributes are no longer read anywhere
— but `pickle.load` still needs to resolve the dotted path to complete
unpickling.

This module exposes a minimal stub `solver_construction` so those loads
succeed.  Don't use it for anything.  It will be deleted once a round of
"resave-all-pickles-with-solver-attrs-stripped" migration is done.
"""
from __future__ import annotations


class solver_construction:  # noqa: N801 — name preserved for pickle compat
    """No-op stub. Only exists so old pickles can be unpickled."""
    def __init__(self, *args, **kwargs):
        pass

    def __reduce__(self):
        return (solver_construction, ())
