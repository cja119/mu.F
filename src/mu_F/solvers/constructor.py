"""Compatibility stub so old case-study pickles still resolve SolverConstruction."""
from __future__ import annotations


class SolverConstruction:  # noqa: N801 — name preserved for pickle compat
    """No-op stub kept only for unpickling legacy graphs.

    Old pickles stored this class as a node attribute; the attribute is no
    longer read anywhere, but pickle.load still needs the dotted path to
    resolve. Not used for anything else.

    """

    # ---- External Methods ----

    def __init__(self, *args, **kwargs):
        """Accept and ignore any stored constructor arguments."""
        pass

    def __reduce__(self):
        """Reconstruct as an argument-free stub on unpickle."""
        return (SolverConstruction, ())
