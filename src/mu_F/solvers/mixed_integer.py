"""
mixed_integer.py — integer-spec resolver for the design domain.

The flat enumeration over integer design dims (`mixed_integer_solve`) and
its `slack_from_cfg` companion lived here until M9.  Both were replaced
by the unified `solvers.integer_nlp.solve_integer_nlp` / `IntegerProblem`
abstraction — every evaluator now routes integer dims through that path,
so the per-leaf loop has nothing left to call it.

What remains is `resolve_integer_spec` — a tiny pure-Python helper used
by every evaluator's `_build_for_key` to translate the cfg's
`design_domain` list into `(int_dims, int_values)`.  Kept here (not moved
into `integer_nlp`) so the import graph stays untangled: `integer_nlp`
already pulls in septal; `resolve_integer_spec` has no JAX deps.
"""
from __future__ import annotations


__all__ = ["resolve_integer_spec"]


def resolve_integer_spec(domain):
    """Split a `design_domain` list into integer-dim indices and value sets.

    Each `domain` entry is either the string `'real'` (continuous slot;
    skipped) or a sequence of allowed integer values for that slot.
    Returns `(int_dims, int_values)` — `int_dims[i]` is the position of
    the i-th integer slot in the design vector; `int_values[i]` lists its
    allowed values.
    """
    if domain is None:
        return [], []
    int_dims, int_values = [], []
    for i, d in enumerate(domain):
        if d == 'real':
            continue
        int_dims.append(i)
        int_values.append(list(d))
    return int_dims, int_values
