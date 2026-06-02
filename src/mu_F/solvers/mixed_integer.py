"""Integer-spec resolver translating a cfg design_domain into dim/value sets."""
from __future__ import annotations


__all__ = ["resolve_integer_spec"]


def resolve_integer_spec(domain):
    """
    Split a design_domain list into integer-dim indices and value sets.
    Called by every evaluator's _build_for_key; lives here (not in
    integer_nlp) to keep the import graph free of JAX deps.
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
