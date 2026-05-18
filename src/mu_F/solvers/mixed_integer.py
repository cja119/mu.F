"""
mixed_integer.py — flat enumeration over integer design dims with septal at
each leaf and JAX-side selection of the best feasible result.

The leaf solver is injected by the caller as a callable taking
`(lb, ub, x0) -> SQPResult`, typically closing over a septal factory.  This
keeps `mixed_integer.py` decoupled from septal's specifics and from any
particular evaluator's bookkeeping.

Two modes:
  - `best`: pick the lowest-objective feasible leaf; if no feasible leaf
    exists, pick the lowest-objective infeasible leaf.
  - `first_feasible`: pick the first leaf (in enumeration order) with
    `success=True`; fall back to the last leaf if none feasible.

Selection is implemented with JAX ops so the call sits cleanly inside
JIT-traced shards.  Integer-domain enumeration is unrolled at trace time
(the integer specs are Python lists known statically), so the per-leaf
`leaf_solver` calls are emitted as part of the traced program.

No integer dims (empty `int_dims`) short-circuits to a single `leaf_solver`
call, so real-only case studies see no overhead.
"""
from __future__ import annotations

from itertools import product
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import jax.numpy as jnp
import numpy as np


__all__ = ["mixed_integer_solve", "resolve_integer_spec", "slack_from_cfg"]


def resolve_integer_spec(domain):
    """Split a `design_domain` list into integer-dim indices and value sets."""
    if domain is None:
        return [], []
    int_dims, int_values = [], []
    for i, d in enumerate(domain):
        if d == 'real':
            continue
        int_dims.append(i)
        int_values.append(list(d))
    return int_dims, int_values


def slack_from_cfg(cfg):
    """Pull `cfg.solvers.mixed_integer.slack` with a 0.0 fallback.  Retained
    for caller API stability; unused in the current flat-enumeration
    implementation (no pruning).
    """
    try:
        return float(cfg.solvers.mixed_integer.slack)
    except (AttributeError, KeyError):
        return 0.0


def mixed_integer_solve(
    leaf_solver: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Any],
    bounds,
    x0,
    *,
    int_dims: Sequence[int],
    int_values: Sequence[Sequence[float]],
    mode: str = "best",
    slack: float = 0.0,
):
    """Enumerate every integer combination, call `leaf_solver` at each, and
    return the best (feasibility-priority).  `slack` is accepted but unused.
    """
    lb_root = jnp.asarray(bounds[0])
    ub_root = jnp.asarray(bounds[1])
    x0_root = jnp.asarray(x0)

    if not int_dims:
        return leaf_solver(lb_root, ub_root, x0_root)

    objs, succs, xs = [], [], []
    for combo in product(*int_values):
        lb_leaf, ub_leaf, x0_leaf = lb_root, ub_root, x0_root
        for dim_idx, v in zip(int_dims, combo):
            lb_leaf = lb_leaf.at[dim_idx].set(float(v))
            ub_leaf = ub_leaf.at[dim_idx].set(float(v))
            x0_leaf = x0_leaf.at[dim_idx].set(float(v))
        res = leaf_solver(lb_leaf, ub_leaf, x0_leaf)
        objs.append(jnp.asarray(res.objective))
        succs.append(jnp.asarray(res.success).astype(bool))
        xs.append(res.decision_variables)

    objs_arr  = jnp.stack(objs)
    succs_arr = jnp.stack(succs)
    any_feas  = jnp.any(succs_arr)

    if mode == "first_feasible":
        first_feas = jnp.argmax(succs_arr.astype(jnp.int32))
        chosen_idx = jnp.where(any_feas, first_feas, jnp.int32(len(objs) - 1))
    else:
        ranked     = jnp.where(succs_arr, objs_arr, jnp.full_like(objs_arr, jnp.inf))
        feas_idx   = jnp.argmin(ranked)
        all_idx    = jnp.argmin(objs_arr)
        chosen_idx = jnp.where(any_feas, feas_idx, all_idx)

    chosen_obj  = objs_arr[chosen_idx]
    chosen_succ = succs_arr[chosen_idx]
    if xs[0] is None:
        chosen_x = None
    else:
        xs_arr = jnp.stack([jnp.asarray(x) for x in xs])
        chosen_x = xs_arr[chosen_idx]

    return SimpleNamespace(
        objective=chosen_obj,
        success=chosen_succ,
        decision_variables=chosen_x,
    )
