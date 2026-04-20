"""
Current-node surrogate evaluators.

Two flavours, both operating on the current node's own live surrogates
(not a successor's / predecessor's):

  - `current_constraint_surrogate`  — box-only feasibility check: min
    classifier(assemble(x, inputs)) over (design, aux).  No general
    constraints.  Used for "rollout constraint" checks.

  - `current_cost_surrogate`  — constrained CTG: minimise the node's CTG
    surrogate subject to the node's classifier being feasible (g(x) <= 0).
    Single-problem (no successor fan-out) so batching happens entirely via
    vmap inside septal — no pmap wrapper.

Both use the reduced decision space (inputs + aux indices masked away from
the full NLP) since those positions are pinned by the supplied `inputs`.
"""
from __future__ import annotations

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit

from mu_F.constraints.utils import (
    standardise_inputs,
    standardise_model_decisions,
    mask_classifier,
    initial_guess,
)
from mu_F.solvers.septal import (
    construct_constrained_solver,
    solve_constrained,
)
# Box-only path — `construct_solver` and `solve` are thin aliases for the
# constrained path with `constraint_func=None`.  They live in `backward.py`
# because that's where every box-only callsite in the repo reaches for them.
from mu_F.constraints.evaluators.backward import construct_solver, solve


__all__ = [
    "prepare_current_constraint_problem",
    "prepare_current_cost_problem",
    "current_constraint_surrogate",
    "current_cost_surrogate",
]


# ---------------------------------------------------------------------------
# Problem builders
# ---------------------------------------------------------------------------

def prepare_current_constraint_problem(inputs, graph, node, cfg):
    """
    Current-node feasibility: find `(design, aux)` satisfying the node's live
    classifier given fixed `inputs`.

    Decision space excludes the input indices (those are pinned by `inputs`
    via `mask_classifier`).  Returns `(objective_func, bounds)`.
    """
    if node is None:
        return None, None

    n_design = graph.nodes[node]['n_design_args']
    n_input  = graph.nodes[node]['n_input_args']
    n_aux    = graph.graph['n_aux_args']

    ndim = n_design + n_input + n_aux
    input_indices = np.arange(n_design, n_design + n_input).astype(int)
    aux_indices   = np.arange(ndim - n_aux, ndim).astype(int)
    fix_indices   = np.hstack([input_indices, aux_indices]).astype(int)

    if cfg.solvers.standardised:
        inputs = inputs.at[:].set(standardise_inputs(graph, inputs, node, fix_indices))
        decision_bounds = standardise_model_decisions(
            graph, graph.nodes[node]["extendedDS_bounds"], node,
        )
    else:
        decision_bounds = graph.nodes[node]["extendedDS_bounds"]

    decision_bounds = [jnp.delete(bound, fix_indices, axis=1) for bound in decision_bounds]

    classifier         = graph.nodes[node]["classifier"]
    wrapped_classifier = mask_classifier(classifier, ndim, input_indices, aux_indices)

    objective_func = jit(partial(
        lambda x, y: wrapped_classifier(x, y).squeeze(),
        y=inputs.reshape(1, -1),
    ))
    return objective_func, decision_bounds


def prepare_current_cost_problem(inputs, graph, node, cfg):
    """
    Current-node CTG: minimise the node's CTG surrogate subject to the
    node's classifier staying feasible (`g(x) <= 0`).

    Same reduced decision space as `prepare_current_constraint_problem`;
    objective is the CTG regressor, classifier moves to the constraint.
    Returns `(objective_func, constraint_func, bounds)`.
    """
    if node is None:
        return None, None, None

    n_design = graph.nodes[node]['n_design_args']
    n_input  = graph.nodes[node]['n_input_args']
    n_aux    = graph.graph['n_aux_args']

    ndim = n_design + n_input + n_aux
    input_indices = np.arange(n_design, n_design + n_input).astype(int)
    aux_indices   = np.arange(ndim - n_aux, ndim).astype(int)
    fix_indices   = np.hstack([input_indices, aux_indices]).astype(int)

    if inputs is None:
        inputs = jnp.empty((1, 0))

    if cfg.solvers.standardised and inputs.size > 0:
        inputs = inputs.at[:].set(standardise_inputs(graph, inputs, node, fix_indices))
        decision_bounds = standardise_model_decisions(
            graph, graph.nodes[node]["extendedDS_bounds"], node,
        )
    else:
        decision_bounds = graph.nodes[node]["extendedDS_bounds"]

    decision_bounds = [jnp.delete(bound, fix_indices, axis=1) for bound in decision_bounds]

    ctg_surrogate      = graph.nodes[node]["ctg_surrogate"]
    classifier         = graph.nodes[node]["classifier"]
    wrapped_ctg        = mask_classifier(ctg_surrogate, ndim, input_indices, aux_indices)
    wrapped_classifier = mask_classifier(classifier,    ndim, input_indices, aux_indices)

    # ctg_surrogate may return a vector of per-uncertainty-sample predictions;
    # collapse to a scalar via sum to match the casadi path's
    # `_ensure_scalar_objective` behaviour.
    objective_func = jit(partial(
        lambda x, y: wrapped_ctg(x, y).reshape(-1).sum(),
        y=inputs.reshape(1, -1),
    ))
    constraint_func = jit(partial(
        lambda x, y: wrapped_classifier(x, y).reshape(-1),
        y=inputs.reshape(1, -1),
    ))

    return objective_func, constraint_func, decision_bounds


# ---------------------------------------------------------------------------
# Top-level evaluators
# ---------------------------------------------------------------------------

def current_constraint_surrogate(inputs, aux, cfg, graph, node):
    """
    Box-only feasibility check via `construct_solver` (septal `g=None`).

    Returns a 3-tuple `(decision_variables, objective, converged)` to match
    the contract the rollout agent consumes:
        decision_variables : shape (n_d, n_samples) — transposed so callers
                             can index as `v.T[i]` for per-sample actions
        objective          : shape (n_samples, 1)
        converged          : shape (n_samples, 1), dtype bool
    """
    objective_func, bounds = prepare_current_constraint_problem(inputs, graph, node, cfg)
    if objective_func is None or bounds is None:
        n = inputs.shape[0]
        return (
            jnp.zeros((0, n)),
            jnp.zeros((n, 1)),
            jnp.zeros((n, 1), dtype=bool),
        )

    solver  = construct_solver(
        objective_func, bounds,
        tol=cfg.solvers.tol,
    )
    guesses = initial_guess(cfg.solvers, bounds)
    result  = solve([solver], [guesses])
    # `params` is (n_samples, n_d) from solve_constrained; transpose to
    # (n_d, n_samples) so agent's `v.T[i]` slicing works.
    return (
        jnp.asarray(result['params']).T,
        result['objective'].reshape(-1, 1),
        result['converged'].reshape(-1, 1),
    )


def current_cost_surrogate(inputs, aux, cfg, graph, node):
    """
    Constrained CTG on the current node.  Single sub-problem; vmap inside
    septal handles batching — no pmap needed.

    Returns `(decision_variables, objective, converged)` in the same shape
    as `current_constraint_surrogate`.
    """
    objective_func, constraint_func, bounds = prepare_current_cost_problem(
        inputs, graph, node, cfg,
    )
    if objective_func is None or bounds is None:
        n = inputs.shape[0]
        return (
            jnp.zeros((0, n)),
            jnp.zeros((n, 1)),
            jnp.zeros((n, 1), dtype=bool),
        )

    solver  = construct_constrained_solver(
        objective_func, constraint_func, bounds,
        tol=cfg.solvers.tol,
    )
    guesses = initial_guess(cfg.solvers, bounds)
    result  = solve_constrained([solver], [guesses])
    return (
        jnp.asarray(result['params']).T,
        result['objective'].reshape(-1, 1),
        result['converged'].reshape(-1, 1),
    )
