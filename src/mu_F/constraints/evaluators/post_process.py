"""
Post-processing global NLP evaluator.

Solves the upper-level reconstruction problem: find decision variables (graph
level, excluding `post_process_decision_indices`) that minimise the last
decision component subject to the upper-level classifier staying feasible
(`classifier(x) <= 0`).

The casadi path (previously `global_graph_upperlevel_NLP` in
`constraints/downstream.py`) encoded the objective as the integer sentinel
`obj_fn = -1`, which `build_objective_function` in the old solver utilities
interpreted as "minimise `x[-1]`".  We lift that into a plain JAX scalar
here — no sentinel indirection.

Classifier is read from the graph in preferred-live-then-rebuild order:
`graph.graph['post_process_upper_classifier']` if callable, otherwise
`graph.graph['post_process_upper_classifier_serialised']` reconstructed
via `mu_F.solvers.utilities.construct_model`.  Live case is normal during
a run; rebuild is the checkpoint-resume path.
"""
from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
import numpy as np

from mu_F.constraints.utils import standardise_model_decisions
from mu_F.solvers.septal import (
    construct_constrained_solver,
    solve_constrained,
)


__all__ = [
    "post_process_upper_level",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_upper_classifier(graph, cfg):
    """
    Resolve the upper-level classifier to a live JAX callable.

    Preferred: the graph already holds a callable under
    `post_process_upper_classifier` (normal runs).  Fallback: rebuild from
    the serialised dict — the checkpoint-resume path.
    """
    live = graph.graph.get('post_process_upper_classifier')
    if callable(live):
        return live

    serialised = graph.graph.get('post_process_upper_classifier_serialised')
    if serialised is None:
        raise RuntimeError(
            "No upper-level classifier available on graph.graph — neither "
            "a live `post_process_upper_classifier` callable nor a "
            "`post_process_upper_classifier_serialised` params dict."
        )
    # Rebuild lazily to avoid paying the surrogate-reconstruction import cost
    # on happy-path runs.
    from mu_F.solvers.utilities import construct_model
    return construct_model(
        serialised, cfg,
        supervised_learner='classification',
        model_type=cfg.surrogate.classifier_selection,
        model_surrogate='live_set_surrogate',
    )


def _upper_level_bounds(graph, cfg) -> list[jnp.ndarray]:
    """
    Compute the upper-level decision bounds.

    Total graph-level vars are `n_design + n_aux`; drop the indices listed
    in `post_process_decision_indices` (those are the lower-level decisions
    held fixed here).  Optionally standardise via the graph-level scaler.
    """
    total_ind = np.arange(graph.graph['n_design_args'] + graph.graph['n_aux_args'])
    fix_ind   = np.array(graph.graph['post_process_decision_indices']).reshape(-1)
    dec_ind   = np.delete(total_ind, fix_ind).astype(int)

    lb = jnp.hstack([
        jnp.array(bound[0]).reshape(-1)
        for bound in graph.graph['bounds'] if bound[0] != 'None'
    ])
    ub = jnp.hstack([
        jnp.array(bound[1]).reshape(-1)
        for bound in graph.graph['bounds'] if bound[1] != 'None'
    ])
    bounds = [lb, ub]
    if cfg.solvers.standardised:
        bounds = standardise_model_decisions(graph, bounds, None)

    return [bounds[0][dec_ind], bounds[1][dec_ind]]


# ---------------------------------------------------------------------------
# Public entry point — partial'd up by constraints/constructor.py
# ---------------------------------------------------------------------------

def post_process_upper_level(cfg, graph, node) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve the upper-level global reconstruction NLP via septal.

    Arguments match `constraint_evaluator(..., constraint_type='post_process_upper_level')`.
    `node` is accepted for signature parity with other evaluators but unused —
    this problem is graph-wide, not node-local.

    Returns
    -------
    (objective, decision_variables) : (scalar, (n_d,))
        Matches the return shape the casadi path yielded, so downstream code
        (sampling schemes, visualisers) consumes the tuple unchanged.
    """
    classifier = _load_upper_classifier(graph, cfg)
    bounds     = _upper_level_bounds(graph, cfg)
    lb         = jnp.asarray(bounds[0]).reshape(-1)
    ub         = jnp.asarray(bounds[1]).reshape(-1)
    n_d        = int(lb.size)

    def objective(x):
        # Epigraph reformulation: minimise the last decision variable.
        return x.reshape(-1)[-1]

    def constraint(x):
        return classifier(x.reshape(1, -1)).reshape(-1)

    lhs = jnp.full((1,), -jnp.inf)
    rhs = jnp.zeros((1,))

    solver  = construct_constrained_solver(
        objective, constraint, [lb, ub],
        tol=cfg.solvers.tol,
        constraint_lhs=lhs,
        constraint_rhs=rhs,
    )
    # Seed with the midpoint — deterministic, fine for the global NLP.
    x0 = 0.5 * (lb + ub)
    guesses = x0.reshape(1, -1)

    result = solve_constrained([solver], [guesses])

    objective_val = jnp.asarray(result['objective']).reshape(())
    x_star        = jnp.asarray(result['params']).reshape(-1)
    return objective_val, x_star
