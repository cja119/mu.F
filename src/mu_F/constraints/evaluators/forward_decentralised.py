"""
Decentralised forward coupling constraint evaluator.

Different from the per-predecessor `forward.py` sub-problems — here every
predecessor of `node` is jammed into a single NLP with a concatenated decision
vector.  Each sample becomes one combined solve.

    x = concat([x_pred_1, x_pred_2, ...])             (concatenated NLP across preds)

    min   -classifier_node(hstack([v, fwd(x)])) - node_backoff
              where fwd(x) stacks each pred's forward-surrogate output
                    v is this node's measured inputs for the sample
    s.t.  classifier_pred_i(x[slice_i]) + backoff_i  <=  0   for every predecessor
          lb_concat <= x <= ub_concat

The returned evaluation is the node classifier's optimal value + backoff
(positive = feasible), matching the casadi path's `-objective` return.

Un-ported variant left over from Phase 3f; this module closes that gap.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from mu_F.constraints.utils import standardise_model_decisions
from mu_F.solvers.septal import (
    construct_constrained_solver,
    solve_constrained,
)


__all__ = [
    "prepare_forward_decentralised_problem",
    "evaluate_forward_decentralised",
    "forward_constraint_decentralised_evaluator",
]


# ---------------------------------------------------------------------------
# Problem builder (single sample)
# ---------------------------------------------------------------------------

def prepare_forward_decentralised_problem(inputs, aux, graph, node, cfg):
    """
    Build the decentralised forward sub-problem for ONE sample.

    Parameters
    ----------
    inputs : (1, n_inputs_of_node) — measured inputs for this sample
    aux    : (1, n_aux)            — auxiliary vars (unused here but carried
                                     through for signature parity)
    graph, node, cfg : as usual.

    Returns
    -------
    objective_func : Callable      f(x) -> scalar
    constraint_func : Callable     g(x) -> (n_g,)
    bounds : [lb, ub]              each shape (n_d_concat,)
    """
    preds = list(graph.predecessors(node))
    if not preds:
        return None, None, None

    # Per-predecessor decision dimensions and slice offsets.
    pred_dims = [
        graph.nodes[p]['n_design_args']
        + graph.nodes[p]['n_input_args']
        + graph.graph['n_aux_args']
        for p in preds
    ]
    slice_starts = [0] + list(np.cumsum(pred_dims[:-1]))
    slice_ends   = list(np.cumsum(pred_dims))
    pred_slices  = [(int(s), int(e)) for s, e in zip(slice_starts, slice_ends)]

    # Per-predecessor live callables + scalar backoffs.
    pred_classifiers = [graph.nodes[p]['classifier'] for p in preds]
    pred_forward_sg  = [graph.edges[p, node]['forward_surrogate'] for p in preds]
    pred_backoffs    = [
        jnp.sum(jnp.asarray(graph.nodes[p]['constraint_backoff']))
        for p in preds
    ]

    # Bounds — concat extendedDS_bounds across preds, optionally standardised.
    lb_parts, ub_parts = [], []
    for p in preds:
        decision_bounds = graph.nodes[p]["extendedDS_bounds"].copy()
        if cfg.solvers.standardised:
            decision_bounds = standardise_model_decisions(graph, decision_bounds, p)
        lb_parts.append(jnp.asarray(decision_bounds[0]).reshape(-1))
        ub_parts.append(jnp.asarray(decision_bounds[1]).reshape(-1))
    lb = jnp.concatenate(lb_parts)
    ub = jnp.concatenate(ub_parts)

    # Node-level classifier + backoff.
    node_classifier = graph.nodes[node]['classifier']
    node_backoff    = jnp.sum(jnp.asarray(graph.nodes[node]['constraint_backoff']))

    # Baked measured-input vector for this sample — same treatment as casadi path.
    v = jnp.asarray(inputs).reshape(1, -1)

    def objective(x):
        # Stack per-predecessor forward-surrogate outputs, concat with `v`,
        # pipe through the node's classifier, negate + subtract backoff.
        fwd_pieces = [
            pred_forward_sg[i](x.reshape(1, -1)[:, s:e]).reshape(1, -1)
            for i, (s, e) in enumerate(pred_slices)
        ]
        combined = jnp.hstack([v] + fwd_pieces)
        return (-node_classifier(combined).reshape(()) - node_backoff).reshape(())

    def constraint(x):
        # Per-predecessor classifier + backoff stacked into a vector g(x) <= 0.
        pieces = []
        for i, (s, e) in enumerate(pred_slices):
            val = (
                pred_classifiers[i](x.reshape(1, -1)[:, s:e]).reshape(-1)
                + pred_backoffs[i]
            )
            pieces.append(val)
        return jnp.concatenate(pieces)

    return objective, constraint, [lb, ub]


# ---------------------------------------------------------------------------
# Sample-by-sample evaluator
# ---------------------------------------------------------------------------

def evaluate_forward_decentralised(inputs, aux, cfg, graph, node):
    """
    Solve the decentralised forward sub-problem for every sample.

    Returns `(evaluations, success_flags)` each of shape `(N_samples, 1)`.
    The evaluation returned is `classifier_node(x*) + backoff` (positive =
    feasible), matching the sign convention of the legacy casadi path.
    """
    n_samples = inputs.shape[0]
    evals, flags = [], []

    for i in range(n_samples):
        objective, constraint, bounds = prepare_forward_decentralised_problem(
            inputs[i, :].reshape(1, -1),
            aux[i, :].reshape(1, -1) if aux is not None else jnp.zeros((1, 0)),
            graph, node, cfg,
        )
        if objective is None or bounds is None:
            evals.append(jnp.zeros((1,)))
            flags.append(jnp.zeros((1,), dtype=bool))
            continue

        lb, ub = bounds
        # Probe constraint width at midpoint.
        x_probe = 0.5 * (lb + ub)
        n_g = int(constraint(x_probe).reshape(-1).shape[0])

        solver = construct_constrained_solver(
            objective, constraint, bounds,
            tol=cfg.solvers.tol,
            constraint_lhs=jnp.full((n_g,), -jnp.inf),
            constraint_rhs=jnp.zeros((n_g,)),
        )
        # Deterministic midpoint start — matches the single-shot casadi pattern.
        x0 = x_probe.reshape(1, -1)
        result = solve_constrained([solver], [x0])

        # Flip sign so we return `classifier + backoff` (positive = feasible),
        # consistent with casadi `ray_evaluation` which also negated.
        evals.append((-result['objective']).reshape(-1))
        flags.append(result['converged'].reshape(-1))

    return (
        jnp.concatenate(evals).reshape(-1, 1),
        jnp.concatenate(flags).reshape(-1, 1),
    )


# ---------------------------------------------------------------------------
# Top-level — the name used by `constraints.constructor` dispatch.
# ---------------------------------------------------------------------------

def forward_constraint_decentralised_evaluator(inputs, aux, cfg, graph, node):
    """
    Drop-in replacement for the legacy casadi class of the same name.

    Serial for now — each sample builds its own concatenated NLP and solves
    via septal.  A pmap shard would be straightforward (same pattern as
    `forward.forward_pmap_batch_evaluator`) if typical batch sizes warrant it.
    """
    return evaluate_forward_decentralised(inputs, aux, cfg, graph, node)
