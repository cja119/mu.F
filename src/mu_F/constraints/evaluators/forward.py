"""
Forward coupling constraint evaluator.

For each predecessor of `node`, solves

    min   classifier_pred(x) + sum(backoff)               (stay in pred's live set)
    s.t.  forward_surrogate(x) = measured_input           (equality — match observation)
          lb_pred <= x <= ub_pred                         (pred's full NLP box)

Decision space is pred's **full** NLP (design + input + aux).  Unlike the
backward/CTG reduced space, we don't mask inputs out — they're pinned by the
equality constraint itself.

The equality is written in septal's static-lhs/rhs form as
`g(x, p) = forward_surrogate(x) - p` with `lhs = rhs = 0`, so the measured
input rides in the closure as `y` (per-sample).  The baked-in `y` triggers a
JIT trace per sample; that's the known p-lifting issue noted elsewhere and
gets addressed in one sweep across all evaluators later.

NOTE: does NOT implement `cfg.solvers.forward_general_constraints` — the
variant that stacks extra successor classifier+surrogate blocks for already-
processed downstream nodes.  Off by default in every yaml in the repo; land
as a follow-up if a case study needs it.
"""
from __future__ import annotations

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit, pmap, devices

from mu_F.constraints.utils import (
    standardise_model_decisions,
    initial_guess,
    get_forward_bounds,
    generate_initial_guess,
    determine_batches,
    create_batches,
)
from mu_F.solvers.septal import (
    construct_constrained_solver,
    solve_constrained,
)


__all__ = [
    "prepare_forward_problem",
    "evaluate_forward",
    "forward_pmap_batch_evaluator",
    "forward_constraint_evaluator",
]


# ---------------------------------------------------------------------------
# Graph helpers — small, private to this file
# ---------------------------------------------------------------------------

def _get_predecessor_inputs(inputs, aux, graph, node):
    """
    Split `inputs` / `aux` into per-edge slices keyed by predecessor.

    Uses the edge-stored `input_indices` to pick out the subset of the
    current node's inputs coming from each predecessor.
    """
    pred_inputs, pred_aux = {}, {}
    for pred in graph.predecessors(node):
        input_indices = np.array(graph.edges[pred, node]['input_indices'], dtype=int)
        pred_inputs[pred] = inputs[:, input_indices]
        pred_aux[pred] = aux[:, :]
    return pred_inputs, pred_aux


def _standardise_edge_outputs(graph, data, pred, node):
    """
    Apply edge-level `y_scalar` standardisation to data flowing through the
    (pred, node) edge.  Returns `data` unchanged if no scaler is attached —
    mirrors the casadi `standardise_inputs` fallback.
    """
    scaler = graph.edges[pred, node].get('y_scalar')
    if scaler is None:
        return data
    mean = jnp.asarray(scaler.mean).reshape(1, -1)
    std = jnp.asarray(scaler.std).reshape(1, -1)
    return (data - mean) / std


# ---------------------------------------------------------------------------
# Problem builder
# ---------------------------------------------------------------------------

def prepare_forward_problem(inputs, aux, graph, node, cfg):
    """
    Build a forward sub-problem per predecessor of `node`.

    Returns `(objective, constraint, bounds)` dicts keyed by predecessor,
    each value a list of length `N_samples` (one solver per sample).
    Objective (classifier+backoff) and constraint (forward_surrogate - target)
    are pulled as live JAX callables from the graph — no reconstruction from
    serialised weights.
    """
    if node is None:
        return None, None, None

    fwd_objective  = {pred: None for pred in graph.predecessors(node)}
    fwd_constraint = {pred: None for pred in graph.predecessors(node)}
    fwd_bounds     = {pred: None for pred in graph.predecessors(node)}

    pred_inputs, pred_aux = _get_predecessor_inputs(inputs, aux, graph, node)

    for pred in graph.predecessors(node):

        # pred's full NLP dimension — no reduction; equality constraint pins inputs.
        n_design = graph.nodes[pred]['n_design_args']
        n_input  = graph.nodes[pred]['n_input_args']
        n_aux    = graph.graph['n_aux_args']
        n_d_j    = n_design + n_input + n_aux  # noqa: F841 — documented, kept for clarity

        # Bounds — pred's extendedDS_bounds, optionally standardised.
        decision_bounds = graph.nodes[pred]["extendedDS_bounds"].copy()
        if cfg.solvers.standardised:
            decision_bounds = standardise_model_decisions(graph, decision_bounds, pred)
        lb = jnp.asarray(decision_bounds[0]).reshape(-1)
        ub = jnp.asarray(decision_bounds[1]).reshape(-1)
        fwd_bounds[pred] = [[lb, ub] for _ in range(pred_inputs[pred].shape[0])]

        # Measured-input target for the equality constraint.
        # [pred_inputs | pred_aux] — matches the forward surrogate's output layout.
        target = jnp.hstack([pred_inputs[pred], pred_aux[pred]])
        if cfg.solvers.standardised:
            target = _standardise_edge_outputs(graph, target, pred, node)

        # Live JAX callables from the graph.
        classifier        = graph.nodes[pred]["classifier"]
        forward_surrogate = graph.edges[pred, node]["forward_surrogate"]
        # `constraint_backoff` is a per-output-channel scalar list; the casadi
        # path sums it into the scalar objective via `_ensure_scalar_objective`.
        # Collapse to a scalar explicitly here so septal gets a scalar f(x, p).
        backoff = jnp.sum(jnp.asarray(graph.nodes[pred]['constraint_backoff']))

        fwd_objective[pred] = [
            jit(partial(
                lambda x, b: classifier(x.reshape(1, -1)).reshape(()) + b,
                b=backoff,
            ))
            for _ in range(target.shape[0])
        ]
        fwd_constraint[pred] = [
            jit(partial(
                lambda x, y: forward_surrogate(x.reshape(1, -1)).reshape(-1) - y.reshape(-1),
                y=target[i],
            ))
            for i in range(target.shape[0])
        ]

    return fwd_objective, fwd_constraint, fwd_bounds


# ---------------------------------------------------------------------------
# Shard-level evaluator
# ---------------------------------------------------------------------------

def evaluate_forward(inputs, aux, graph, node, cfg, sobol_pts_dict=None):
    """
    Evaluate the forward problem per predecessor within a pmap shard.

    Mirrors `evaluate_ctg` structurally: build one solver per sample per
    predecessor, hand each a sobol-seeded initial-guess batch, stack the
    scalar `(objective, converged)` summaries across predecessors.
    """
    objective, constraint, bounds = prepare_forward_problem(inputs, aux, graph, node, cfg)
    if objective is None or bounds is None:
        return jnp.zeros((inputs.shape[0], 1)), None

    pred_fn_evaluations = {}
    for pred in graph.predecessors(node):
        pred_sobol = sobol_pts_dict.get(pred) if sobol_pts_dict is not None else None

        n_samples = inputs.shape[0]
        # Equality constraint — septal needs lhs = rhs = 0 with matching width.
        # Probe the constraint at midpoint to discover n_g.
        lb = bounds[pred][0][0]
        ub = bounds[pred][0][1]
        x_probe = 0.5 * (lb + ub)
        n_g = int(constraint[pred][0](x_probe).reshape(-1).shape[0])
        eq = jnp.zeros((n_g,))

        solvers = [
            construct_constrained_solver(
                objective[pred][i], constraint[pred][i], bounds[pred][i],
                tol=cfg.solvers.tol,
                sobol_pts=pred_sobol,
                constraint_lhs=eq,
                constraint_rhs=eq,
            )
            for i in range(n_samples)
        ]
        initial_guesses = [
            initial_guess(cfg.solvers, bounds[pred][i])
            for i in range(n_samples)
        ]
        pred_fn_evaluations[pred] = solve_constrained(solvers, initial_guesses)

    fn_evaluations = [
        pred_fn_evaluations[pred]['objective'].reshape(-1, 1)
        for pred in graph.predecessors(node)
    ]
    success_flags = [
        pred_fn_evaluations[pred]['converged'].reshape(-1, 1)
        for pred in graph.predecessors(node)
    ]
    return jnp.hstack(fn_evaluations), jnp.hstack(success_flags)


# ---------------------------------------------------------------------------
# pmap shard wrapper
# ---------------------------------------------------------------------------

def forward_pmap_batch_evaluator(inputs, aux, cfg, graph, node):
    """
    Fan the sample axis out across CPU devices.  Sobol points are
    precomputed outside pmap (bounds are static per-predecessor) and passed
    in via `in_axes=None` as a tuple ordered by the predecessor list.
    """
    n_sobol_screen    = getattr(cfg.solvers, 'n_sobol_screen', 16_384)
    forward_bounds    = get_forward_bounds(graph, node, cfg)
    sobol_pts_tuple   = ()
    predecessor_order = None
    if forward_bounds is not None:
        predecessor_order = list(forward_bounds.keys())
        sobol_pts_tuple = tuple(
            generate_initial_guess(n_sobol_screen, None, bounds)
            for bounds in forward_bounds.values()
        )

    def shard_call(inputs_s, aux_s, sobol_tuple):
        sobol_dict = None
        if predecessor_order is not None and sobol_tuple is not None:
            sobol_dict = {pred: pts for pred, pts in zip(predecessor_order, sobol_tuple)}
        return evaluate_forward(inputs_s, aux_s, graph, node, cfg,
                                sobol_pts_dict=sobol_dict)

    devs = [d for i, d in enumerate(devices('cpu')) if i < inputs.shape[0]]
    return pmap(shard_call, in_axes=(0, 0, None), out_axes=0, devices=devs)(
        inputs, aux, sobol_pts_tuple,
    )


# ---------------------------------------------------------------------------
# Top-level evaluator
# ---------------------------------------------------------------------------

def forward_constraint_evaluator(inputs, aux, cfg, graph, node):
    """
    Top-level forward evaluator — mirrors `backward_constraint_evaluator`.

    Shards samples across CPU devices via pmap.  Returns
    `(fwd_evaluations, success_flags)` as `(N_samples, N_predecessors)`
    matrices hstacked across predecessors.
    """
    max_devices = cfg.max_devices
    batch_sizes, _ = determine_batches(inputs.shape[0], max_devices)

    input_batches = create_batches(batch_sizes, inputs)
    aux_batches   = create_batches(
        batch_sizes,
        jnp.repeat(jnp.expand_dims(aux, axis=1), inputs.shape[1], axis=1),
    )

    evals, flags = [], []
    for input_batch, aux_batch in zip(input_batches, aux_batches):
        evals_i, flags_i = forward_pmap_batch_evaluator(
            input_batch, aux_batch, cfg, graph, node,
        )
        evals.append(evals_i)
        flags.append(flags_i)

    del input_batches, aux_batches, batch_sizes

    return jnp.vstack(evals), jnp.vstack(flags)
