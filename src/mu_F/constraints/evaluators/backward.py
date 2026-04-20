"""
Backward coupling constraint evaluator (box-only feasibility check).

For each successor of `node`, solves

    min  classifier_succ(x, succ_input)       (live-set feasibility score)
    s.t. lb <= x <= ub                        (reduced box; no general constraints)

where `x` is the successor's reduced decision vector (design + non-current-
edge inputs + aux) and `succ_input` is the current node's output slice feeding
that edge, held fixed via `mask_classifier`.

Also houses:

  - `construct_solver` / `solve` — box-only aliases for
    `construct_constrained_solver` / `solve_constrained` with
    `constraint_func=None`.  Kept here because every evaluator in the repo
    that wants a box-only solver reaches for this pair.
  - `prepare_global_problem` / `evaluate` — the graph-wide (post-processing)
    variant + the branching dispatcher.  Box-only in both branches.
"""
from __future__ import annotations

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit, pmap, devices

from mu_F.constraints.utils import (
    standardise_inputs,
    standardise_model_decisions,
    mask_classifier,
    get_successor_inputs,
    shaping_function,
    initial_guess,
    get_backward_bounds,
    generate_initial_guess,
    determine_batches,
    create_batches,
)
from mu_F.solvers.septal import (
    construct_constrained_solver,
    solve_constrained,
)


__all__ = [
    # Box-only solver aliases (used by every evaluator that wants `g=None`).
    "construct_solver",
    "solve",
    # Backward / global problem builders and evaluators.
    "prepare_backward_problem",
    "prepare_global_problem",
    "evaluate",
    "jax_pmap_evaluator",
    "backward_surrogate_pmap_batch_evaluator",
    "backward_constraint_evaluator",
]


# ---------------------------------------------------------------------------
# Box-only solver aliases.  `construct_solver` and `solve` originated as the
# jaxopt-LBFGSB entry points; post-2b they delegate straight to septal with
# `constraint_func=None`.  Callers elsewhere import these names for box-only
# problems.
# ---------------------------------------------------------------------------

def construct_solver(objective_func, bounds, tol, sobol_pts=None):
    """Box-only NLP solver — thin wrapper over septal with `g=None`."""
    return construct_constrained_solver(
        objective_func,
        constraint_func=None,
        bounds=bounds,
        tol=tol,
        sobol_pts=sobol_pts,
    )


def solve(solver, initial_guesses):
    """Alias for `solve_constrained` — unified dict-returning driver."""
    return solve_constrained(solver, initial_guesses)


# ---------------------------------------------------------------------------
# Node-local backward problem
# ---------------------------------------------------------------------------

def prepare_backward_problem(outputs, graph, node, cfg):
    """
    Build the backward sub-problem for each successor of `node`.

    Decision variables live in the successor's reduced space (design +
    non-current-edge inputs + aux).  Objective is the successor's classifier
    masked to scatter the fixed `succ_inputs` into the full NLP at call time.
    No general constraints — box-only.
    """
    if node is None:
        return None, None

    # TODO: confirm this doesn't trip pmap tracing when node has no successors
    backward_bounds    = {succ: None for succ in graph.successors(node)}
    backward_objective = {succ: None for succ in graph.successors(node)}

    # inputs carried across each out-edge from `node`
    succ_inputs = get_successor_inputs(graph, node, outputs)

    for succ in graph.successors(node):

        n_d           = graph.nodes[succ]['n_design_args']
        input_indices = np.copy(np.array(
            [n_d + input_ for input_ in graph.edges[node, succ]['input_indices']]
        ))
        aux_indices   = np.copy(np.array(
            [input_ for input_ in graph.edges[node, succ]['auxiliary_indices']]
        ))

        if cfg.solvers.standardised:
            succ_inputs[succ] = succ_inputs[succ].at[:].set(
                standardise_inputs(
                    graph, succ_inputs[succ], succ,
                    jnp.hstack([input_indices, aux_indices]).astype(int),
                )
            )

        decision_bounds = graph.nodes[succ]["extendedDS_bounds"].copy()
        ndim = (graph.nodes[succ]['n_design_args']
                + graph.nodes[succ]['n_input_args']
                + graph.graph['n_aux_args'])
        if cfg.solvers.standardised:
            decision_bounds = standardise_model_decisions(graph, decision_bounds, succ)

        decision_bounds = [
            jnp.delete(bound, np.hstack([input_indices, aux_indices]).astype(int), axis=1)
            for bound in decision_bounds
        ]
        backward_bounds[succ] = [decision_bounds.copy() for _ in range(succ_inputs[succ].shape[0])]

        classifier         = graph.nodes[succ]["classifier"]
        wrapped_classifier = mask_classifier(classifier, ndim, input_indices, aux_indices)
        backward_objective[succ] = [
            jit(partial(
                lambda x, y: wrapped_classifier(x, y).squeeze(),
                y=succ_inputs[succ][i].reshape(1, -1),
            ))
            for i in range(succ_inputs[succ].shape[0])
        ]

    return backward_objective, backward_bounds


# ---------------------------------------------------------------------------
# Graph-wide post-processing problem
# ---------------------------------------------------------------------------

def prepare_global_problem(inputs, aux, graph, cfg):
    """
    Build the graph-wide (post-processing) feasibility problem.

    Loads the graph-level post-process classifier, masks off the fixed
    indices, and returns `(objective, bounds)` on the reduced design space.
    The sign is flipped to a maximisation (negative = infeasible).
    """
    n_d   = graph.graph['n_design_args']  # design vars in root-node successors
    n_aux = graph.graph['n_aux_args']

    dec_ind   = np.array(graph.graph['post_process_decision_indices'])
    total_ind = np.arange(n_d + n_aux)
    fix_ind   = np.delete(total_ind, dec_ind).astype(int)

    lb = jnp.hstack([jnp.array(bound[0]).reshape(-1)
                     for bound in graph.graph['bounds'] if bound[0] != 'None'])
    ub = jnp.hstack([jnp.array(bound[1]).reshape(-1)
                     for bound in graph.graph['bounds'] if bound[1] != 'None'])
    bounds = [lb, ub]

    if cfg.solvers.standardised:
        inputs = standardise_inputs(graph, inputs, None,
                                    jnp.hstack([fix_ind]).astype(int))
        bounds = standardise_model_decisions(graph, bounds, None)

    classifier = mask_classifier(
        graph.graph['post_process_lower_classifier'],
        n_d + n_aux, fix_ind, np.empty((0,)).astype(int),
    )

    # Maximisation formulation — negative objective = infeasible.
    objective_func = partial(
        lambda x, y: -classifier(x, y).squeeze(),
        y=inputs.reshape(1, -1),
    )

    bounds = [jnp.delete(bounds[0], fix_ind), jnp.delete(bounds[1], fix_ind)]
    return objective_func, bounds


# ---------------------------------------------------------------------------
# Branching dispatcher: graph-wide vs. node-local
# ---------------------------------------------------------------------------

def evaluate(outputs, aux, graph, node, cfg, sobol_pts_dict=None):
    """
    Dispatch to the graph-wide or node-local backward branch depending on
    `graph.graph["solve_post_processing_problem"]`.

    Returns `(shaped_objective, warmstart_params_or_None)` in the node-local
    case; in the graph-wide case returns `(shaped_objective, None)`.
    """
    evaluate_method = solve

    def graph_wide_branch(args):
        (outputs, aux) = args
        if outputs.ndim < 2:
            outputs = outputs.reshape(-1, 1)
        if aux.ndim < 2:
            aux = aux.reshape(-1, 1)
        objective, bounds = prepare_global_problem(outputs, aux, graph, cfg)
        solver = construct_solver(objective, bounds,
                                  tol=cfg.solvers.tol)
        initial_guesses = initial_guess(cfg.solvers, bounds)
        result = evaluate_method([solver], [initial_guesses])
        fn_evaluations = result['objective'].reshape(-1, 1)
        return -shaping_function(fn_evaluations, cfg)  # maximisation

    def node_local_branch(args):
        (outputs, aux) = args
        objective, bounds = prepare_backward_problem(outputs, graph, node, cfg)
        if objective is None or bounds is None:
            return jnp.zeros((outputs.shape[0], 1)), None

        succ_fn_evaluations = {}
        for succ in graph.successors(node):
            succ_sobol = sobol_pts_dict.get(succ) if sobol_pts_dict is not None else None
            backward_solver = [
                construct_solver(
                    objective[succ][i], bounds[succ][i],
                    tol=cfg.solvers.tol,
                    sobol_pts=succ_sobol,
                )
                for i in range(outputs.shape[0])
            ]
            initial_guesses = [
                initial_guess(cfg.solvers, bounds[succ][i])
                for i in range(outputs.shape[0])
            ]
            succ_fn_evaluations[succ] = evaluate_method(backward_solver, initial_guesses)

        fn_evaluations = [
            succ_fn_evaluations[succ]['objective'].reshape(-1, 1)
            for succ in graph.successors(node)
        ]
        warmstart_params = {
            succ: succ_fn_evaluations[succ]['params'] for succ in graph.successors(node)
        }
        return shaping_function(jnp.hstack(fn_evaluations), cfg), warmstart_params

    is_graph_wide = bool(graph.graph["solve_post_processing_problem"])
    if is_graph_wide:
        return graph_wide_branch((outputs, aux)), None
    else:
        return node_local_branch((outputs, aux))


# ---------------------------------------------------------------------------
# pmap shard + top-level evaluator
# ---------------------------------------------------------------------------

def jax_pmap_evaluator(outputs, aux, sobol_pts_tuple, cfg, graph, node,
                       successor_order=None):
    """
    Single-shard entry point — reconstructs the `sobol_pts_dict` from the
    ordered tuple passed through pmap, then calls `evaluate`.
    """
    sobol_pts_dict = None
    if successor_order is not None and sobol_pts_tuple is not None:
        sobol_pts_dict = {succ: pts for succ, pts in zip(successor_order, sobol_pts_tuple)}

    constraint_evaluator = partial(
        evaluate, graph=graph, node=node, cfg=cfg, sobol_pts_dict=sobol_pts_dict,
    )
    return constraint_evaluator(outputs, aux)


def backward_surrogate_pmap_batch_evaluator(outputs, aux, cfg, graph, node):
    """
    Fan samples out across CPU devices via pmap.  Sobol points precomputed
    outside pmap (bounds are static per-successor) and passed in with
    `in_axes=None` as a tuple ordered by the successor list.
    """
    n_sobol_screen  = getattr(cfg.solvers, 'n_sobol_screen', 16_384)
    backward_bounds = get_backward_bounds(graph, node, cfg)
    sobol_pts_tuple = ()
    successor_order = None
    if backward_bounds is not None:
        successor_order = list(backward_bounds.keys())
        sobol_pts_tuple = tuple(
            generate_initial_guess(n_sobol_screen, None, bounds)
            for bounds in backward_bounds.values()
        )

    feasibility_call = partial(
        jax_pmap_evaluator, cfg=cfg, graph=graph, node=node,
        successor_order=successor_order,
    )

    # sobol_pts_tuple passed as pmap arg with in_axes=None so it's a dynamic
    # input, not a traced constant.
    return pmap(
        feasibility_call, in_axes=(0, 0, None), out_axes=0,
        devices=[d for i, d in enumerate(devices('cpu')) if i < outputs.shape[0]],
    )(outputs, aux, sobol_pts_tuple)


def backward_constraint_evaluator(outputs, aux, cfg, graph, node):
    """
    Top-level backward feasibility evaluator — box-only, shards samples
    across CPU devices via pmap.  Returns `(evaluations, warmstarts)` where
    `warmstarts` is a dict keyed by successor (consumed by the CTG path).
    """
    max_devices = cfg.max_devices
    batch_sizes, _ = determine_batches(outputs.shape[0], max_devices)

    output_batches = create_batches(batch_sizes, outputs)
    aux_batches    = create_batches(
        batch_sizes,
        jnp.repeat(jnp.expand_dims(aux, axis=1), outputs.shape[1], axis=1),
    )

    results, warmstarts = [], []
    for output_batch, aux_batch in zip(output_batches, aux_batches):
        evals, params = backward_surrogate_pmap_batch_evaluator(
            output_batch, aux_batch, cfg, graph, node,
        )
        results.append(evals)
        if params is not None:
            warmstarts.append(params)

    warmstarts = (
        {succ: jnp.vstack([w[succ] for w in warmstarts]) for succ in warmstarts[0]}
        if warmstarts else None
    )

    del output_batches, aux_batches, batch_sizes

    return jnp.vstack(results), warmstarts
