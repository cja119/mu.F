"""
Cost-to-go (CTG) evaluator.

For each successor of `node`, solves

    min   ctg_surrogate(x, succ_input)                  (regression)
    s.t.  classifier(x, succ_input) <= 0                (live-set feasibility)
          lb <= x <= ub                                 (reduced box)

where `x` is the successor's reduced decision vector (design + non-current-
edge inputs + aux) and `succ_input` is the current node's output slice feeding
that edge, held fixed.

Structure mirrors `prepare_backward_problem` (same standardisation, same
bound-masking) with the CTG surrogate added as objective and the classifier
reused as a feasibility constraint rather than the objective.  Consumes
`warmstarts` produced by the backward constraint evaluator when provided.
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
    "prepare_ctg_problem",
    "evaluate_ctg",
    "ctg_pmap_batch_evaluator",
    "cost_to_go_evaluator",
]


# ---------------------------------------------------------------------------
# Problem builder
# ---------------------------------------------------------------------------

def prepare_ctg_problem(outputs, graph, node, cfg):
    """
    Build the CTG sub-problem for each successor of `node`.

    Returns three dicts keyed by successor: `(objective, constraint, bounds)`,
    each with a list of length `N_samples` carrying per-sample jit'd closures.
    """
    if node is None:
        return None, None, None

    ctg_objective  = {succ: None for succ in graph.successors(node)}
    ctg_constraint = {succ: None for succ in graph.successors(node)}
    ctg_bounds     = {succ: None for succ in graph.successors(node)}

    succ_inputs = get_successor_inputs(graph, node, outputs)

    for succ in graph.successors(node):

        n_d           = graph.nodes[succ]['n_design_args']
        input_indices = np.copy(np.array(
            [n_d + inp for inp in graph.edges[node, succ]['input_indices']]
        ))
        aux_indices   = np.copy(np.array(
            [inp for inp in graph.edges[node, succ]['auxiliary_indices']]
        ))
        fix_indices   = np.hstack([input_indices, aux_indices]).astype(int)

        # standardise fixed inputs if requested
        if cfg.solvers.standardised:
            succ_inputs[succ] = succ_inputs[succ].at[:].set(
                standardise_inputs(graph, succ_inputs[succ], succ, fix_indices)
            )

        # reduced-space bounds (drop the indices held fixed by the edge)
        decision_bounds = graph.nodes[succ]["extendedDS_bounds"].copy()
        ndim = (graph.nodes[succ]['n_design_args']
                + graph.nodes[succ]['n_input_args']
                + graph.graph['n_aux_args'])
        if cfg.solvers.standardised:
            decision_bounds = standardise_model_decisions(graph, decision_bounds, succ)
        decision_bounds = [jnp.delete(bound, fix_indices, axis=1) for bound in decision_bounds]
        ctg_bounds[succ] = [decision_bounds for _ in range(succ_inputs[succ].shape[0])]

        # live JAX callables straight off the graph — no reconstruction
        ctg_surrogate = graph.nodes[succ]["ctg_surrogate"]
        classifier    = graph.nodes[succ]["classifier"]

        wrapped_ctg        = mask_classifier(ctg_surrogate, ndim, input_indices, aux_indices)
        wrapped_classifier = mask_classifier(classifier,    ndim, input_indices, aux_indices)

        # ctg_surrogate may return a vector of per-uncertainty-sample
        # predictions; collapse to a scalar objective via sum to match the
        # casadi path's `_ensure_scalar_objective` behaviour.
        ctg_objective[succ] = [
            jit(partial(lambda x, y: wrapped_ctg(x, y).reshape(-1).sum(),
                        y=succ_inputs[succ][i].reshape(1, -1)))
            for i in range(succ_inputs[succ].shape[0])
        ]
        ctg_constraint[succ] = [
            jit(partial(lambda x, y: wrapped_classifier(x, y).reshape(-1),
                        y=succ_inputs[succ][i].reshape(1, -1)))
            for i in range(succ_inputs[succ].shape[0])
        ]

    return ctg_objective, ctg_constraint, ctg_bounds


# ---------------------------------------------------------------------------
# Shard-level evaluator
# ---------------------------------------------------------------------------

def evaluate_ctg(outputs, aux, graph, node, cfg, sobol_pts_dict=None, warmstarts=None):
    """
    Evaluate the CTG surrogate for each successor of `node` within a pmap shard.

    Builds per-sample solvers, runs them, stacks the scalar `(objective,
    converged)` summaries into `(N_samples, N_successors)` matrices.
    """
    objective, constraint, bounds = prepare_ctg_problem(outputs, graph, node, cfg)
    if objective is None or bounds is None:
        return jnp.zeros((outputs.shape[0], 1)), None

    succ_fn_evaluations = {}
    for succ in graph.successors(node):
        succ_sobol = sobol_pts_dict.get(succ) if sobol_pts_dict is not None else None

        solvers = [
            construct_constrained_solver(
                objective[succ][i], constraint[succ][i], bounds[succ][i],
                tol=cfg.solvers.tol,
                sobol_pts=succ_sobol,
            )
            for i in range(outputs.shape[0])
        ]

        # warmstart per sample if provided; otherwise sobol-sampled initial guess
        if warmstarts is not None and succ in warmstarts:
            initial_guesses = [warmstarts[succ][i].reshape(1, -1)
                               for i in range(outputs.shape[0])]
        else:
            initial_guesses = [initial_guess(cfg.solvers, bounds[succ][i])
                               for i in range(outputs.shape[0])]

        succ_fn_evaluations[succ] = solve_constrained(solvers, initial_guesses)

    fn_evaluations = [
        succ_fn_evaluations[succ]['objective'].reshape(-1, 1)
        for succ in graph.successors(node)
    ]
    success_flags = [
        succ_fn_evaluations[succ]['converged'].reshape(-1, 1)
        for succ in graph.successors(node)
    ]
    return jnp.hstack(fn_evaluations), jnp.hstack(success_flags)


# ---------------------------------------------------------------------------
# pmap shard wrapper
# ---------------------------------------------------------------------------

def ctg_pmap_batch_evaluator(outputs, aux, cfg, graph, node, warmstarts=None):
    """
    Fan the sample axis out across CPU devices.  Sobol points are
    precomputed outside pmap (bounds are static per-successor) and passed in
    via `in_axes=None` as a tuple ordered by the successor list.
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

    def shard_call(outputs_s, aux_s, sobol_tuple, warmstarts_s):
        sobol_dict = None
        if successor_order is not None and sobol_tuple is not None:
            sobol_dict = {succ: pts for succ, pts in zip(successor_order, sobol_tuple)}
        return evaluate_ctg(outputs_s, aux_s, graph, node, cfg,
                            sobol_pts_dict=sobol_dict, warmstarts=warmstarts_s)

    devs = [d for i, d in enumerate(devices('cpu')) if i < outputs.shape[0]]
    return pmap(shard_call, in_axes=(0, 0, None, 0), out_axes=0, devices=devs)(
        outputs, aux, sobol_pts_tuple, warmstarts
    )


# ---------------------------------------------------------------------------
# Top-level evaluator
# ---------------------------------------------------------------------------

def cost_to_go_evaluator(outputs, aux, cfg, graph, node, warmstarts=None):
    """
    Top-level CTG evaluator — mirrors `backward_constraint_evaluator`.

    Shards samples across CPU devices via pmap.  Returns
    `(ctg_evaluations, success_flags)` as `(N_samples, N_successors)` matrices.
    """
    max_devices = cfg.max_devices
    batch_sizes, _ = determine_batches(outputs.shape[0], max_devices)

    output_batches = create_batches(batch_sizes, outputs)
    aux_batches    = create_batches(
        batch_sizes,
        jnp.repeat(jnp.expand_dims(aux, axis=1), outputs.shape[1], axis=1),
    )

    warmstart_batches = None
    if warmstarts is not None:
        per_succ_batches = {succ: create_batches(batch_sizes, warmstarts[succ])
                            for succ in warmstarts}
        warmstart_batches = [
            {succ: per_succ_batches[succ][i] for succ in per_succ_batches}
            for i in range(len(batch_sizes))
        ]

    evals, flags = [], []
    for i, (output_batch, aux_batch) in enumerate(zip(output_batches, aux_batches)):
        ws_i = warmstart_batches[i] if warmstart_batches is not None else None
        evals_i, flags_i = ctg_pmap_batch_evaluator(
            output_batch, aux_batch, cfg, graph, node, warmstarts=ws_i,
        )
        evals.append(evals_i)
        flags.append(flags_i)

    del output_batches, aux_batches, batch_sizes

    return jnp.vstack(evals), jnp.vstack(flags)
