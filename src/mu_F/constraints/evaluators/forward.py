"""
Forward coupling constraint evaluator.

For each predecessor of `node`, solves

    min   classifier_pred(x) + sum(backoff)               (stay in pred's live set)
    s.t.  forward_surrogate(x) = measured_input           (equality — match observation)
          lb_pred <= x <= ub_pred                         (pred's full NLP box)

Decision space is pred's **full** NLP (design + input + aux).  Unlike the
backward/CTG reduced space, we don't mask inputs out — they're pinned by the
equality constraint itself.

### p-lifting

The measured-input target `y` is threaded through as the parametric `p`
argument of septal's `ParametricNLPProblem` instead of being baked into a
Python closure.  The equality becomes

    g(x, p) = forward_surrogate(x) - p

with `constraint_lhs = constraint_rhs = 0`.  One
`ParametricSQPFactory` per predecessor is built in `_build_for_key`; every
subsequent thread call feeds new `(x_batch, p_batch)` to the same compiled
scan body.

"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import devices

from mu_F.constraints.evaluators.base import (
    BaseEvaluator,
    build_factory,
    build_penalty_screener,
    cached_parallel_thread,
    parallel_thread,
    pick_best,
    pick_x0_batch,
    precompute_sobol_pool,
    skip_if_masked,
)
from mu_F.constraints.utils import (
    pad_to_multiple,
    batch_mask,
    poison_padded,
)


__all__ = [
    "ForwardEvaluator",
    "forward_constraint_evaluator",
]


# =============================================================================
# ForwardEvaluator — one instance per (cfg, graph, node)
# =============================================================================

class ForwardEvaluator(BaseEvaluator):
    """
    Stateful forward evaluator.

    Owns the factory + screener + sobol pool for every predecessor.  The
    thread entry point `evaluate(inputs_s, aux_s)` is pinned to a stable
    bound-method attribute in `__init__` so pmap's compile cache keys on
    the same callable identity across every call.
    """

    def __init__(self, cfg, graph, node):
        # Per-predecessor static state populated in `_build_for_key`.
        self.n_g: dict           = {}
        self.n_fix: dict         = {}
        self.n_d: dict           = {}
        self.bounds: dict        = {}
        self.input_indices: dict = {}
        self.objective_fn: dict  = {}
        self.constraint_fn: dict = {}

        super().__init__(cfg, graph, node)

        self._thread = self.evaluate

    # ------------------------------------------------------------------
    # BaseEvaluator contract
    # ------------------------------------------------------------------

    def _keys(self) -> list:
        if self.node is None:
            return []
        return list(self.graph.predecessors(self.node))

    def _build_for_key(self, pred: int) -> None:
        """
        Build the per-predecessor parametric NLP.

        Decision space is pred's full NLP.  `p` is the measured-input
        target (same width as the forward surrogate's output).
        """
        # pred's full NLP dimension — no reduction; equality constraint pins
        # the inputs.
        n_design = self.graph.nodes[pred]['n_design_args']
        n_input  = self.graph.nodes[pred]['n_input_args']
        n_aux    = self.graph.graph['n_aux_args']
        n_d      = n_design + n_input + n_aux

        # Bounds — pred's extendedDS_bounds, in real-world units.  The
        # classifier / forward surrogate callables self-scale internally.
        decision_bounds = self.graph.nodes[pred]["extendedDS_bounds"].copy()
        lb = jnp.asarray(decision_bounds[0]).reshape(-1)
        ub = jnp.asarray(decision_bounds[1]).reshape(-1)
        bounds = [lb, ub]

        # `input_indices` picks out the pred->node slice of the current
        # node's input stream; used at thread time to build `p`.
        input_indices = np.array(
            self.graph.edges[pred, self.node]['input_indices'], dtype=int,
        )

        # Live callables — keep stable identity from the graph.
        classifier        = self.graph.nodes[pred]["classifier"]
        forward_surrogate = self.graph.edges[pred, self.node]["forward_surrogate"]
        backoff = jnp.sum(jnp.asarray(self.graph.nodes[pred]['constraint_backoff']))

        # Discover forward-surrogate output width at build time to know `p`
        # dimension + equality constraint width.
        x_probe = 0.5 * (lb + ub)
        fwd_probe = forward_surrogate(x_probe.reshape(1, -1)).reshape(-1)
        n_g = int(fwd_probe.size)
        n_fix = n_g  # `p` matches the equality constraint width.

        def objective(x, p):
            return classifier(x.reshape(1, -1)).reshape(()) + backoff

        def constraint(x, p):
            return forward_surrogate(x.reshape(1, -1)).reshape(-1) - p.reshape(-1)

        eq = jnp.zeros((n_g,))
        self.factories[pred] = build_factory(
            objective, constraint, bounds,
            n_decision=n_d,
            n_params=n_fix,
            n_constraints=n_g,
            tol=self.tol,
            constraint_lhs=eq,
            constraint_rhs=eq,
        )
        self.screeners[pred] = build_penalty_screener(
            objective, constraint, self.screen_penalty,
        )
        self.sobol_pool[pred] = precompute_sobol_pool(
            bounds, n_d, self.n_sobol_screen,
        )

        self.n_g[pred]           = n_g
        self.n_fix[pred]         = n_fix
        self.n_d[pred]           = n_d
        self.bounds[pred]        = bounds
        self.input_indices[pred] = input_indices
        self.objective_fn[pred]  = objective
        self.constraint_fn[pred] = constraint

    # ------------------------------------------------------------------
    # Shard entry point — pmap target
    # ------------------------------------------------------------------

    def evaluate(self, inputs_s, aux_s, mask_s):
        """
        Shard body.

        Parameters
        ----------
        inputs_s : (N_uncertainty, N_input_dim) — current node's measured inputs
        aux_s    : (N_uncertainty, N_aux)       — auxiliary variables
        mask_s   : scalar bool — True on real lanes, False on padded lanes

        Returns `(evals, flags)` each shape `(1, N_predecessors)` for this
        thread.  Padded lanes skip the SQP via `lax.cond`.
        """
        def real():
            evals, flags = [], []
            for pred in self._keys():

                pred_inputs = inputs_s[0, self.input_indices[pred]]
                pred_aux    = aux_s[0]
                target = jnp.hstack([pred_inputs, pred_aux])

                n_fix = self.n_fix[pred]
                y = jnp.zeros(n_fix).at[:min(target.size, n_fix)].set(
                    target[:n_fix]
                )

                x0_batch = pick_x0_batch(
                    self.sobol_pool[pred],
                    self.screeners[pred],
                    y,
                    self.n_starts,
                )
                p_batch = jnp.broadcast_to(
                    y.reshape(1, -1), (self.n_starts, n_fix),
                )

                result = self.factories[pred].solve_batch(x0_batch, p_batch)
                best_f, best_c, _ = pick_best(result)

                evals.append(best_f.reshape(-1, 1))
                flags.append(best_c.reshape(-1, 1))

            return jnp.hstack(evals), jnp.hstack(flags)

        return skip_if_masked(mask_s, real)


# =============================================================================
# Evaluator cache + top-level entry point
# =============================================================================

_FORWARD_EVALUATOR_CACHE: dict = {}


def _get_evaluator(cfg, graph, node) -> ForwardEvaluator:
    key = (id(graph), node)
    evaluator = _FORWARD_EVALUATOR_CACHE.get(key)
    if evaluator is None:
        evaluator = ForwardEvaluator(cfg, graph, node)
        _FORWARD_EVALUATOR_CACHE[key] = evaluator
    return evaluator


def forward_constraint_evaluator(inputs, aux, cfg, graph, node):
    """
    Top-level forward evaluator — fixed-width pmap with padding + mask
    (see `cost_to_go_evaluator` for the full rationale).

    Parameters
    ----------
    inputs : (N_batch, N_uncertainty, N_input_dim)
    aux    : (N_batch, N_aux)

    Returns
    -------
    evaluations   : (N_batch, N_predecessors) — NaN on padded rows.
    success_flags : (N_batch, N_predecessors)
    """
    evaluator = _get_evaluator(cfg, graph, node)
    if evaluator._keys() == []:
        return jnp.zeros((inputs.shape[0], 1)), None

    cpu_devs = list(devices('cpu'))
    W = min(int(cfg.max_devices), len(cpu_devs))
    n_real = inputs.shape[0]

    aux_expanded = jnp.repeat(
        jnp.expand_dims(aux, axis=1), inputs.shape[1], axis=1,
    )
    padded_in,  _, _ = pad_to_multiple(inputs, W, axis=0)
    padded_aux, _, _ = pad_to_multiple(aux_expanded, W, axis=0)
    total = padded_in.shape[0]
    mask = batch_mask(n_real, total)

    n_chunks = total // W
    in_chunks   = padded_in.reshape((n_chunks, W) + padded_in.shape[1:])
    aux_chunks  = padded_aux.reshape((n_chunks, W) + padded_aux.shape[1:])
    mask_chunks = mask.reshape(n_chunks, W)

    devs = cpu_devs[:W]
    pmap_fn = cached_parallel_thread(
        evaluator,
        '_pmap_cache',
        evaluator._thread,
        in_axes=(0, 0, 0),
        devices=devs,
        use_vmap=bool(getattr(cfg.solvers, "use_vmap", False)),
    )

    evals_chunks, flags_chunks = [], []
    for i in range(n_chunks):
        e, f = pmap_fn(in_chunks[i], aux_chunks[i], mask_chunks[i])
        evals_chunks.append(e)
        flags_chunks.append(f)

    full_evals = jnp.concatenate(evals_chunks, axis=0)
    full_flags = jnp.concatenate(flags_chunks, axis=0)
    full_evals = poison_padded(full_evals, mask, fill=jnp.nan)
    full_flags = poison_padded(full_flags, mask, fill=False)
    return full_evals[:n_real], full_flags[:n_real]
