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

### BaseEvaluator port

Single sub-problem per node (`_keys() = [node]`).  The sample-dependent
measured-input vector `v` is threaded as the parametric `p` argument, so
one factory serves every sample.  Loop over samples happens in
`evaluate(...)`, which reuses the compiled factory across all of them.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from mu_F.constraints.evaluators.base import (
    BaseEvaluator,
    build_factory,
    build_penalty_screener,
    pick_best,
    pick_x0_batch,
    precompute_sobol_pool,
)


__all__ = [
    "ForwardDecentralisedEvaluator",
    "forward_constraint_decentralised_evaluator",
]


# =============================================================================
# ForwardDecentralisedEvaluator — one instance per (cfg, graph, node)
# =============================================================================

class ForwardDecentralisedEvaluator(BaseEvaluator):
    """
    Stateful decentralised forward evaluator.

    Single concatenated NLP per node; the per-sample measured input `v`
    is lifted to the parametric `p` so one compiled factory serves every
    sample.
    """

    def __init__(self, cfg, graph, node):
        self.bounds: dict        = {}
        self.n_d: dict           = {}
        self.n_g: dict           = {}
        self.n_fix: dict         = {}
        self.objective_fn: dict  = {}
        self.constraint_fn: dict = {}
        self.pred_slices: dict   = {}

        super().__init__(cfg, graph, node)

        self._thread = self.evaluate

    # ------------------------------------------------------------------
    # BaseEvaluator contract
    # ------------------------------------------------------------------

    def _keys(self) -> list:
        if self.node is None or not list(self.graph.predecessors(self.node)):
            return []
        return [self.node]

    def _build_for_key(self, key) -> None:
        preds = list(self.graph.predecessors(key))

        # Per-predecessor decision dimensions and slice offsets.
        pred_dims = [
            self.graph.nodes[p]['n_design_args']
            + self.graph.nodes[p]['n_input_args']
            + self.graph.graph['n_aux_args']
            for p in preds
        ]
        slice_starts = [0] + list(np.cumsum(pred_dims[:-1]))
        slice_ends   = list(np.cumsum(pred_dims))
        pred_slices  = [(int(s), int(e)) for s, e in zip(slice_starts, slice_ends)]

        # Per-predecessor callables + backoffs.
        pred_classifiers = [self.graph.nodes[p]['classifier'] for p in preds]
        pred_forward_sg  = [self.graph.edges[p, key]['forward_surrogate'] for p in preds]
        pred_backoffs    = [
            jnp.sum(jnp.asarray(self.graph.nodes[p]['constraint_backoff']))
            for p in preds
        ]

        # Bounds — concat extendedDS_bounds across preds, in real-world units.
        lb_parts, ub_parts = [], []
        for p in preds:
            decision_bounds = self.graph.nodes[p]["extendedDS_bounds"].copy()
            lb_parts.append(jnp.asarray(decision_bounds[0]).reshape(-1))
            ub_parts.append(jnp.asarray(decision_bounds[1]).reshape(-1))
        lb = jnp.concatenate(lb_parts)
        ub = jnp.concatenate(ub_parts)
        bounds = [lb, ub]
        n_d = int(lb.size)

        # Node-level classifier + backoff.
        node_classifier = self.graph.nodes[key]['classifier']
        node_backoff    = jnp.sum(jnp.asarray(self.graph.nodes[key]['constraint_backoff']))

        # Determine `p` dimension by probing the first predecessor's
        def _fwd_width(p_idx: int) -> int:
            sl_s, sl_e = pred_slices[p_idx]
            x_probe = 0.5 * (lb[sl_s:sl_e] + ub[sl_s:sl_e])
            return int(pred_forward_sg[p_idx](x_probe.reshape(1, -1)).reshape(-1).size)

        fwd_widths = [_fwd_width(i) for i in range(len(preds))]
        n_fix = int(sum(fwd_widths))

        def objective(x, p):
            # Stack per-predecessor forward-surrogate outputs,
            fwd_pieces = [
                pred_forward_sg[i](x.reshape(1, -1)[:, s:e]).reshape(1, -1)
                for i, (s, e) in enumerate(pred_slices)
            ]
            combined = jnp.hstack([p.reshape(1, -1)] + fwd_pieces)
            return (-node_classifier(combined).reshape(()) - node_backoff).reshape(())

        def constraint(x, p):
            # Per-predecessor classifier + backoff stacked into g(x) <= 0.
            pieces = []
            for i, (s, e) in enumerate(pred_slices):
                val = (
                    pred_classifiers[i](x.reshape(1, -1)[:, s:e]).reshape(-1)
                    + pred_backoffs[i]
                )
                pieces.append(val)
            return jnp.concatenate(pieces)

        # Probe constraint width at midpoint — septal needs explicit n_g.
        x_probe_all = 0.5 * (lb + ub)
        p_probe     = jnp.zeros(n_fix)
        n_g = int(constraint(x_probe_all, p_probe).reshape(-1).shape[0])

        self.factories[key] = build_factory(
            objective, constraint, bounds,
            n_decision=n_d,
            n_params=n_fix,
            n_constraints=n_g,
            tol=self.tol,
        )
        self.screeners[key] = build_penalty_screener(
            objective, constraint, self.screen_penalty,
        )
        self.sobol_pool[key] = precompute_sobol_pool(
            bounds, n_d, self.n_sobol_screen,
        )

        self.bounds[key]        = bounds
        self.n_d[key]           = n_d
        self.n_g[key]           = n_g
        self.n_fix[key]         = n_fix
        self.pred_slices[key]   = pred_slices
        self.objective_fn[key]  = objective
        self.constraint_fn[key] = constraint

    # ------------------------------------------------------------------
    # Evaluate — loops over samples; each iteration reuses the compiled
    # factory with a new `p`.
    # ------------------------------------------------------------------

    def evaluate(self, inputs, aux):
        """
        Solve the decentralised forward NLP for every sample.

        Returns `(evaluations, success_flags)` each shape `(N_samples, 1)`.
        The evaluation returned is `classifier_node(x*) + backoff`
        (positive = feasible).
        """
        key = self.node
        if self._keys() == []:
            n = inputs.shape[0]
            return jnp.zeros((n, 1)), jnp.zeros((n, 1), dtype=bool)

        n_fix = self.n_fix[key]
        n_samples = inputs.shape[0]

        evals, flags = [], []
        for i in range(n_samples):
            v = inputs[i].reshape(-1)
            p = jnp.zeros(n_fix).at[:min(v.size, n_fix)].set(v[:n_fix])

            x0_batch = pick_x0_batch(
                self.sobol_pool[key], self.screeners[key], p, self.n_starts,
            )
            p_batch = jnp.broadcast_to(
                p.reshape(1, -1), (self.n_starts, n_fix),
            )

            result = self.factories[key].solve_batch(x0_batch, p_batch)
            best_f, best_c, _ = pick_best(result)

            evals.append((-best_f).reshape(-1))
            flags.append(best_c.reshape(-1))

        return (
            jnp.concatenate(evals).reshape(-1, 1),
            jnp.concatenate(flags).reshape(-1, 1),
        )


# =============================================================================
# Evaluator cache + top-level entry point
# =============================================================================

_FWDDEC_EVALUATOR_CACHE: dict = {}


def _get_evaluator(cfg, graph, node) -> ForwardDecentralisedEvaluator:
    key = (id(graph), node)
    evaluator = _FWDDEC_EVALUATOR_CACHE.get(key)
    if evaluator is None:
        evaluator = ForwardDecentralisedEvaluator(cfg, graph, node)
        _FWDDEC_EVALUATOR_CACHE[key] = evaluator
    return evaluator


def forward_constraint_decentralised_evaluator(inputs, aux, cfg, graph, node):
    """
    Drop-in replacement for the legacy casadi class of the same name.

    Serial across samples for now — each sample reuses the compiled
    factory with a new `p`.  A pmap thread would be straightforward if
    typical batch sizes warrant it.
    """
    return _get_evaluator(cfg, graph, node).evaluate(inputs, aux)
