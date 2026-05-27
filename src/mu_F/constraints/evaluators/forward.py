"""
Forward coupling constraint evaluator.

For each predecessor of `node`, solves

    min   classifier_pred(x) + sum(backoff)               (stay in pred's live set)
    s.t.  forward_surrogate(x) = target                   (equality — match observation)
          lb_pred <= x <= ub_pred                         (pred's full NLP box)

Decision space is pred's **full** NLP (design + input + aux).  Unlike
backward/CTG which reduce by masking input/aux, here the input slots stay
in the decision vector — they're pinned by the equality constraint itself.
Integer design dims still route through the parametric tail of `p_aug`
via `IntegerProblem`.

### Parametric tail layout

The classifier (scalar / `onehot_sum`) and the forward surrogate
(`vector_diff`) share the same `p_aug`.  To keep the layouts compatible
we pass `n_y = n_g` (forward surrogate output width) to *both* masks:

    p_aug = [ target (n_g) | int_values (n_int) | head_onehot (n_heads) ]

The classifier's empty `fix_ind` / `aux_ind` means it ignores the leading
`target` slice — but `n_y=n_g` ensures the int / head slots sit at the
same offset for both callables.

### Theta axis

`evaluate(inputs_s, aux_s, mask_s)` vmaps `solve_integer_nlp` over the
uncertainty axis of `inputs_s` — trivially `n_theta = 1` in deterministic
settings but kept explicit (no silent axis drop).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import devices

from mu_F.constraints.evaluators.base import (
    BaseEvaluator,
    build_factory,
    build_penalty_screener,
    cached_parallel_thread,
    precompute_sobol_pool,
    shard_dispatch,
    skip_if_masked,
)
from mu_F.constraints.utils import (
    mask_surrogate,
    pad_to_multiple,
    batch_mask,
    poison_padded,
)
from mu_F.solvers.integer_nlp import (
    IntegerNLPSpec,
    IntegerProblem,
    solve_integer_nlp_batched,
)
from mu_F.solvers.mixed_integer import resolve_integer_spec


__all__ = ["ForwardEvaluator", "forward_constraint_evaluator"]


# Local copy of the multi-head resolver — same rule as in CTG / current /
# backward.  Duplicated here to avoid cross-module imports between sibling
# evaluators.
def _resolve_classifier(cfg, graph, key):
    is_multihead = (
        cfg.samplers.ns.get('rejector', '') == 'sumb-xmeans'
        and graph.nodes[key].get('cluster_classifier_head') is not None
    )
    if is_multihead:
        return (int(graph.nodes[key]['cluster_classifier_n_heads']),
                graph.nodes[key]['cluster_classifier_head'])
    return (0, graph.nodes[key]['classifier'])


# =============================================================================
# ForwardEvaluator — one instance per (cfg, graph, node)
# =============================================================================

class ForwardEvaluator(BaseEvaluator):
    """Stateful forward evaluator.

    One `IntegerNLPSpec` per predecessor, built once in `_build_for_key`.
    Per call, the spec is reused across every theta; JIT cache stays warm.
    """
    _eval_name = 'forward'

    def __init__(self, cfg, graph, node):
        self.specs: dict          = {}
        self.input_indices: dict  = {}
        self.n_y: dict            = {}        # forward surrogate output width
        super().__init__(cfg, graph, node)
        self._thread = self.evaluate

    def _keys(self) -> list:
        if self.node is None:
            return []
        return list(self.graph.predecessors(self.node))

    def _build_for_key(self, pred: int) -> None:
        # ── Geometry: pred's full NLP (design + input + aux) ──
        n_design = int(self.graph.nodes[pred]['n_design_args'])
        n_input  = int(self.graph.nodes[pred]['n_input_args'])
        n_aux    = int(self.graph.graph['n_aux_args'])
        ndim     = n_design + n_input + n_aux

        # Integer design dims: positions within the design vector
        # ([0..n_design-1]), which coincide with their positions in the
        # full vector since design comes first.
        int_dims, _ = resolve_integer_spec(
            self.cfg.case_study.get('design_domain', None)
        )
        int_indices = np.array(int_dims, dtype=int)
        n_int = int(int_indices.size)

        # ── Multi-head classifier resolution ──
        n_heads, classifier = _resolve_classifier(self.cfg, self.graph, pred)

        # ── Forward surrogate: probe to discover output width ──
        forward_surrogate = self.graph.edges[pred, self.node]['forward_surrogate']
        decision_bounds = self.graph.nodes[pred]['extendedDS_bounds'].copy()
        lb_full = jnp.asarray(decision_bounds[0]).reshape(-1)
        ub_full = jnp.asarray(decision_bounds[1]).reshape(-1)
        x_probe = 0.5 * (lb_full + ub_full)
        fwd_probe = forward_surrogate(x_probe.reshape(1, -1)).reshape(-1)
        n_g = int(fwd_probe.size)

        # ── Continuous-only bounds: drop integer slots, keep input + aux ──
        lb = jnp.delete(jnp.asarray(decision_bounds[0]), int_indices, axis=1).reshape(-1)
        ub = jnp.delete(jnp.asarray(decision_bounds[1]), int_indices, axis=1).reshape(-1)
        bounds = [lb, ub]
        n_d_cont = int(lb.size)
        n_params = n_g + n_int + n_heads

        # ── Build masks — both callables share `p_aug`, both pass n_y=n_g ──

        empty_ind = np.empty((0,), dtype=int)
        backoff = jnp.sum(jnp.asarray(self.graph.nodes[pred]['constraint_backoff']))

        masked_clf = mask_surrogate(
            classifier,
            ndim=ndim,
            fix_ind=empty_ind,
            aux_ind=empty_ind,
            int_ind=int_indices,
            n_heads=n_heads,
            n_y=n_g,                                    # align with vector_diff layout
            # aggregator inferred from n_heads
        )
        def objective(x_red, p_aug):
            return masked_clf(x_red, p_aug) + backoff

        masked_fwd = mask_surrogate(
            forward_surrogate,
            ndim=ndim,
            fix_ind=empty_ind,
            aux_ind=empty_ind,
            int_ind=int_indices,
            n_heads=n_heads,
            aggregator='vector_diff',
            n_y=n_g,
        )
        def constraint(x_red, p_aug):
            return masked_fwd(x_red, p_aug)

        # ── Septal factory: equality constraint (lhs = rhs = 0) ──
        eq = jnp.zeros((n_g,))
        factory = build_factory(
            objective, constraint, bounds,
            n_decision=n_d_cont,
            n_params=n_params,
            n_constraints=n_g,
            feasibility_tol=self.feasibility_tol,
            optimality_tol=self.optimality_tol,
            max_iter=self.max_iter,
            constraint_lhs=eq,
            constraint_rhs=eq,
        )
        screener = build_penalty_screener(objective, constraint, self.screen_penalty)
        sobol_pool = precompute_sobol_pool(bounds, n_d_cont, self.n_sobol_screen)

        integer_problem = IntegerProblem.from_cfg(
            design_domain=self.cfg.case_study.get('design_domain', None),
            n_heads=n_heads,
        )

        self.specs[pred] = IntegerNLPSpec(
            integer_problem    = integer_problem,
            continuous_factory = factory,
            screener           = screener,
            sobol_pool         = sobol_pool,
            n_starts           = self.n_starts,
            feasibility_tol    = self.feasibility_tol,
        )
        self.input_indices[pred] = np.array(
            self.graph.edges[pred, self.node]['input_indices'], dtype=int,
        )
        self.n_y[pred] = n_g

    # ------------------------------------------------------------------
    # Shard entry point — pmap target
    # ------------------------------------------------------------------

    def evaluate(self, inputs_s, aux_s, mask_s):
        """Shard body — per-scenario forward feasibility via outer-theta vmap.

        Parameters
        ----------
        inputs_s : (N_uncertainty, N_input_dim)
        aux_s    : (N_uncertainty, N_aux)
        mask_s   : scalar bool — True on real lanes, False on padded lanes

        Returns
        -------
        (evals, viable, converged) : each shape (N_uncertainty, N_predecessors).
                                     Padded lanes skip the SQP via `lax.cond`.
                                     `viable` + `converged` feed the
                                     base-class counters at the entry.
        """
        def real():
            evals, viable, converged = [], [], []
            for pred in self._keys():
                n_y = self.n_y[pred]
                # Per-theta target: pred's input slice from inputs_s, plus aux.
                pred_inputs = inputs_s[:, self.input_indices[pred]]   # (n_theta, len(idx))
                target_raw  = jnp.concatenate([pred_inputs, aux_s], axis=-1)
                n_raw = target_raw.shape[-1]

                # Pad / truncate to n_y per theta — same flexibility the
                # legacy preserved (forward surrogate output dim may differ
                # from input_slice + aux width).
                if n_raw >= n_y:
                    ys = target_raw[:, :n_y]
                else:
                    ys = jnp.zeros((target_raw.shape[0], n_y)).at[:, :n_raw].set(target_raw)

                # ys : (n_theta, n_y) — module-level batched solver, one
                # cached compiled program per spec.
                per_theta = solve_integer_nlp_batched(self.specs[pred], ys)
                evals.append(per_theta.objective.reshape(-1, 1))
                viable.append(per_theta.success.reshape(-1, 1))
                converged.append(per_theta.kkt_converged.reshape(-1, 1))
            return jnp.hstack(evals), jnp.hstack(viable), jnp.hstack(converged)

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
    """Top-level forward evaluator — fixed-width pmap with padding + mask.

    Parameters
    ----------
    inputs : (N_batch, N_uncertainty, N_input_dim)
    aux    : (N_batch, N_aux)

    Returns
    -------
    evaluations : (N_batch, N_uncertainty, N_predecessors) — NaN on padded rows.
                  SQP convergence flags are not returned; they're recorded
                  internally via `BaseEvaluator._record_sqp_outcome`.
    """
    evaluator = _get_evaluator(cfg, graph, node)
    if evaluator._keys() == []:
        return jnp.zeros((inputs.shape[0], 1))

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

    devs = cpu_devs[:W]
    pmap_fn = cached_parallel_thread(
        evaluator,
        '_pmap_cache',
        evaluator._thread,
        in_axes=(0, 0, 0),
        devices=devs,
        use_vmap=bool(getattr(cfg.solvers, "use_vmap", False)),
    )
    full_evals, full_viable, full_conv = shard_dispatch(
        pmap_fn, (padded_in, padded_aux, mask), W=W,
    )
    full_evals  = poison_padded(full_evals,  mask, fill=jnp.nan)
    full_viable = poison_padded(full_viable, mask, fill=False)
    full_conv   = poison_padded(full_conv,   mask, fill=False)
    evaluator._record_sqp_outcome(
        viable_flags=full_viable[:n_real],
        converged_flags=full_conv[:n_real],
        node_label=f"forward node={node}",
    )
    return full_evals[:n_real]
