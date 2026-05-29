"""
Probabilistic feasibility evaluator.

For each successor of `node`, builds an `IntegerNLPSpec` that maximises
the `probability_map` surrogate over the successor's reduced (continuous-
only) design space, with the upstream output threaded as the parametric
tail `y`.  At evaluate-time the theta axis is handled by an outer `vmap`
over the per-scenario `ys` — composes with the (assignment × start)
vmaps inside `solve_integer_nlp` into one fused XLA program.

The probability map is single-output regression (no multi-head wiring),
so `n_heads = 0` and the aggregator is always `'scalar'`.  Integer
design dims still route through the parametric tail via `IntegerProblem`.

Inner `evaluate` returns `(N_uncertainty, N_successors)`; the top-level
`backward_pmap` shards across the design-batch axis and returns
`(N_batch, N_uncertainty, N_successors)`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mu_F.constraints.evaluators.base import (
    BaseEvaluator,
    build_factory,
    build_penalty_screener,
    precompute_sobol_pool,
    shard_dispatch,
    skip_if_masked,
)
from mu_F.constraints.utils import (
    get_successor_inputs,
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


__all__ = ["BackwardPmapEvaluator", "backward_pmap"]


# =============================================================================
# BackwardPmapEvaluator — one instance per (cfg, graph, node)
# =============================================================================

class BackwardPmapEvaluator(BaseEvaluator):
    """Per-successor probabilistic feasibility evaluator.

    Holds an `IntegerNLPSpec` per successor.  The objective is the masked
    `probability_map` surrogate negated (so minimisation drives P_feas
    up); no general constraint — the surrogate IS the feasibility signal.

    The per-call hot loop runs:

        per_theta = vmap(solve_integer_nlp, in_axes=(None, 0))(spec, ys)

    where `ys` is the per-theta upstream input for this successor.
    """
    _eval_name = 'probability'

    def __init__(self, cfg, graph, node):
        self.specs: dict          = {}
        self.input_indices: dict  = {}
        self.aux_indices: dict    = {}
        super().__init__(cfg, graph, node)
        self._thread = self.evaluate

    def _keys(self) -> list:
        if self.node is None:
            return []
        return list(self.graph.successors(self.node))

    def _build_for_key(self, succ: int) -> None:
        # ── Geometry: successor's full NLP shape + which slots are fix / aux / int ──
        n_d_succ = int(self.graph.nodes[succ]['n_design_args'])
        input_indices = np.array(
            [n_d_succ + inp for inp in self.graph.edges[self.node, succ]['input_indices']],
            dtype=int,
        )
        aux_indices = np.array(
            [n_d_succ + idx for idx in self.graph.edges[self.node, succ]['auxiliary_indices']],
            dtype=int,
        )
        fix_indices = np.hstack([input_indices, aux_indices]).astype(int)
        ndim = (
            n_d_succ
            + int(self.graph.nodes[succ]['n_input_args'])
            + int(self.graph.graph['n_aux_args'])
        )
        n_fix = int(len(fix_indices))

        int_dims, int_values = resolve_integer_spec(
            self.cfg.case_study.get('design_domain', None)
        )
        int_indices = np.array(int_dims, dtype=int)
        n_int = int(int_indices.size)

        # ── Continuous-only bounds: drop fix/aux/int slots ──
        decision_bounds = self.graph.nodes[succ]['extendedDS_bounds'].copy()
        drop = np.concatenate([fix_indices, int_indices]).astype(int)
        lb = jnp.delete(jnp.asarray(decision_bounds[0]), drop, axis=1).reshape(-1)
        ub = jnp.delete(jnp.asarray(decision_bounds[1]), drop, axis=1).reshape(-1)
        bounds = [lb, ub]
        n_d_cont = int(lb.size)
        n_params = n_fix + n_int                                # no head one-hots

        # ── Probability surrogate as objective: minimise `-p_feas` ──
        probability_map = self.graph.nodes[succ]['probability_map']
        masked = mask_surrogate(
            probability_map,
            ndim=ndim,
            fix_ind=input_indices,
            aux_ind=aux_indices,
            int_ind=int_indices,
            n_heads=0,
            aggregator='scalar',
        )
        def objective(x_red, p_aug):
            return -masked(x_red, p_aug)

        # ── Septal factory + screener + Sobol pool (no general constraint) ──
        factory = build_factory(
            objective, None, bounds,
            n_decision=n_d_cont,
            n_params=n_params,
            n_constraints=0,
            feasibility_tol=self.feasibility_tol,
            optimality_tol=self.optimality_tol,
            max_iter=self.max_iter,
        )
        screener = build_penalty_screener(objective, None, self.screen_penalty)
        sobol_pool = precompute_sobol_pool(bounds, n_d_cont, self.n_sobol_screen)

        # ── Integer problem (no head selector — single-output regression) ──
        integer_problem = IntegerProblem.from_cfg(
            design_domain=self.cfg.case_study.get('design_domain', None),
            n_heads=0,
        )

        self.specs[succ] = IntegerNLPSpec(
            integer_problem    = integer_problem,
            continuous_factory = factory,
            screener           = screener,
            sobol_pool         = sobol_pool,
            n_starts           = self.n_starts,
            feasibility_tol    = self.feasibility_tol,
        )
        self.input_indices[succ] = input_indices
        self.aux_indices[succ]   = aux_indices

    # ------------------------------------------------------------------
    # Shard entry point — pmap target
    # ------------------------------------------------------------------

    def evaluate(self, outputs_s, aux_s, mask_s):
        """Shard body — per-scenario probability via outer-theta vmap.

        Parameters
        ----------
        outputs_s : (N_uncertainty, N_output_dim)
        aux_s     : (N_uncertainty, N_aux)
        mask_s    : scalar bool — True on real lanes, False on padded lanes

        Returns
        -------
        (probs, viable, converged) : each shape (N_uncertainty, N_successors).
                                     `probs` = max-achievable P_feas;
                                     `viable` + `converged` feed the
                                     base-class counters at the entry.
        """
        def real():
            succ_inputs = get_successor_inputs(self.graph, self.node, outputs_s)
            evals, viable, converged = [], [], []
            for succ in self._keys():
                ys = succ_inputs[succ]
                if aux_s is not None and aux_s.size > 0:
                    ys = jnp.concatenate([ys, aux_s], axis=-1)
                # ys : (n_theta, n_y) — module-level batched solver, one
                # cached compiled program per spec.
                per_theta = solve_integer_nlp_batched(self.specs[succ], ys)
                # objective = -P_feas  →  negate to recover the probability.
                evals.append((-per_theta.objective).reshape(-1, 1))
                viable.append(per_theta.success.reshape(-1, 1))
                converged.append(per_theta.kkt_converged.reshape(-1, 1))
            return jnp.hstack(evals), jnp.hstack(viable), jnp.hstack(converged)

        return skip_if_masked(mask_s, real)


# =============================================================================
# Evaluator cache + top-level entry point
# =============================================================================

_BACKWARDPMAP_EVALUATOR_CACHE: dict = {}


def _get_pmap_evaluator(cfg, graph, node) -> BackwardPmapEvaluator:
    key = (id(graph), node)
    evaluator = _BACKWARDPMAP_EVALUATOR_CACHE.get(key)
    if evaluator is None:
        evaluator = BackwardPmapEvaluator(cfg, graph, node)
        _BACKWARDPMAP_EVALUATOR_CACHE[key] = evaluator
    return evaluator


def _drive_pmap(evaluator, outputs, aux, cfg, succ_count_fallback: int = 1):
    """Pmap-shard the per-scenario thread across CPU devices.

    Parameters
    ----------
    evaluator : has `_thread(outputs_s, aux_s, mask_s)` returning shape
                `(N_uncertainty, N_succ)`.
    outputs   : (N_batch, N_uncertainty, N_output_dim)
    aux       : (N_batch, N_aux)

    Returns `(N_batch, N_uncertainty, N_succ)`.
    """
    if evaluator._keys() == []:
        # No successors — return ones so the caller's product / sum
        # collapses to a no-op contribution (P=1 picked at call site).
        return jnp.ones((outputs.shape[0], outputs.shape[1], succ_count_fallback))

    W, pmap_fn = evaluator._build_dispatch_fn(evaluator._thread, in_axes=(0, 0, 0))
    n_real = outputs.shape[0]

    aux_expanded = jnp.repeat(
        jnp.expand_dims(aux, axis=1), outputs.shape[1], axis=1,
    )
    padded_out, _, _ = pad_to_multiple(outputs, W, axis=0)
    padded_aux, _, _ = pad_to_multiple(aux_expanded, W, axis=0)
    total = padded_out.shape[0]
    mask = batch_mask(n_real, total)

    full_evals, full_viable, full_conv = shard_dispatch(
        pmap_fn, (padded_out, padded_aux, mask), W=W,
    )
    full_evals  = poison_padded(full_evals,  mask, fill=jnp.nan)
    full_viable = poison_padded(full_viable, mask, fill=False)
    full_conv   = poison_padded(full_conv,   mask, fill=False)
    evaluator._record_sqp_outcome(
        viable_flags=full_viable[:n_real],
        converged_flags=full_conv[:n_real],
        node_label=f"probability node={evaluator.node}",
    )
    return full_evals[:n_real]


def backward_pmap(outputs, aux, cfg, graph, node):
    """Top-level per-scenario downstream-feasibility evaluator.

    Returns
    -------
    (evaluations, None)
        evaluations : (N_batch, N_uncertainty, N_successors) — max P_feas
        per (design, scenario, successor).

    Trailing `None` keeps tuple parity with `backward_constraint_evaluator`.
    """
    evaluator = _get_pmap_evaluator(cfg, graph, node)
    return _drive_pmap(evaluator, outputs, aux, cfg), None
