"""
Cost-to-go (CTG) evaluator.

For each successor of `node`, builds an `IntegerNLPSpec` that wraps:

  • objective : the successor's CTG surrogate, wrapped by `mask_surrogate`
    with `aggregator='scalar'` (CTG is always single-output).
  • constraint: the successor's classifier, wrapped by `mask_surrogate` with
    `aggregator='scalar'` (single-head) or `'onehot_sum'` (K-head SOS1).
  • integer problem: from `cfg.case_study.design_domain` plus K-head SOS1.
  • Sobol pool, screener: standard per-successor warm-start machinery.

At evaluate-time the theta axis is handled by an outer `vmap` over the
per-scenario upstream input `ys`. JAX collapses this with the inner
(assignment × start) vmaps inside `solve_integer_nlp` — which itself
calls septal's JAX-pure single-instance core directly — into one fused
XLA program. First call per node compiles once; subsequent calls hit
the cache.
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


__all__ = ["CTGEvaluator", "cost_to_go_evaluator"]


# =============================================================================
# CTG evaluator — one instance per (cfg, graph, node)
# =============================================================================

class CTGEvaluator(BaseEvaluator):
    """Stateful CTG evaluator.

    Holds an `IntegerNLPSpec` per successor. The per-call hot loop runs:

        per_theta = vmap(solve_integer_nlp, in_axes=(None, 0))(spec, ys)

    where `ys` is the per-theta upstream input vector for this successor.
    JAX collapses the outer theta vmap with the (assignment × start)
    vmaps inside `solve_integer_nlp` into one trace.
    """

    def __init__(self, cfg, graph, node):
        self.specs: dict          = {}
        self.input_indices: dict  = {}
        self.aux_indices: dict    = {}
        super().__init__(cfg, graph, node)
        self._thread = self.evaluate

    def _keys(self) -> list:
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

        # ── Multi-head resolution ──
        n_heads, classifier = _resolve_classifier(self.cfg, self.graph, succ)

        # ── Continuous-only bounds: drop fix/aux/int slots ──
        decision_bounds = self.graph.nodes[succ]['extendedDS_bounds'].copy()
        drop = np.concatenate([fix_indices, int_indices]).astype(int)
        lb = jnp.delete(jnp.asarray(decision_bounds[0]), drop, axis=1).reshape(-1)
        ub = jnp.delete(jnp.asarray(decision_bounds[1]), drop, axis=1).reshape(-1)
        bounds = [lb, ub]
        n_d_cont = int(lb.size)
        n_params = n_fix + n_int + n_heads

        # ── Surrogates wrapped via mask_surrogate ──
        ctg_surrogate = self.graph.nodes[succ]['ctg_surrogate']
        objective = mask_surrogate(
            ctg_surrogate,
            ndim=ndim,
            fix_ind=input_indices,
            aux_ind=aux_indices,
            int_ind=int_indices,
            n_heads=0,                              # CTG always scalar regression
            aggregator='scalar',
        )
        masked_clf = mask_surrogate(
            classifier,
            ndim=ndim,
            fix_ind=input_indices,
            aux_ind=aux_indices,
            int_ind=int_indices,
            n_heads=n_heads,
            # aggregator inferred from n_heads
        )
        # mask_surrogate's 'scalar' / 'onehot_sum' aggregators return a
        # true scalar `()`; septal's lagrangian_grad expects the constraint
        # function to return shape `(n_constraints,) = (1,)` so the
        # jacobian comes out (1, n_d) and `jac_g.T @ lam` aligns.
        def constraint(x_red, p_aug):
            return jnp.atleast_1d(masked_clf(x_red, p_aug))

        # ── Septal factory + screener + Sobol pool ──
        factory = build_factory(
            objective, constraint, bounds,
            n_decision=n_d_cont,
            n_params=n_params,
            n_constraints=1,
            feasibility_tol=self.feasibility_tol,
            optimality_tol=self.optimality_tol,
            max_iter=self.max_iter,
        )
        screener = build_penalty_screener(objective, constraint, self.screen_penalty)
        sobol_pool = precompute_sobol_pool(bounds, n_d_cont, self.n_sobol_screen)

        # ── Integer problem (SOS1 head selector when n_heads > 0) ──
        integer_problem = IntegerProblem.from_cfg(
            design_domain=self.cfg.case_study.get('design_domain', None),
            n_heads=n_heads,
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
        """Shard body — per-scenario CTG via outer-theta vmap + integer-NLP solve.

        Parameters
        ----------
        outputs_s : (N_uncertainty, N_output_dim)
        aux_s     : (N_uncertainty, N_aux)
        mask_s    : scalar bool — True on real lanes, False on padded lanes

        Returns
        -------
        (evals, viable, converged) : each shape (N_uncertainty, N_successors).
                                     `viable` is returned to the caller for
                                     the CTG Bellman NaN-mask; `converged`
                                     feeds the base-class diagnostic counter
                                     at the entry function only.
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
                evals.append(per_theta.objective.reshape(-1, 1))
                viable.append(per_theta.success.reshape(-1, 1))
                converged.append(per_theta.kkt_converged.reshape(-1, 1))
            return jnp.hstack(evals), jnp.hstack(viable), jnp.hstack(converged)

        return skip_if_masked(mask_s, real)


# Local copy of the resolver — see current.py for the canonical version.
# Duplicated to avoid an import cycle between current.py and cost_to_go.py.
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
# Evaluator cache — one CTGEvaluator per (id(graph), node)
# =============================================================================

_CTG_EVALUATOR_CACHE: dict = {}


def _get_evaluator(cfg, graph, node) -> CTGEvaluator:
    key = (id(graph), node)
    evaluator = _CTG_EVALUATOR_CACHE.get(key)
    if evaluator is None:
        evaluator = CTGEvaluator(cfg, graph, node)
        _CTG_EVALUATOR_CACHE[key] = evaluator
    return evaluator


# =============================================================================
# Public entry point — called by constraints/constructor.py
# =============================================================================

def cost_to_go_evaluator(outputs, aux, cfg, graph, node):
    """Top-level CTG evaluator with fixed-width pmap sharding."""
    evaluator = _get_evaluator(cfg, graph, node)

    cpu_devs = list(devices('cpu'))
    W = min(int(cfg.max_devices), len(cpu_devs))
    n_real = outputs.shape[0]

    aux_expanded = jnp.repeat(
        jnp.expand_dims(aux, axis=1), outputs.shape[1], axis=1,
    )
    padded_out, _, _ = pad_to_multiple(outputs, W, axis=0)
    padded_aux, _, _ = pad_to_multiple(aux_expanded, W, axis=0)
    total = padded_out.shape[0]
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
        pmap_fn, (padded_out, padded_aux, mask), W=W,
    )
    full_evals  = poison_padded(full_evals,  mask, fill=jnp.nan)
    full_viable = poison_padded(full_viable, mask, fill=False)
    full_conv   = poison_padded(full_conv,   mask, fill=False)
    evaluator._record_sqp_outcome(
        viable_flags=full_viable[:n_real],
        converged_flags=full_conv[:n_real],
        node_label=f"CTG node={node}",
    )
    # Caller (integration.py) uses the viable flag to NaN-mask the Bellman
    # input.  KKT-convergence is recorded for the diagnostic but not
    # surfaced — an unconverged-but-feasible CTG value is still usable.
    return full_evals[:n_real], full_viable[:n_real]
