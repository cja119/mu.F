"""Cost-to-go (CTG) evaluator: per-successor constrained CTG minimisation."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mu_F.constraints.evaluators.base import (
    BaseEvaluator,
    sampled_tail,
    theta_indices,
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


__all__ = ["CTGEvaluator", "cost_to_go_evaluator"]


# ---------------------------------------------------------------------------
# CTG evaluator — one instance per (cfg, graph, node)
# ---------------------------------------------------------------------------

class CTGEvaluator(BaseEvaluator):
    """Stateful CTG evaluator.

    Holds one IntegerNLPSpec per successor; the per-call hot loop solves
    each spec across the per-theta upstream inputs, with JAX collapsing the
    outer theta vmap and the inner assignment/start vmaps into one trace.

    """

    _eval_name = 'cost_to_go'

    # ---- External Methods ----

    def __init__(self, cfg, graph, node):
        self.specs: dict          = {}
        self.input_indices: dict  = {}
        self.aux_indices: dict    = {}
        super().__init__(cfg, graph, node)
        self._thread = self.evaluate

    # ---- Private Methods ----

    def _keys(self) -> list:
        """Sub-problem keys: the successors of this node."""
        return list(self.graph.successors(self.node))

    def _build_for_key(self, succ: int) -> None:
        """Build the per-successor constrained-CTG IntegerNLPSpec."""
        # Geometry: successor NLP shape and which slots are fix / aux / int.
        n_d_succ = int(self.graph.nodes[succ]['n_design_args'])
        input_indices = np.array(
            [n_d_succ + inp for inp in self.graph.edges[self.node, succ]['input_indices']],
            dtype=int,
        )
        # Pin the successor's whole aux block, not just what this edge couples: an
        # uncoupled local_param has no edge seat but is still given, not chosen.
        n_in_succ = int(self.graph.nodes[succ]['n_input_args'])
        n_aux     = int(self.graph.graph['n_aux_args'])
        aux_indices = np.arange(n_d_succ + n_in_succ,
                                n_d_succ + n_in_succ + n_aux).astype(int)
        fix_indices = np.hstack([input_indices, aux_indices,
                                 theta_indices(self.graph, n_d_succ, n_in_succ)]).astype(int)
        # pinned but not a coupling input: derived from fix_indices so they cannot diverge
        pinned_tail = np.setdiff1d(fix_indices, input_indices).astype(int)
        ndim = (
            n_d_succ
            + int(self.graph.nodes[succ]['n_input_args'])
            + sum(sampled_tail(self.graph))
        )
        n_fix = int(len(fix_indices))

        int_dims, int_values = resolve_integer_spec(
            self.cfg.case_study.get('design_domain', None)
        )
        int_indices = np.array(int_dims, dtype=int)
        n_int = int(int_indices.size)

        # Continuous-only bounds: drop fix/aux/int slots.
        decision_bounds = self.graph.nodes[succ]['extendedDS_bounds'].copy()
        drop = np.concatenate([fix_indices, int_indices]).astype(int)
        lb = jnp.delete(jnp.asarray(decision_bounds[0]), drop, axis=1).reshape(-1)
        ub = jnp.delete(jnp.asarray(decision_bounds[1]), drop, axis=1).reshape(-1)
        bounds = [lb, ub]
        n_d_cont = int(lb.size)

        # Objective: successor's CTG Surrogate (always scalar regression).
        objective = mask_surrogate(
            self.graph.nodes[succ]['ctg_surrogate'],
            ndim=ndim,
            fix_ind=input_indices,
            aux_ind=pinned_tail,
            int_ind=int_indices,
            n_heads=0,
            aggregator='scalar',
        )

        # Feasible region: classifier, or chance constraint under direct-probability.
        constraint, n_heads = self._feasibility_constraint(
            succ, ndim, input_indices, pinned_tail, int_indices,
        )
        n_params = n_fix + n_int + n_heads
        assert n_d_cont == ndim - len(fix_indices) - n_int, (
            f"layout: {n_d_cont} free slots, ndim {ndim} less {len(fix_indices)} pinned "
            f"and {n_int} integer — a pinned slot has leaked into the free set")

        # Septal factory + screener + Sobol pool.
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

        # Integer problem (SOS1 head selector when n_heads > 0).
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
            backend            = self.integer_backend,
            bb_max_nodes       = self.bb_max_nodes,
        )
        self.input_indices[succ] = input_indices
        self.aux_indices[succ]   = aux_indices

    def _feasibility_constraint(self, succ, ndim, input_indices, aux_indices, int_indices):
        """Feasible-region constraint for the successor's CTG optimisation."""
        direct_prob = (bool(self.cfg.samplers.deus.get('direct_probability', False))
                       and self.cfg.formulation == 'probabilistic')
        if direct_prob:
            return self._chance_constraint(succ, ndim, input_indices, aux_indices, int_indices)
        return self._classifier_constraint(succ, ndim, input_indices, aux_indices, int_indices)

    def _chance_constraint(self, succ, ndim, input_indices, aux_indices, int_indices):
        """Chance constraint `P_feas_succ(x) >= p_target` from the successor's
        probability_map.  The factory enforces `g <= 0`, so this is encoded as
        `g = p_target - P`.
        """
        uw = self.cfg.samplers.unit_wise_target_reliability
        try:
            p_target = float(uw[succ])
        except (TypeError, KeyError, IndexError):
            p_target = float(uw)
        masked = mask_surrogate(
            self.graph.nodes[succ]['probability_map'],
            ndim=ndim, fix_ind=input_indices, aux_ind=aux_indices,
            int_ind=int_indices, n_heads=0, aggregator='scalar',
        )
        def constraint(x_red, p_aug):
            return jnp.atleast_1d(p_target - masked(x_red, p_aug))
        return constraint, 0

    def _classifier_constraint(self, succ, ndim, input_indices, aux_indices, int_indices):
        """Successor feasibility classifier (multi-head aware)."""
        n_heads, classifier = _resolve_classifier(self.cfg, self.graph, succ)
        masked = mask_surrogate(
            classifier, ndim=ndim, fix_ind=input_indices, aux_ind=aux_indices,
            int_ind=int_indices, n_heads=n_heads,
        )
        def constraint(x_red, p_aug):
            return jnp.atleast_1d(masked(x_red, p_aug))
        return constraint, n_heads

    def _nominal_theta(self, n):
        """
        Theta at its best estimate, tiled over the batch; empty unless theta rides
        the design space. The cost-to-go is never worst-cased over it.
        """
        _, n_theta = sampled_tail(self.graph)
        if not n_theta:
            return jnp.empty((n, 0))

        pbe = self.cfg.case_study.parameters_best_estimate[self.node]
        return jnp.tile(jnp.asarray(pbe, dtype=float).reshape(1, -1), (n, 1))

    # Shard entry point — pmap target.
    def evaluate(self, outputs_s, aux_s, mask_s):
        """
        Shard body: per-scenario CTG via outer-theta vmap and integer-NLP
        solve.  Returns per-successor evals, viability (for the Bellman
        NaN-mask) and KKT-convergence flags.
        """
        def real():
            succ_inputs = get_successor_inputs(self.graph, self.node, outputs_s)
            evals, viable, converged = [], [], []
            for succ in self._keys():
                ys = succ_inputs[succ]
                if aux_s is not None and aux_s.size > 0:
                    ys = jnp.concatenate([ys, aux_s], axis=-1)
                # the cost-to-go is read at nominal theta, never worst-cased
                ys = jnp.concatenate([ys, self._nominal_theta(ys.shape[0])], axis=-1)
                # ys is (n_theta, n_y); module-level solver caches one program per spec.
                per_theta = solve_integer_nlp_batched(self.specs[succ], ys)
                evals.append(per_theta.objective.reshape(-1, 1))
                viable.append(per_theta.success.reshape(-1, 1))
                converged.append(per_theta.kkt_converged.reshape(-1, 1))
            return jnp.hstack(evals), jnp.hstack(viable), jnp.hstack(converged)

        return skip_if_masked(mask_s, real)


# Local copy of the resolver (canonical in current.py) — duplicated to avoid an import cycle.
def _resolve_classifier(cfg, graph, key):
    is_multihead = (
        cfg.samplers.ns.get('rejector', '') == 'sumb-xmeans'
        and graph.nodes[key].get('cluster_classifier_head') is not None
    )
    if is_multihead:
        return (int(graph.nodes[key]['cluster_classifier_n_heads']),
                graph.nodes[key]['cluster_classifier_head'])
    return (0, graph.nodes[key]['classifier'])


# ---------------------------------------------------------------------------
# Evaluator cache — one CTGEvaluator per (id(graph), node)
# ---------------------------------------------------------------------------

_CTG_EVALUATOR_CACHE: dict = {}


def _get_evaluator(cfg, graph, node) -> CTGEvaluator:
    """Return the cached CTGEvaluator for this (graph, node)."""
    key = (id(graph), node)
    evaluator = _CTG_EVALUATOR_CACHE.get(key)
    if evaluator is None:
        evaluator = CTGEvaluator(cfg, graph, node)
        _CTG_EVALUATOR_CACHE[key] = evaluator
    return evaluator


# ---------------------------------------------------------------------------
# Public entry point — called by constraints/constructor.py
# ---------------------------------------------------------------------------

def cost_to_go_evaluator(outputs, aux, cfg, graph, node):
    """Top-level CTG evaluator with fixed-width pmap sharding."""
    evaluator = _get_evaluator(cfg, graph, node)

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
        node_label=f"CTG node={node}",
    )
    # Caller uses the viable flag to NaN-mask the Bellman input; convergence is diagnostic only.
    return full_evals[:n_real], full_viable[:n_real]
