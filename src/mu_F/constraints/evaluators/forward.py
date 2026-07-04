"""Forward coupling constraint evaluator: per-predecessor equality-matched NLP."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mu_F._types import typecheck, InputBatch, AuxBatch
from mu_F.constraints.evaluators.base import (
    BaseEvaluator,
    build_factory,
    build_penalty_screener,
    precompute_sobol_pool,
    shard_dispatch,
    skip_if_masked,
)
from mu_F.constraints.utils import (
    mask_aux,
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


# Local copy of the multi-head resolver, duplicated to avoid sibling imports.
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
# ForwardEvaluator — one instance per (cfg, graph, node)
# ---------------------------------------------------------------------------

class ForwardEvaluator(BaseEvaluator):
    """Stateful forward evaluator.

    One `IntegerNLPSpec` per predecessor, built once in `_build_for_key`.
    Per call, the spec is reused across every theta; JIT cache stays warm.

    """

    # ---- External Methods ----

    _eval_name = 'forward'

    def __init__(self, cfg, graph, node):
        self.specs: dict          = {}
        self.input_indices: dict  = {}
        self.n_y: dict            = {}        # forward Surrogate output width
        super().__init__(cfg, graph, node)
        self._thread = self.evaluate

    # ---- Private Methods ----

    def _keys(self) -> list:
        """Predecessors of this node; one NLP spec is built per entry."""
        if self.node is None:
            return []
        return list(self.graph.predecessors(self.node))

    def _build_for_key(self, pred: int) -> None:
        """Build the per-predecessor equality-matched integer NLP spec."""
        # Geometry: pred's full NLP (design + input + aux).
        n_design = int(self.graph.nodes[pred]['n_design_args'])
        n_input  = int(self.graph.nodes[pred]['n_input_args'])
        n_aux    = int(self.graph.graph['n_aux_args'])
        ndim     = n_design + n_input + n_aux

        # Integer design dims: positions within the design vector (design comes first).
        int_dims, _ = resolve_integer_spec(
            self.cfg.case_study.get('design_domain', None)
        )
        int_indices = np.array(int_dims, dtype=int)
        n_int = int(int_indices.size)

        # Multi-head classifier resolution.
        n_heads, classifier = _resolve_classifier(self.cfg, self.graph, pred)

        # Forward Surrogate: probe to discover output width.
        forward_surrogate = self.graph.edges[pred, self.node]['forward_surrogate']
        decision_bounds = self.graph.nodes[pred]['extendedDS_bounds'].copy()
        lb_full = jnp.asarray(decision_bounds[0]).reshape(-1)
        ub_full = jnp.asarray(decision_bounds[1]).reshape(-1)
        x_probe = 0.5 * (lb_full + ub_full)
        fwd_probe = forward_surrogate(x_probe.reshape(1, -1)).reshape(-1)
        n_g = int(fwd_probe.size)

        # Pred's shared aux occupies the trailing block of its decision vector.
        aux_indices = np.arange(n_design + n_input, n_design + n_input + n_aux).astype(int)

        # Free continuous bounds: drop integer slots and the pinned aux slots.
        drop = np.concatenate([int_indices, aux_indices]).astype(int)
        lb = jnp.delete(jnp.asarray(decision_bounds[0]), drop, axis=1).reshape(-1)
        ub = jnp.delete(jnp.asarray(decision_bounds[1]), drop, axis=1).reshape(-1)
        bounds = [lb, ub]
        n_d_cont = int(lb.size)
        n_params = n_g + n_aux + n_int + n_heads

        backoff = jnp.sum(jnp.asarray(self.graph.nodes[pred]['constraint_backoff']))

        masked_clf = mask_aux(
            classifier,
            ndim=ndim,
            aux_ind=aux_indices,
            int_ind=int_indices,
            n_heads=n_heads,
            n_g=n_g,                                    # aggregator inferred from n_heads
        )
        def objective(x_red, p_aug):
            return masked_clf(x_red, p_aug) + backoff

        masked_fwd = mask_aux(
            forward_surrogate,
            ndim=ndim,
            aux_ind=aux_indices,
            int_ind=int_indices,
            n_heads=n_heads,
            aggregator='vector_diff',
            n_g=n_g,
        )
        def constraint(x_red, p_aug):
            return masked_fwd(x_red, p_aug)

        # Septal factory: equality constraint (lhs = rhs = 0).
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
            backend            = self.integer_backend,
            bb_max_nodes       = self.bb_max_nodes,
        )
        self.input_indices[pred] = np.array(
            self.graph.edges[pred, self.node]['input_indices'], dtype=int,
        )
        self.n_y[pred] = n_g

    # ---- External Methods ----

    def evaluate(self, inputs_s, aux_s, mask_s):
        """
        Shard body (pmap target): per-scenario forward feasibility via an
        outer-theta vmap over the uncertainty axis. Padded lanes skip the SQP.
        """
        def real():
            evals, viable, converged = [], [], []
            for pred in self._keys():
                # Per-theta tail: inputs matched at the output, aux pinned at the input.
                pred_inputs = inputs_s[:, self.input_indices[pred]]   # (n_theta, n_g)
                ys = jnp.concatenate([pred_inputs, aux_s], axis=-1)   # (n_theta, n_g + n_aux)

                per_theta = solve_integer_nlp_batched(self.specs[pred], ys)
                evals.append(per_theta.objective.reshape(-1, 1))
                viable.append(per_theta.success.reshape(-1, 1))
                converged.append(per_theta.kkt_converged.reshape(-1, 1))
            return jnp.hstack(evals), jnp.hstack(viable), jnp.hstack(converged)

        return skip_if_masked(mask_s, real)


# ---------------------------------------------------------------------------
# Evaluator cache + top-level entry point
# ---------------------------------------------------------------------------

_FORWARD_EVALUATOR_CACHE: dict = {}


def _get_evaluator(cfg, graph, node) -> ForwardEvaluator:
    """Cached evaluator lookup keyed on graph id and node."""
    key = (id(graph), node)
    evaluator = _FORWARD_EVALUATOR_CACHE.get(key)
    if evaluator is None:
        evaluator = ForwardEvaluator(cfg, graph, node)
        _FORWARD_EVALUATOR_CACHE[key] = evaluator
    return evaluator


@typecheck
def forward_constraint_evaluator(inputs: InputBatch, aux: AuxBatch, cfg, graph, node):
    """
    Top-level forward evaluator: fixed-width pmap with padding + mask.
    Returns per-(design, scenario, predecessor) evaluations, NaN on padded rows.
    """
    evaluator = _get_evaluator(cfg, graph, node)
    if evaluator._keys() == []:
        return jnp.zeros((inputs.shape[0], 1))

    W, pmap_fn = evaluator._build_dispatch_fn(evaluator._thread, in_axes=(0, 0, 0))
    n_real = inputs.shape[0]

    padded_in,  _, _ = pad_to_multiple(inputs, W, axis=0)
    padded_aux, _, _ = pad_to_multiple(aux, W, axis=0)   # (N, A); one aux set per design
    total = padded_in.shape[0]
    mask = batch_mask(n_real, total)

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
