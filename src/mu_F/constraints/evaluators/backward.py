"""Backward coupling constraint evaluator (box-only successor feasibility check)."""
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
    shaping_function,
)
from mu_F.solvers.integer_nlp import (
    IntegerNLPSpec,
    IntegerProblem,
    solve_integer_nlp_batched,
)
from mu_F.solvers.mixed_integer import resolve_integer_spec


__all__ = ["BackwardEvaluator", "backward_constraint_evaluator"]


# Local copy of the multi-head resolver — duplicated across sibling
# evaluator modules to avoid import cycles.
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
# BackwardEvaluator — one instance per (cfg, graph, node)
# ---------------------------------------------------------------------------

class BackwardEvaluator(BaseEvaluator):
    """Stateful backward feasibility evaluator.

    Builds one IntegerNLPSpec per key (per-successor in node-local mode, a
    single None in graph-wide mode) once in _build_for_key, then reuses each
    spec across every theta so the JIT cache stays warm between calls.

    """

    _eval_name = 'backward'

    # ---- External Methods ----

    def __init__(self, cfg, graph, node):
        self._graph_wide = bool(graph.graph.get('solve_post_processing_problem', False))
        # Per-key static state.
        self.specs: dict          = {}
        self.input_indices: dict  = {}
        self.aux_indices: dict    = {}
        self.fix_indices: dict    = {}
        self.n_fix: dict          = {}
        super().__init__(cfg, graph, node)
        self._thread_node_local = self.evaluate_node_local
        self._thread_graph_wide = self.evaluate_graph_wide

    def evaluate(self, outputs_s, aux_s, mask_s=None):
        if self._graph_wide:
            return self.evaluate_graph_wide(outputs_s, aux_s)
        if mask_s is None:
            mask_s = jnp.asarray(True)
        return self.evaluate_node_local(outputs_s, aux_s, mask_s)

    # ---- Private Methods ----

    def _keys(self) -> list:
        """Sub-problem keys: successors (node-local) or [None] (graph-wide)."""
        if self._graph_wide:
            return [None]
        if self.node is None:
            return []
        return list(self.graph.successors(self.node))

    def _build_for_key(self, key) -> None:
        """Dispatch to the node-local or graph-wide builder."""
        if self._graph_wide:
            self._build_graph_wide(key)
        else:
            self._build_node_local(key)

    # Build — node-local (per-successor).
    def _build_node_local(self, succ: int) -> None:
        """Build the per-successor box-only IntegerNLPSpec for this edge."""
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

        # Multi-head resolution: single-head ANN or K-head cluster classifier.
        n_heads, classifier = _resolve_classifier(self.cfg, self.graph, succ)

        # Continuous-only bounds: drop fix/aux/int slots.
        decision_bounds = self.graph.nodes[succ]['extendedDS_bounds'].copy()
        drop = np.concatenate([fix_indices, int_indices]).astype(int)
        lb = jnp.delete(jnp.asarray(decision_bounds[0]), drop, axis=1).reshape(-1)
        ub = jnp.delete(jnp.asarray(decision_bounds[1]), drop, axis=1).reshape(-1)
        bounds = [lb, ub]
        n_d_cont = int(lb.size)
        n_params = n_fix + n_int + n_heads

        # Classifier is the objective in this box-only NLP; aggregator from n_heads.
        objective = mask_surrogate(
            classifier,
            ndim=ndim,
            fix_ind=input_indices,
            aux_ind=aux_indices,
            int_ind=int_indices,
            n_heads=n_heads,
        )

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
        self.fix_indices[succ]   = fix_indices
        self.n_fix[succ]         = n_fix

    # Build — graph-wide (post-process lower classifier).
    def _build_graph_wide(self, key) -> None:
        """Build the single graph-level IntegerNLPSpec over the decision space."""
        n_d   = int(self.graph.graph['n_design_args'])
        n_aux = int(self.graph.graph['n_aux_args'])
        total_ind = np.arange(n_d + n_aux)
        dec_ind = np.array(self.graph.graph['post_process_decision_indices'])
        fix_ind = np.delete(total_ind, dec_ind).astype(int)

        lb = jnp.hstack([jnp.array(b[0]).reshape(-1)
                         for b in self.graph.graph['bounds'] if b[0] not in (None, 'None')])
        ub = jnp.hstack([jnp.array(b[1]).reshape(-1)
                         for b in self.graph.graph['bounds'] if b[1] not in (None, 'None')])
        lb_red = jnp.delete(lb, fix_ind)
        ub_red = jnp.delete(ub, fix_ind)
        bounds = [lb_red, ub_red]
        ndim = n_d + n_aux
        n_d_k = int(lb_red.size)
        n_fix = int(len(fix_ind))

        # No design integers / heads at graph level — single-head only.
        classifier = self.graph.graph['post_process_lower_classifier']
        masked = mask_surrogate(
            classifier,
            ndim=ndim,
            fix_ind=fix_ind,
            aux_ind=np.empty((0,), dtype=int),
            int_ind=np.empty((0,), dtype=int),
            n_heads=0,
            aggregator='scalar',
        )
        # Maximisation in casadi/legacy convention — negate so the SQP minimises.
        def objective(x_red, p_aug):
            return -masked(x_red, p_aug)

        factory = build_factory(
            objective, None, bounds,
            n_decision=n_d_k,
            n_params=n_fix,
            n_constraints=0,
            feasibility_tol=self.feasibility_tol,
            optimality_tol=self.optimality_tol,
            max_iter=self.max_iter,
        )
        screener = build_penalty_screener(objective, None, self.screen_penalty)
        sobol_pool = precompute_sobol_pool(bounds, n_d_k, self.n_sobol_screen)

        integer_problem = IntegerProblem.from_cfg(design_domain=None, n_heads=0)

        self.specs[key] = IntegerNLPSpec(
            integer_problem    = integer_problem,
            continuous_factory = factory,
            screener           = screener,
            sobol_pool         = sobol_pool,
            n_starts           = self.n_starts,
            feasibility_tol    = self.feasibility_tol,
            backend            = self.integer_backend,
            bb_max_nodes       = self.bb_max_nodes,
        )
        self.fix_indices[key] = fix_ind
        self.n_fix[key]       = n_fix

    # Thread entry points — pmap targets.
    def evaluate_node_local(self, outputs_s, aux_s, mask_s):
        """
        Per-scenario backward feasibility via outer-theta vmap.
        Returns per-successor evals, viability and KKT-convergence flags.
        """
        def real():
            succ_inputs = get_successor_inputs(self.graph, self.node, outputs_s)
            evals, viable, converged = [], [], []
            for succ in self._keys():
                ys = succ_inputs[succ]
                if aux_s is not None and aux_s.size > 0:
                    ys = jnp.concatenate([ys, aux_s], axis=-1)
                # ys is (n_theta, n_y); module-level solver caches one program per spec.
                per_theta = solve_integer_nlp_batched(self.specs[succ], ys)
                evals.append(per_theta.objective.reshape(-1, 1))
                viable.append(per_theta.success.reshape(-1, 1))
                converged.append(per_theta.kkt_converged.reshape(-1, 1))
            return (shaping_function(jnp.hstack(evals), self.cfg),
                    jnp.hstack(viable),
                    jnp.hstack(converged))

        return skip_if_masked(mask_s, real)

    def evaluate_graph_wide(self, outputs_s, aux_s):
        """
        Single graph-wide solve with outputs_s as the graph-level p.
        Inner objective is -classifier; the outer negation restores the
        maximisation sign before the cfg-driven shaping_function flip.
        """
        key = None
        if outputs_s.ndim < 2:
            outputs_s = outputs_s.reshape(-1, 1)
        y_raw = outputs_s.reshape(-1)[: self.n_fix[key]]
        y = jnp.zeros(self.n_fix[key]).at[:y_raw.size].set(y_raw)

        result = solve_integer_nlp(self.specs[key], y)
        return -shaping_function(
            jnp.asarray(result.objective).reshape(-1, 1), self.cfg,
        )


# ---------------------------------------------------------------------------
# Evaluator cache + top-level entry point
# ---------------------------------------------------------------------------

_BACKWARD_EVALUATOR_CACHE: dict = {}


def _get_evaluator(cfg, graph, node) -> BackwardEvaluator:
    """Return the cached BackwardEvaluator for this (graph, node, mode)."""
    key = (id(graph), node,
           bool(graph.graph.get('solve_post_processing_problem', False)))
    evaluator = _BACKWARD_EVALUATOR_CACHE.get(key)
    if evaluator is None:
        evaluator = BackwardEvaluator(cfg, graph, node)
        _BACKWARD_EVALUATOR_CACHE[key] = evaluator
    return evaluator


def backward_constraint_evaluator(outputs, aux, cfg, graph, node):
    """
    Top-level backward feasibility evaluator; threads samples across CPU
    devices via pmap.  Trailing None keeps tuple parity with the unused
    legacy warmstart slot.
    """
    evaluator = _get_evaluator(cfg, graph, node)

    if evaluator._graph_wide:
        thread_out = evaluator.evaluate_graph_wide(outputs, aux)
        n_batch = outputs.shape[0]
        return (
            jnp.broadcast_to(thread_out.reshape(1, -1), (n_batch, thread_out.size)),
            None,
        )

    if evaluator._keys() == []:
        return jnp.zeros((outputs.shape[0], 1)), None

    W, pmap_fn = evaluator._build_dispatch_fn(
        evaluator._thread_node_local, in_axes=(0, 0, 0),
    )
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
        node_label=f"backward node={node}",
    )
    return full_evals[:n_real], None
