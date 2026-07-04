"""Decentralised forward evaluator: one joint NLP over all predecessors per node."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from mu_F.constraints.evaluators.base import (
    BaseEvaluator,
    build_factory,
    build_penalty_screener,
    precompute_sobol_pool,
)
from mu_F.solvers.integer_nlp import (
    DesignIntegerDim,
    IntegerNLPSpec,
    IntegerProblem,
    LinearEq,
    solve_integer_nlp_batched,
)
from mu_F.solvers.mixed_integer import resolve_integer_spec


__all__ = [
    "ForwardDecentralisedEvaluator",
    "forward_constraint_decentralised_evaluator",
]


# Local multi-head resolver (same rule as in every other rewired evaluator).
def _resolve_classifier(cfg, graph, key):
    is_multihead = (
        cfg.samplers.ns.get('rejector', '') == 'sumb-xmeans'
        and graph.nodes[key].get('cluster_classifier_head') is not None
    )
    if is_multihead:
        return (int(graph.nodes[key]['cluster_classifier_n_heads']),
                graph.nodes[key]['cluster_classifier_head'])
    return (0, graph.nodes[key]['classifier'])


def _build_joint_integer_problem(design_domain, pred_offsets_in_concat,
                                  pred_n_heads, n_heads_node):
    """
    Build the joint integer problem: concatenated per-pred design integers
    plus one local SOS1 K-head selector group (and Σ=1) per classifier.
    """
    # Design integers.
    design_dims = []
    int_dims_local, _ = resolve_integer_spec(design_domain)
    for offset in pred_offsets_in_concat:
        for d_local in int_dims_local:
            design_dims.append(DesignIntegerDim(
                slot=int(offset) + int(d_local),
                domain=tuple(design_domain[d_local]),
            ))

    # Structural (K-head) selectors: preds first, node last.
    all_n_heads = list(pred_n_heads) + [int(n_heads_node)]
    n_total_struct = int(sum(all_n_heads))

    structural_domains = ()
    sos1_groups = ()
    linear_constraints = ()

    struct_offset = 0
    for n_h in all_n_heads:
        if n_h > 0:
            structural_domains += tuple((0, 1) for _ in range(n_h))
            sos1_groups += (tuple(range(struct_offset, struct_offset + n_h)),)
            coeffs = tuple(
                1.0 if struct_offset <= j < struct_offset + n_h else 0.0
                for j in range(n_total_struct)
            )
            linear_constraints += (LinearEq(coeffs=coeffs, rhs=1.0),)
            struct_offset += n_h

    return IntegerProblem(
        design_dims=tuple(design_dims),
        structural_domains=structural_domains,
        sos1_groups=sos1_groups,
        linear_constraints=linear_constraints,
    )


# ---------------------------------------------------------------------------
# ForwardDecentralisedEvaluator — one instance per (cfg, graph, node)
# ---------------------------------------------------------------------------

class ForwardDecentralisedEvaluator(BaseEvaluator):
    """Stateful decentralised forward evaluator.

    Single sub-problem per node (`_keys() = [node]`).  The sample-dependent
    measured-input vector `v` is threaded as the parametric tail's leading
    `n_y_node` slots; integer values and head one-hots fill the rest of
    `p_aug`.

    """

    # ---- External Methods ----

    _eval_name = 'forward_decentralised'

    def __init__(self, cfg, graph, node):
        self.specs: dict = {}
        self.n_y: dict   = {}                                # = n_y_node per key
        super().__init__(cfg, graph, node)
        self._thread = self.evaluate

    # ---- Private Methods ----

    def _keys(self) -> list:
        """Single joint sub-problem key (the node itself) when it has predecessors."""
        if self.node is None or not list(self.graph.predecessors(self.node)):
            return []
        return [self.node]

    def _build_for_key(self, key) -> None:
        """Build the joint NLP spec over all predecessors' concatenated vectors."""
        preds = list(self.graph.predecessors(key))
        n_preds = len(preds)
        n_aux = int(self.graph.graph['n_aux_args'])
        design_domain = self.cfg.case_study.get('design_domain', None)

        # Per-pred geometry.
        pred_full_dims = [
            int(self.graph.nodes[p]['n_design_args'])
            + int(self.graph.nodes[p]['n_input_args'])
            + n_aux
            for p in preds
        ]
        pred_offsets_in_concat = [0] + list(np.cumsum(pred_full_dims[:-1]))
        pred_offsets_in_concat = [int(o) for o in pred_offsets_in_concat]
        ndim_concat = int(sum(pred_full_dims))

        int_dims_local, _ = resolve_integer_spec(design_domain)
        int_ind_local = np.array(int_dims_local, dtype=int)
        n_int_per_pred = int(int_ind_local.size)
        n_int_total = n_preds * n_int_per_pred

        pred_int_indices = [int_ind_local for _ in preds]                # same per pred
        pred_cont_indices = [
            np.delete(np.arange(d), int_ind_local).astype(int)
            for d in pred_full_dims
        ]
        pred_cont_dims = [len(c) for c in pred_cont_indices]
        cont_slice_starts = [0] + [int(s) for s in np.cumsum(pred_cont_dims[:-1])]
        cont_slice_ends   = [int(e) for e in np.cumsum(pred_cont_dims)]
        n_d_cont = int(sum(pred_cont_dims))

        # Multi-head resolution per pred + node.
        pred_n_heads = []
        pred_classifiers = []
        for p in preds:
            n_h, clf = _resolve_classifier(self.cfg, self.graph, p)
            pred_n_heads.append(n_h)
            pred_classifiers.append(clf)
        n_heads_node, node_classifier = _resolve_classifier(self.cfg, self.graph, key)

        n_heads_total = int(sum(pred_n_heads) + n_heads_node)
        head_offsets  = [0] + [int(o) for o in np.cumsum(pred_n_heads)]
        node_head_offset = int(sum(pred_n_heads))

        # Forward surrogates + backoffs.
        pred_forward_sg = [self.graph.edges[p, key]['forward_surrogate'] for p in preds]
        pred_backoffs = [
            jnp.sum(jnp.asarray(self.graph.nodes[p]['constraint_backoff']))
            for p in preds
        ]
        node_backoff = jnp.sum(jnp.asarray(self.graph.nodes[key]['constraint_backoff']))

        # Continuous-only concat bounds (drop integer slots per pred).
        lb_parts, ub_parts = [], []
        for i, p in enumerate(preds):
            decision_bounds = self.graph.nodes[p]['extendedDS_bounds'].copy()
            lb_full = jnp.asarray(decision_bounds[0]).reshape(-1)
            ub_full = jnp.asarray(decision_bounds[1]).reshape(-1)
            lb_parts.append(jnp.delete(lb_full, int_ind_local))
            ub_parts.append(jnp.delete(ub_full, int_ind_local))
        lb = jnp.concatenate(lb_parts)
        ub = jnp.concatenate(ub_parts)
        bounds = [lb, ub]

        # Probe forward Surrogate output widths.
        def _fwd_width(i):
            decision_bounds = self.graph.nodes[preds[i]]['extendedDS_bounds'].copy()
            lb_full = jnp.asarray(decision_bounds[0]).reshape(-1)
            ub_full = jnp.asarray(decision_bounds[1]).reshape(-1)
            x_probe = 0.5 * (lb_full + ub_full)
            return int(pred_forward_sg[i](x_probe.reshape(1, -1)).reshape(-1).size)
        fwd_widths = [_fwd_width(i) for i in range(n_preds)]
        n_y_node = int(sum(fwd_widths))                     # parametric tail leading-slice width
        n_params = n_y_node + n_int_total + n_heads_total

        # Closure helpers: hoist Python lists into JAX-friendly form once.
        pred_cont_idx_jnp = [jnp.asarray(c) for c in pred_cont_indices]
        pred_int_idx_jnp  = [jnp.asarray(c) for c in pred_int_indices]

        def _unpack_p_aug(p_aug):
            p_aug = jnp.asarray(p_aug)
            if p_aug.ndim == 1:
                p_aug = p_aug.reshape(1, -1)
            v            = p_aug[:, :n_y_node]
            int_values   = p_aug[0, n_y_node : n_y_node + n_int_total]
            head_onehots = p_aug[0, n_y_node + n_int_total
                                  : n_y_node + n_int_total + n_heads_total]
            return v, int_values, head_onehots

        def _reconstruct_pred_full(i, x_red, int_values):
            """Splice x_red's continuous slice and int_values's integer slice
            into pred i's full vector at the right local positions."""
            x_red_slice = x_red[cont_slice_starts[i]:cont_slice_ends[i]]
            x_full = jnp.zeros(pred_full_dims[i])
            x_full = x_full.at[pred_cont_idx_jnp[i]].set(x_red_slice)
            if n_int_per_pred > 0:
                int_start = i * n_int_per_pred
                int_vals_slice = int_values[int_start:int_start + n_int_per_pred]
                x_full = x_full.at[pred_int_idx_jnp[i]].set(int_vals_slice)
            return x_full

        def _aggregate(out, head_onehot, n_h):
            if n_h == 0:
                return out.reshape(())
            return jnp.sum(head_onehot * out)

        def objective(x_red, p_aug):
            v, int_values, head_onehots = _unpack_p_aug(p_aug)
            fwd_outs = [
                pred_forward_sg[i](
                    _reconstruct_pred_full(i, x_red, int_values).reshape(1, -1)
                ).reshape(-1)
                for i in range(n_preds)
            ]
            combined = jnp.concatenate([v.reshape(-1)] + fwd_outs)
            node_out = node_classifier(combined.reshape(1, -1)).reshape(-1)
            node_oh = head_onehots[node_head_offset:node_head_offset + n_heads_node]
            node_val = _aggregate(node_out, node_oh, n_heads_node)
            return -node_val - node_backoff

        def constraint(x_red, p_aug):
            _, int_values, head_onehots = _unpack_p_aug(p_aug)
            pieces = []
            for i in range(n_preds):
                x_pred = _reconstruct_pred_full(i, x_red, int_values)
                pred_out = pred_classifiers[i](x_pred.reshape(1, -1)).reshape(-1)
                pred_oh = head_onehots[head_offsets[i]:head_offsets[i] + pred_n_heads[i]]
                pred_val = _aggregate(pred_out, pred_oh, pred_n_heads[i])
                pieces.append((pred_val + pred_backoffs[i]).reshape(-1))
            return jnp.concatenate(pieces)

        # Probe constraint width for septal's n_constraints
        x_probe_all = 0.5 * (lb + ub)
        p_probe = jnp.zeros(n_params)
        n_g = int(constraint(x_probe_all, p_probe).reshape(-1).shape[0])

        factory = build_factory(
            objective, constraint, bounds,
            n_decision=n_d_cont,
            n_params=n_params,
            n_constraints=n_g,
            feasibility_tol=self.feasibility_tol,
            optimality_tol=self.optimality_tol,
            max_iter=self.max_iter,
        )
        screener = build_penalty_screener(objective, constraint, self.screen_penalty)
        sobol_pool = precompute_sobol_pool(bounds, n_d_cont, self.n_sobol_screen)

        integer_problem = _build_joint_integer_problem(
            design_domain=design_domain,
            pred_offsets_in_concat=pred_offsets_in_concat,
            pred_n_heads=pred_n_heads,
            n_heads_node=n_heads_node,
        )

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
        self.n_y[key] = n_y_node

    # ---- External Methods ----

    def evaluate(self, inputs, aux):
        """
        Solve the joint decentralised NLP for every sample (vmap over samples).
        Returns `(N_samples, 1)`, sign flipped so positive = feasible.
        """
        key = self.node
        if self._keys() == []:
            n = inputs.shape[0]
            return jnp.zeros((n, 1))

        # Per-sample `v`, padded/truncated to n_y.
        inputs2d = jnp.asarray(inputs).reshape(jnp.asarray(inputs).shape[0], -1)
        n_y = self.n_y[key]
        n_raw = inputs2d.shape[-1]
        if n_raw >= n_y:
            ys = inputs2d[:, :n_y]
        else:
            ys = jnp.zeros((inputs2d.shape[0], n_y)).at[:, :n_raw].set(inputs2d)

        # Batched solver, one cached compiled program per spec.
        per_sample = solve_integer_nlp_batched(self.specs[key], ys)
        self._record_sqp_outcome(
            viable_flags=per_sample.success,
            converged_flags=per_sample.kkt_converged,
            node_label=f"forward_dec node={self.node}",
        )
        # objective stored as `-classifier - backoff`; negate so positive = feasible.
        return (-per_sample.objective).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Evaluator cache + top-level entry point
# ---------------------------------------------------------------------------

_FWDDEC_EVALUATOR_CACHE: dict = {}


def _get_evaluator(cfg, graph, node) -> ForwardDecentralisedEvaluator:
    """Cached evaluator lookup keyed on graph id and node."""
    key = (id(graph), node)
    evaluator = _FWDDEC_EVALUATOR_CACHE.get(key)
    if evaluator is None:
        evaluator = ForwardDecentralisedEvaluator(cfg, graph, node)
        _FWDDEC_EVALUATOR_CACHE[key] = evaluator
    return evaluator


def forward_constraint_decentralised_evaluator(inputs, aux, cfg, graph, node):
    """
    Top-level decentralised forward entry point, sample-batched inside
    `evaluate` (no pmap layer).
    """
    return _get_evaluator(cfg, graph, node).evaluate(inputs, aux)
