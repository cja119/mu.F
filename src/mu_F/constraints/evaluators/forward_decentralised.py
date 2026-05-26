"""
Decentralised forward coupling constraint evaluator.

Single joint NLP per node — every predecessor's decision vector is
concatenated into one big `x`:

    x_concat = concat([x_pred_1, x_pred_2, ...])

    min   -classifier_node(hstack([v, fwd_1(x_1), fwd_2(x_2), ...])) - node_backoff
              where v   = this sample's measured input vector (parametric `p`)
                    fwd_i = pred_i's forward surrogate applied to x's slice for pred i

    s.t.  classifier_pred_i(x_i) + backoff_i  <=  0   for every predecessor
          lb_concat <= x_concat <= ub_concat

Contrasts with `forward.py`:

  - `forward.py` solves one **independent** NLP per predecessor, each with a
    hard equality `forward_surrogate_i(x_i) = target_i`.
  - This file solves one **joint** NLP over all predecessors together.  The
    node classifier appears as the objective (evaluated on stacked surrogate
    outputs alongside `v`).  Per-pred classifiers are inequality constraints.

### Joint integer problem

When integer design dims live in `cfg.case_study.design_domain` (one
position list shared across nodes) or when one or more classifiers are
K-head (`rejector='sumb-xmeans'`), `IntegerProblem` is built **jointly**:

  - `design_dims`  — one `DesignIntegerDim` per (pred, integer dim).  Slots
    are global positions in the concatenated full vector (pred offset +
    local dim position).
  - `structural_domains` — concatenation of every pred's K-head binaries,
    then the node's K-head binaries (if multi-head).
  - `sos1_groups`        — one SOS1 group per multi-head classifier
    (preds first, node last).
  - `linear_constraints` — one Σ=1 per multi-head classifier.

`feasible_assignments()` enumerates the Cartesian product of every per-pred
integer × every per-pred K-head × the node's K-head; SOS1 filtering keeps
only the K^n_classifiers one-hot rows.

### Parametric tail layout

`p_aug` is shared between objective and constraint:

    p_aug = [ v (n_y_node) | int_values (n_int_total) | head_onehots (n_heads_total) ]

  - `v`             — node's measured input for this sample (the target).
  - `int_values`    — integer design values, concatenated across preds in
                      iteration order.
  - `head_onehots`  — head selector one-hots, concatenated across preds
                      then the node.

The objective and constraint hand-roll their reconstruction of each pred's
full vector instead of going through `mask_surrogate` — the node-side
hstacking of surrogate outputs is too custom to fit `mask_surrogate`'s
single-callable shape, and constraint-side requires per-pred slicing of
the concatenated `x_red`.

### Evaluate path

`evaluate(inputs, aux)` vmaps `solve_integer_nlp` over the sample axis —
each sample gets its own `v`, the spec is shared.  `aux` is unused (each
pred's aux slots live inside its own slice of the concat decision vector).
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
    """Concatenated per-pred design integers + per-classifier K-head SOS1 selectors.

    Parameters
    ----------
    design_domain         : `cfg.case_study.design_domain` list (per-position
                            entries: `'real'` or a tuple of allowed ints).
    pred_offsets_in_concat: start position of each pred's full slice in
                            the concatenated decision vector.  Length n_preds.
    pred_n_heads          : K per predecessor (0 = single-head).  Length n_preds.
    n_heads_node          : K for the node classifier (0 = single-head).

    Layout in `feasible_assignments()` rows:

        [ pred_0_int_dim_0, pred_0_int_dim_1, …,
          pred_1_int_dim_0, …,
          | pred_0_head_0, … pred_0_head_K0,
          pred_1_head_0, …,
          node_head_0, … node_head_Kn ]

    Each SOS1 group + Σ=1 constraint is *local* to one classifier — the
    coeff vector zeroes out other classifiers' binaries.
    """
    # ── Design integers ──
    design_dims = []
    int_dims_local, _ = resolve_integer_spec(design_domain)
    for offset in pred_offsets_in_concat:
        for d_local in int_dims_local:
            design_dims.append(DesignIntegerDim(
                slot=int(offset) + int(d_local),
                domain=tuple(design_domain[d_local]),
            ))

    # ── Structural (K-head) selectors: preds first, node last ──
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


# =============================================================================
# ForwardDecentralisedEvaluator — one instance per (cfg, graph, node)
# =============================================================================

class ForwardDecentralisedEvaluator(BaseEvaluator):
    """Stateful decentralised forward evaluator.

    Single sub-problem per node (`_keys() = [node]`).  The sample-dependent
    measured-input vector `v` is threaded as the parametric tail's leading
    `n_y_node` slots; integer values and head one-hots fill the rest of
    `p_aug`.
    """

    def __init__(self, cfg, graph, node):
        self.specs: dict = {}
        self.n_y: dict   = {}                                # = n_y_node per key
        super().__init__(cfg, graph, node)
        self._thread = self.evaluate

    def _keys(self) -> list:
        if self.node is None or not list(self.graph.predecessors(self.node)):
            return []
        return [self.node]

    def _build_for_key(self, key) -> None:
        preds = list(self.graph.predecessors(key))
        n_preds = len(preds)
        n_aux = int(self.graph.graph['n_aux_args'])
        design_domain = self.cfg.case_study.get('design_domain', None)

        # ── Per-pred geometry ──
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

        # ── Multi-head resolution per pred + node ──
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

        # ── Forward surrogates + backoffs ──
        pred_forward_sg = [self.graph.edges[p, key]['forward_surrogate'] for p in preds]
        pred_backoffs = [
            jnp.sum(jnp.asarray(self.graph.nodes[p]['constraint_backoff']))
            for p in preds
        ]
        node_backoff = jnp.sum(jnp.asarray(self.graph.nodes[key]['constraint_backoff']))

        # ── Continuous-only concat bounds (drop integer slots per pred) ──
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

        # ── Probe forward surrogate output widths ──
        def _fwd_width(i):
            decision_bounds = self.graph.nodes[preds[i]]['extendedDS_bounds'].copy()
            lb_full = jnp.asarray(decision_bounds[0]).reshape(-1)
            ub_full = jnp.asarray(decision_bounds[1]).reshape(-1)
            x_probe = 0.5 * (lb_full + ub_full)
            return int(pred_forward_sg[i](x_probe.reshape(1, -1)).reshape(-1).size)
        fwd_widths = [_fwd_width(i) for i in range(n_preds)]
        n_y_node = int(sum(fwd_widths))                     # parametric tail leading-slice width
        n_params = n_y_node + n_int_total + n_heads_total

        # ── Closure helpers ──
        # Hoist Python lists into JAX-friendly form once at build time.
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
        )
        self.n_y[key] = n_y_node

    # ------------------------------------------------------------------
    # Evaluate — vmap over samples
    # ------------------------------------------------------------------

    def evaluate(self, inputs, aux):
        """Solve the joint decentralised NLP for every sample.

        Returns `evaluations` of shape `(N_samples, 1)` — `classifier_node(x*)
        + node_backoff` (positive = feasible — sign flipped at the return,
        matching legacy).  SQP convergence flags are not returned; they're
        recorded internally via `BaseEvaluator._record_sqp_outcome`.
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

        # Module-level batched solver, one cached compiled program per
        # spec.  Per-sample (not per-theta) here but the vmap structure
        # is identical — outer batch axis on ys, spec held static.
        per_sample = solve_integer_nlp_batched(self.specs[key], ys)
        self._record_sqp_outcome(
            viable_flags=per_sample.success,
            converged_flags=per_sample.kkt_converged,
            node_label=f"forward_dec node={self.node}",
        )
        # objective stored as `-classifier - backoff` (minimised);
        # negate at return so positive = feasible (legacy convention).
        return (-per_sample.objective).reshape(-1, 1)


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
    """Drop-in replacement for the legacy entry point.

    Sample-batched via `jax.vmap` inside `evaluate`; no pmap layer (this
    evaluator is typically called on modest batches — add pmap if profiling
    shows the inner solve is the hot path).
    """
    return _get_evaluator(cfg, graph, node).evaluate(inputs, aux)
