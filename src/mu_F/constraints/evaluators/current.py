"""
Current-node surrogate evaluators.

Two flavours, both operating on the current node's own live surrogates
(not a successor's / predecessor's):

  - `current_constraint_surrogate`  — box-only feasibility check: minimise
    the (masked) classifier over the reduced decision space.  Used for
    "rollout constraint" checks.

  - `current_cost_surrogate`  — constrained CTG: minimise the node's CTG
    surrogate subject to the node's classifier being feasible.

Both consume the integer-NLP abstraction in `mu_F.solvers.integer_nlp`:
each evaluator builds an `IntegerNLPSpec` once per node in `_build_for_key`
and runs `solve_integer_nlp(spec, y)` per call.  At rollout there is only
one theta — the call returns a single solve result which is reshaped to
`(1, …)` to keep the shape contract identical to the CTG / backward path.

Multi-head classifiers (SOS1 active-head selector) are routed via
`mask_surrogate` with the right aggregator; design integers are routed
via the parametric tail of `p_aug` instead of being SQP decision vars.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from mu_F.constraints.evaluators.base import (
    BaseEvaluator,
    build_factory,
    build_penalty_screener,
    precompute_sobol_pool,
)
from mu_F.constraints.utils import mask_surrogate
from mu_F.solvers.integer_nlp import (
    IntegerNLPSpec,
    IntegerProblem,
    solve_integer_nlp,
    splice_design_integers,
)
from mu_F.solvers.mixed_integer import resolve_integer_spec


__all__ = [
    "CurrentConstraintEvaluator",
    "CurrentCostEvaluator",
    "current_constraint_surrogate",
    "current_cost_surrogate",
]


# =============================================================================
# Shared geometry helper — used by both evaluators
# =============================================================================

def _current_geometry(graph, node, cfg):
    """Compute the geometry of a current-node sub-problem.

    Returns
    -------
    ndim          : int   — full NLP dimension (design + input + aux)
    n_design      : int   — pure design dim (= len(KS_bounds))
    n_input       : int
    n_aux         : int
    input_indices : np.ndarray[int]  — positions of input slots in the full vector
    aux_indices   : np.ndarray[int]  — positions of aux slots
    int_indices   : np.ndarray[int]  — positions of design-integer slots
                                       (same as cfg.case_study.design_domain integer positions)
    int_values    : tuple[tuple[int, ...], ...]
    ks_lb / ks_ub : jnp.ndarray      — bounds on the design portion (length n_design)
    """
    n_design = int(graph.nodes[node]['n_design_args'])
    n_input  = int(graph.nodes[node]['n_input_args'])
    n_aux    = int(graph.graph['n_aux_args'])
    ndim     = n_design + n_input + n_aux

    input_indices = np.arange(n_design, n_design + n_input).astype(int)
    aux_indices   = np.arange(ndim - n_aux, ndim).astype(int)

    int_dims, int_values = resolve_integer_spec(
        cfg.case_study.get('design_domain', None)
    )
    int_indices = np.array(int_dims, dtype=int)

    ks_design = graph.nodes[node]['KS_bounds']
    ks_lb = jnp.asarray([b[0] for b in ks_design]).reshape(-1)
    ks_ub = jnp.asarray([b[1] for b in ks_design]).reshape(-1)

    return (ndim, n_design, n_input, n_aux,
            input_indices, aux_indices, int_indices, int_values,
            ks_lb, ks_ub)


def _resolve_classifier(cfg, graph, key):
    """Pick the single-head vs K-head classifier callable for this node."""
    is_multihead = (
        cfg.samplers.ns.get('rejector', '') == 'sumb-xmeans'
        and graph.nodes[key].get('cluster_classifier_head') is not None
    )
    if is_multihead:
        return (int(graph.nodes[key]['cluster_classifier_n_heads']),
                graph.nodes[key]['cluster_classifier_head'])
    return (0, graph.nodes[key]['classifier'])


def _design_bounds_continuous(ks_lb, ks_ub, int_indices):
    """Drop integer-design slots from the design bounds — they live in p_aug."""
    if int_indices.size == 0:
        return [ks_lb, ks_ub]
    lb = jnp.asarray(np.delete(np.asarray(ks_lb), int_indices))
    ub = jnp.asarray(np.delete(np.asarray(ks_ub), int_indices))
    return [lb, ub]


def _prepare_inputs(inputs, n_y_expected):
    """Flatten inputs to a single (n_y_expected,) vector, zero-padded if short."""
    if inputs is None:
        return jnp.zeros((n_y_expected,))
    p = inputs.reshape(-1)
    return jnp.zeros(n_y_expected).at[:min(p.size, n_y_expected)].set(p[:n_y_expected])


# =============================================================================
# CurrentConstraintEvaluator — box-only feasibility on current node
# =============================================================================

class CurrentConstraintEvaluator(BaseEvaluator):
    """Stateful current-node feasibility evaluator.

    Box-only NLP: minimise the (masked) classifier over the reduced decision
    space.  The classifier output IS the objective — no general constraint.

    Single sub-problem (`_keys() = [node]`); built once, reused across every
    `evaluate(inputs, aux)` call.
    """

    def __init__(self, cfg, graph, node):
        self.specs: dict          = {}
        self.n_design_full: dict  = {}
        self.n_y: dict            = {}
        super().__init__(cfg, graph, node)
        self._thread = self.evaluate

    def _keys(self) -> list:
        return [self.node]

    def _build_for_key(self, key) -> None:
        (ndim, n_design, n_input, n_aux,
         input_indices, aux_indices, int_indices, int_values,
         ks_lb, ks_ub) = _current_geometry(self.graph, key, self.cfg)
        n_heads, classifier = _resolve_classifier(self.cfg, self.graph, key)
        n_int = int(int_indices.size)
        n_y   = n_input + n_aux

        # Constraint callable here is the objective (no general constraint).
        objective = mask_surrogate(
            classifier,
            ndim=ndim,
            fix_ind=input_indices,
            aux_ind=aux_indices,
            int_ind=int_indices,
            n_heads=n_heads,
            # aggregator inferred: 'scalar' or 'onehot_sum'
        )

        bounds = _design_bounds_continuous(ks_lb, ks_ub, int_indices)
        n_d_cont = int(bounds[0].size)
        n_params = n_y + n_int + n_heads

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

        self.specs[key] = IntegerNLPSpec(
            integer_problem    = integer_problem,
            continuous_factory = factory,
            screener           = screener,
            sobol_pool         = sobol_pool,
            n_starts           = self.n_starts,
            feasibility_tol    = self.feasibility_tol,
        )
        self.n_design_full[key] = n_design
        self.n_y[key]           = n_y

    def evaluate(self, inputs, aux):
        """Solve the box-only current-node NLP.

        Returns `(decision_variables, objective, converged)` in the legacy shape.
        At rollout `n_theta = 1`; the theta dim is preserved via outer `vmap`
        for consistency with the CTG / backward path.
        """
        key = self.node
        if key is None:
            n = 0 if inputs is None else inputs.shape[0]
            return (jnp.zeros((n, 0)),
                    jnp.zeros((n, 1)),
                    jnp.zeros((n, 1), dtype=bool))

        y = _prepare_inputs(inputs, self.n_y[key])             # (n_y,)
        # Rollout is single-theta; call once. The (1, …) shape on the return
        # value preserves the "theta dim is always present" contract for
        # consistency with the CTG / backward path.
        result = solve_integer_nlp(self.specs[key], y)

        full_design = splice_design_integers(
            result.x, self.specs[key], result.assignment_idx, self.n_design_full[key],
        )
        return (full_design.reshape(1, -1),
                jnp.asarray(result.objective).reshape(1, 1),
                jnp.asarray(result.success, dtype=bool).reshape(1, 1))


# =============================================================================
# CurrentCostEvaluator — constrained CTG on current node
# =============================================================================

class CurrentCostEvaluator(BaseEvaluator):
    """Stateful constrained-CTG evaluator for the current node.

    Minimises the node's CTG surrogate (via `mask_surrogate('scalar')`)
    subject to the classifier being feasible (via `mask_surrogate('scalar' |
    'onehot_sum')` depending on the rejector mode).  Multi-start seeds use
    the L1-penalty screen over the Sobol pool.

    Single sub-problem; one `IntegerNLPSpec` built per node in `_build_for_key`.
    Theta is handled via outer `vmap` (trivially `n_theta = 1` at rollout).
    """

    def __init__(self, cfg, graph, node):
        self.specs: dict          = {}
        self.n_design_full: dict  = {}
        self.n_y: dict            = {}
        super().__init__(cfg, graph, node)
        self._thread = self.evaluate

    def _keys(self) -> list:
        return [self.node]

    def _build_for_key(self, key) -> None:
        (ndim, n_design, n_input, n_aux,
         input_indices, aux_indices, int_indices, int_values,
         ks_lb, ks_ub) = _current_geometry(self.graph, key, self.cfg)
        n_heads, classifier = _resolve_classifier(self.cfg, self.graph, key)
        n_int = int(int_indices.size)
        n_y   = n_input + n_aux

        # Objective: CTG surrogate, scalar (always single-output regression).
        ctg_surrogate = self.graph.nodes[key]['ctg_surrogate']
        objective = mask_surrogate(
            ctg_surrogate,
            ndim=ndim,
            fix_ind=input_indices,
            aux_ind=aux_indices,
            int_ind=int_indices,
            n_heads=0,                # CTG is always single-output
            aggregator='scalar',
        )

        # Constraint: classifier ≤ 0 — single-head or K-head SOS1 depending on rejector.
        masked_clf = mask_surrogate(
            classifier,
            ndim=ndim,
            fix_ind=input_indices,
            aux_ind=aux_indices,
            int_ind=int_indices,
            n_heads=n_heads,
            # aggregator inferred: 'scalar' or 'onehot_sum'
        )
        # mask_surrogate's scalar/onehot_sum aggregators return () — wrap
        # to (1,) so septal's lagrangian_grad's `jac_g.T @ lam` aligns
        # with `n_constraints=1`.
        def constraint(x_red, p_aug):
            return jnp.atleast_1d(masked_clf(x_red, p_aug))

        bounds = _design_bounds_continuous(ks_lb, ks_ub, int_indices)
        n_d_cont = int(bounds[0].size)
        n_params = n_y + n_int + n_heads

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

        integer_problem = IntegerProblem.from_cfg(
            design_domain=self.cfg.case_study.get('design_domain', None),
            n_heads=n_heads,
        )

        self.specs[key] = IntegerNLPSpec(
            integer_problem    = integer_problem,
            continuous_factory = factory,
            screener           = screener,
            sobol_pool         = sobol_pool,
            n_starts           = self.n_starts,
            feasibility_tol    = self.feasibility_tol,
        )
        self.n_design_full[key] = n_design
        self.n_y[key]           = n_y

    def evaluate(self, inputs, aux):
        """Solve the constrained-CTG current-node NLP.

        Returns `(decision_variables, objective, converged)` matching the
        rollout agent's contract.  Theta dim is trivially 1 at rollout —
        we solve once and reshape to `(1, …)` to preserve the contract.
        """
        key = self.node
        if key is None:
            n = 0 if inputs is None else inputs.shape[0]
            return (jnp.zeros((n, 0)),
                    jnp.zeros((n, 1)),
                    jnp.zeros((n, 1), dtype=bool))

        y = _prepare_inputs(inputs, self.n_y[key])             # (n_y,)
        result = solve_integer_nlp(self.specs[key], y)

        full_design = splice_design_integers(
            result.x, self.specs[key], result.assignment_idx, self.n_design_full[key],
        )
        return (full_design.reshape(1, -1),
                jnp.asarray(result.objective).reshape(1, 1),
                jnp.asarray(result.success, dtype=bool).reshape(1, 1))


# =============================================================================
# Evaluator caches + public entry points
# =============================================================================

_CURRENT_CONSTRAINT_CACHE: dict = {}
_CURRENT_COST_CACHE: dict       = {}


def _get_constraint_evaluator(cfg, graph, node) -> CurrentConstraintEvaluator:
    key = (id(graph), node)
    evaluator = _CURRENT_CONSTRAINT_CACHE.get(key)
    if evaluator is None:
        evaluator = CurrentConstraintEvaluator(cfg, graph, node)
        _CURRENT_CONSTRAINT_CACHE[key] = evaluator
    return evaluator


def _get_cost_evaluator(cfg, graph, node) -> CurrentCostEvaluator:
    key = (id(graph), node)
    evaluator = _CURRENT_COST_CACHE.get(key)
    if evaluator is None:
        evaluator = CurrentCostEvaluator(cfg, graph, node)
        _CURRENT_COST_CACHE[key] = evaluator
    return evaluator


def current_constraint_surrogate(inputs, aux, cfg, graph, node):
    """Box-only feasibility check — minimise classifier over the reduced
    decision space with fixed `inputs` threaded as `p`.
    """
    if node is None:
        n = 0 if inputs is None else inputs.shape[0]
        return (jnp.zeros((0, n)),
                jnp.zeros((n, 1)),
                jnp.zeros((n, 1), dtype=bool))
    return _get_constraint_evaluator(cfg, graph, node).evaluate(inputs, aux)


def current_cost_surrogate(inputs, aux, cfg, graph, node):
    """Constrained-CTG on the current node — minimise CTG s.t. classifier <= 0."""
    if node is None:
        n = 0 if inputs is None else inputs.shape[0]
        return (jnp.zeros((0, n)),
                jnp.zeros((n, 1)),
                jnp.zeros((n, 1), dtype=bool))
    return _get_cost_evaluator(cfg, graph, node).evaluate(inputs, aux)
