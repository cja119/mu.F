"""Current-node Surrogate evaluators: box-only feasibility and constrained CTG."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import vmap

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
    solve_integer_nlp_batched,
    splice_design_integers,
)
from mu_F.solvers.mixed_integer import resolve_integer_spec


__all__ = [
    "CurrentConstraintEvaluator",
    "CurrentCostEvaluator",
    "current_constraint_surrogate",
    "current_cost_surrogate",
]


# ---------------------------------------------------------------------------
# Shared geometry helper — used by both evaluators
# ---------------------------------------------------------------------------

def _current_geometry(graph, node, cfg):
    """
    Compute the geometry of a current-node sub-problem: dimensions, the
    fix / aux / integer slot indices, and the design-portion bounds.
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


def _assemble_ys(inputs, aux, n_y):
    """Pack the recovery NLP's fixed params as [input, aux], matching the
    backward sweep so the surrogate is queried at the trajectory's true aux."""
    in2 = jnp.atleast_2d(inputs) if inputs is not None else jnp.zeros((1, 0))
    n = in2.shape[0]
    aux2 = (jnp.atleast_2d(aux) if aux is not None and jnp.asarray(aux).size
            else jnp.zeros((n, 0)))
    if aux2.shape[0] == 1 and n > 1:
        aux2 = jnp.broadcast_to(aux2, (n, aux2.shape[1]))
    n_input = int(n_y) - aux2.shape[1]
    take = min(int(in2.shape[1]), n_input)
    ins = jnp.zeros((n, n_input)).at[:, :take].set(in2[:, :take])
    return jnp.concatenate([ins, aux2], axis=-1)                          # (N, n_y)


# ---------------------------------------------------------------------------
# CurrentConstraintEvaluator — box-only feasibility on current node
# ---------------------------------------------------------------------------

class CurrentConstraintEvaluator(BaseEvaluator):
    """Stateful current-node feasibility evaluator.

    Box-only NLP minimising the masked classifier over the reduced decision
    space, where the classifier output is the objective.  Holds a single
    sub-problem built once and reused across every evaluate call.

    """

    _eval_name = 'current_constraint'

    # ---- External Methods ----

    def __init__(self, cfg, graph, node):
        self.specs: dict          = {}
        self.n_design_full: dict  = {}
        self.n_y: dict            = {}
        super().__init__(cfg, graph, node)
        self._thread = self.evaluate

    def evaluate(self, inputs, aux):
        """
        Solve the box-only current-node NLP, returning
        (decision_variables, objective, converged) with the theta dim kept
        explicit (trivially 1 at rollout).
        """
        key = self.node
        if key is None:
            n = 0 if inputs is None else inputs.shape[0]
            return (jnp.zeros((n, 0)),
                    jnp.zeros((n, 1)),
                    jnp.zeros((n, 1), dtype=bool))

        # Pack the fixed [input, aux] params for the N trajectory states.
        ys = _assemble_ys(inputs, aux, self.n_y[key])                         # (N, n_y)

        per_theta = solve_integer_nlp_batched(self.specs[key], ys)
        full_design = vmap(
            lambda x, a: splice_design_integers(x, self.specs[key], a, self.n_design_full[key])
        )(per_theta.x, per_theta.assignment_idx)                              # (N, n_design)
        return (full_design,
                per_theta.objective.reshape(-1, 1),
                per_theta.success.reshape(-1, 1).astype(bool))

    # ---- Private Methods ----

    def _keys(self) -> list:
        """Sub-problem keys: the current node only."""
        return [self.node]

    def _build_for_key(self, key) -> None:
        """Build the box-only current-node IntegerNLPSpec."""
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


# ---------------------------------------------------------------------------
# CurrentCostEvaluator — constrained CTG on current node
# ---------------------------------------------------------------------------

class CurrentCostEvaluator(BaseEvaluator):
    """Stateful constrained-CTG evaluator for the current node.

    Minimises the node's CTG Surrogate subject to the classifier being
    feasible, with multi-start seeds drawn from the L1-penalty Sobol screen.
    Holds a single sub-problem built once per node; theta is an outer vmap.

    """

    _eval_name = 'current_cost'

    # ---- External Methods ----

    def __init__(self, cfg, graph, node):
        self.specs: dict          = {}
        self.n_design_full: dict  = {}
        self.n_y: dict            = {}
        super().__init__(cfg, graph, node)
        self._thread = self.evaluate

    # ---- Private Methods ----

    def _keys(self) -> list:
        """Sub-problem keys: the current node only."""
        return [self.node]

    def _build_for_key(self, key) -> None:
        """Build the constrained-CTG current-node IntegerNLPSpec."""
        (ndim, n_design, n_input, n_aux,
         input_indices, aux_indices, int_indices, int_values,
         ks_lb, ks_ub) = _current_geometry(self.graph, key, self.cfg)
        n_int = int(int_indices.size)
        n_y   = n_input + n_aux

        # Objective: CTG Surrogate, scalar (always single-output regression).
        objective = mask_surrogate(
            self.graph.nodes[key]['ctg_surrogate'],
            ndim=ndim,
            fix_ind=input_indices,
            aux_ind=aux_indices,
            int_ind=int_indices,
            n_heads=0,                # CTG is always single-output
            aggregator='scalar',
        )

        # Feasible region: classifier, or chance constraint under direct-probability.
        constraint, n_heads = self._feasibility_constraint(
            key, ndim, input_indices, aux_indices, int_indices,
        )

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

    def _feasibility_constraint(self, key, ndim, input_indices, aux_indices, int_indices):
        """Feasible-region constraint for the current-node CTG optimisation."""
        direct_prob = (bool(self.cfg.samplers.deus.get('direct_probability', False))
                       and self.cfg.formulation == 'probabilistic')
        if direct_prob:
            return self._chance_constraint(key, ndim, input_indices, aux_indices, int_indices)
        return self._classifier_constraint(key, ndim, input_indices, aux_indices, int_indices)

    def _chance_constraint(self, key, ndim, input_indices, aux_indices, int_indices):
        """Chance constraint `P_feas(x) >= p_target` from the node's probability_map.

        The factory enforces `g <= 0`, so feasibility `P >= p_target` is encoded
        as `g = p_target - P`.
        """
        uw = self.cfg.samplers.unit_wise_target_reliability
        try:
            p_target = float(uw[key])
        except (TypeError, KeyError, IndexError):
            p_target = float(uw)
        masked = mask_surrogate(
            self.graph.nodes[key]['probability_map'],
            ndim=ndim, fix_ind=input_indices, aux_ind=aux_indices,
            int_ind=int_indices, n_heads=0, aggregator='scalar',
        )
        def constraint(x_red, p_aug):
            return jnp.atleast_1d(p_target - masked(x_red, p_aug))
        return constraint, 0

    def _classifier_constraint(self, key, ndim, input_indices, aux_indices, int_indices):
        """Node feasibility classifier (multi-head aware), tightened by the
        a-posteriori backoff so the recovery lands inside the feasible region."""
        n_heads, classifier = _resolve_classifier(self.cfg, self.graph, key)
        masked = mask_surrogate(
            classifier, ndim=ndim, fix_ind=input_indices, aux_ind=aux_indices,
            int_ind=int_indices, n_heads=n_heads,
        )
        backoff = jnp.sum(jnp.asarray(self.graph.nodes[key]['constraint_backoff']))
        def constraint(x_red, p_aug):
            return jnp.atleast_1d(masked(x_red, p_aug) + backoff)
        return constraint, n_heads

    def evaluate(self, inputs, aux):
        """
        Solve the constrained-CTG current-node NLP, returning
        (decision_variables, objective, converged) per the rollout agent's
        contract.  Theta dim is kept explicit (trivially 1 at rollout).
        """
        key = self.node
        if key is None:
            n = 0 if inputs is None else inputs.shape[0]
            return (jnp.zeros((n, 0)),
                    jnp.zeros((n, 1)),
                    jnp.zeros((n, 1), dtype=bool))

        # Pack the fixed [input, aux] params for the N trajectory states.
        ys = _assemble_ys(inputs, aux, self.n_y[key])                         # (N, n_y)

        per_theta = solve_integer_nlp_batched(self.specs[key], ys)
        full_design = vmap(
            lambda x, a: splice_design_integers(x, self.specs[key], a, self.n_design_full[key])
        )(per_theta.x, per_theta.assignment_idx)                              # (N, n_design)
        return (full_design,
                per_theta.objective.reshape(-1, 1),
                per_theta.success.reshape(-1, 1).astype(bool))


# ---------------------------------------------------------------------------
# Evaluator caches + public entry points
# ---------------------------------------------------------------------------

_CURRENT_CONSTRAINT_CACHE: dict = {}
_CURRENT_COST_CACHE: dict       = {}


def _get_constraint_evaluator(cfg, graph, node) -> CurrentConstraintEvaluator:
    """Return the cached CurrentConstraintEvaluator for this (graph, node)."""
    key = (id(graph), node)
    evaluator = _CURRENT_CONSTRAINT_CACHE.get(key)
    if evaluator is None:
        evaluator = CurrentConstraintEvaluator(cfg, graph, node)
        _CURRENT_CONSTRAINT_CACHE[key] = evaluator
    return evaluator


def _get_cost_evaluator(cfg, graph, node) -> CurrentCostEvaluator:
    """Return the cached CurrentCostEvaluator for this (graph, node)."""
    key = (id(graph), node)
    evaluator = _CURRENT_COST_CACHE.get(key)
    if evaluator is None:
        evaluator = CurrentCostEvaluator(cfg, graph, node)
        _CURRENT_COST_CACHE[key] = evaluator
    return evaluator


def current_constraint_surrogate(inputs, aux, cfg, graph, node):
    """
    Box-only feasibility check: minimise the classifier over the reduced
    decision space with fixed inputs threaded as p.
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
