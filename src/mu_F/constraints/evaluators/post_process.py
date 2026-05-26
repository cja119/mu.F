"""
Post-processing global NLP evaluator.

Solves the upper-level reconstruction problem: find decision variables
(graph-level, excluding `post_process_decision_indices`) that minimise
the last decision component subject to the upper-level classifier staying
feasible (`classifier(x) <= 0`).

Epigraph reformulation: the casadi path encoded the objective as the
integer sentinel `obj_fn = -1` (interpreted as "minimise `x[-1]`").  We
write that as a plain JAX scalar — no sentinel indirection.

Classifier resolution: preferred-live-then-rebuild via the graph attrs
`post_process_upper_classifier` (callable) or
`post_process_upper_classifier_serialised` (params dict — checkpoint-
resume path).

Graph-wide, so `_keys()` yields `[None]`.  No `p` (problem is fully static
at construction); no integer enumeration (graph-level NLP is purely
continuous); no multi-head (upper classifier is single-output).  Runs
through the unified `solve_integer_nlp` path with an empty
`IntegerProblem` for uniformity with the other rewired evaluators —
`feasible_assignments()` returns the trivial `(1, 0)` and the solve
degenerates to a standard multi-start SQP.
"""
from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
import numpy as np

from mu_F.constraints.evaluators.base import (
    BaseEvaluator,
    build_factory,
    build_penalty_screener,
    precompute_sobol_pool,
)
from mu_F.solvers.integer_nlp import (
    IntegerNLPSpec,
    IntegerProblem,
    solve_integer_nlp,
)


__all__ = ["PostProcessUpperLevelEvaluator", "post_process_upper_level"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_upper_classifier(graph, cfg):
    """Resolve the upper-level classifier to a live JAX callable.

    Preferred: live `post_process_upper_classifier` on `graph.graph`.
    Fallback: rebuild from `..._serialised` — the checkpoint-resume path.
    """
    live = graph.graph.get('post_process_upper_classifier')
    if callable(live):
        return live

    serialised = graph.graph.get('post_process_upper_classifier_serialised')
    if serialised is None:
        raise RuntimeError(
            "No upper-level classifier available on graph.graph — neither "
            "a live `post_process_upper_classifier` callable nor a "
            "`post_process_upper_classifier_serialised` params dict."
        )
    # Lazy import: avoid paying the surrogate-reconstruction cost on the
    # happy path.
    from mu_F.solvers.utilities import construct_model
    return construct_model(
        serialised, cfg,
        supervised_learner='classification',
        model_type=cfg.surrogate.classifier_selection,
        model_surrogate='live_set_surrogate',
    )


def _upper_level_bounds(graph, cfg) -> list:
    """Drop `post_process_decision_indices` from the graph-level box.

    Bounds stay in real-world units — the upper-level classifier
    self-scales internally.
    """
    total_ind = np.arange(graph.graph['n_design_args'] + graph.graph['n_aux_args'])
    fix_ind   = np.array(graph.graph['post_process_decision_indices']).reshape(-1)
    dec_ind   = np.delete(total_ind, fix_ind).astype(int)

    lb = jnp.hstack([
        jnp.array(bound[0]).reshape(-1)
        for bound in graph.graph['bounds'] if bound[0] != 'None'
    ])
    ub = jnp.hstack([
        jnp.array(bound[1]).reshape(-1)
        for bound in graph.graph['bounds'] if bound[1] != 'None'
    ])
    return [lb[dec_ind], ub[dec_ind]]


# =============================================================================
# PostProcessUpperLevelEvaluator — one instance per (cfg, graph)
# =============================================================================

class PostProcessUpperLevelEvaluator(BaseEvaluator):
    """Stateful upper-level post-processing evaluator.

    Single graph-wide sub-problem; one `IntegerNLPSpec` built in
    `_build_for_key` and reused across calls.
    """

    def __init__(self, cfg, graph, node=None):
        # `node` accepted for signature parity with other evaluators but unused.
        self.specs: dict = {}
        self.n_d: dict   = {}
        super().__init__(cfg, graph, node)
        self._thread = self.evaluate

    def _keys(self) -> list:
        return [None]

    def _build_for_key(self, key) -> None:
        classifier = _load_upper_classifier(self.graph, self.cfg)
        lb_raw, ub_raw = _upper_level_bounds(self.graph, self.cfg)
        lb = jnp.asarray(lb_raw).reshape(-1)
        ub = jnp.asarray(ub_raw).reshape(-1)
        bounds = [lb, ub]
        n_d = int(lb.size)

        # Epigraph reformulation: minimise the last decision variable.
        def objective(x, p):
            return x.reshape(-1)[-1]

        def constraint(x, p):
            return classifier(x.reshape(1, -1)).reshape(-1)

        # Probe the classifier at midpoint to discover n_g (usually 1, but
        # don't assume — some upper classifiers bundle multiple outputs).
        x_probe = 0.5 * (lb + ub)
        n_g = int(classifier(x_probe.reshape(1, -1)).reshape(-1).shape[0])

        factory = build_factory(
            objective, constraint, bounds,
            n_decision=n_d,
            n_params=0,                                     # no parametric input
            n_constraints=n_g,
            feasibility_tol=self.feasibility_tol,
            optimality_tol=self.optimality_tol,
            max_iter=self.max_iter,
        )
        screener   = build_penalty_screener(objective, constraint, self.screen_penalty)
        sobol_pool = precompute_sobol_pool(bounds, n_d, self.n_sobol_screen)

        # Empty integer problem — purely continuous graph-level NLP.
        # feasible_assignments() returns (1, 0) and the solve becomes a
        # standard multi-start SQP from screened Sobol seeds.
        integer_problem = IntegerProblem()

        self.specs[key] = IntegerNLPSpec(
            integer_problem    = integer_problem,
            continuous_factory = factory,
            screener           = screener,
            sobol_pool         = sobol_pool,
            n_starts           = self.n_starts,
            feasibility_tol    = self.feasibility_tol,
        )
        self.n_d[key] = n_d

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Solve the upper-level NLP via the unified integer-NLP path.

        Returns `(objective, decision_variables)` matching the legacy
        contract.  Empty `y` since the problem is fully static.
        """
        key = None
        y = jnp.zeros((0,))                                 # n_params = 0
        result = solve_integer_nlp(self.specs[key], y)
        return jnp.asarray(result.objective).reshape(()), result.x.reshape(-1)


# =============================================================================
# Evaluator cache + public entry point
# =============================================================================

_POST_PROCESS_EVALUATOR_CACHE: dict = {}


def _get_evaluator(cfg, graph, node) -> PostProcessUpperLevelEvaluator:
    key = id(graph)
    evaluator = _POST_PROCESS_EVALUATOR_CACHE.get(key)
    if evaluator is None:
        evaluator = PostProcessUpperLevelEvaluator(cfg, graph, node)
        _POST_PROCESS_EVALUATOR_CACHE[key] = evaluator
    return evaluator


def post_process_upper_level(cfg, graph, node) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Solve the upper-level global reconstruction NLP via septal.

    Arguments match `constraint_evaluator(..., constraint_type='post_process_upper_level')`.
    `node` is accepted for signature parity with other evaluators but
    unused — this problem is graph-wide, not node-local.

    Returns
    -------
    (objective, decision_variables) : (scalar, (n_d,))
    """
    return _get_evaluator(cfg, graph, node).evaluate()
