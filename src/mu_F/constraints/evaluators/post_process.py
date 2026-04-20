"""
Post-processing global NLP evaluator.

Solves the upper-level reconstruction problem: find decision variables (graph
level, excluding `post_process_decision_indices`) that minimise the last
decision component subject to the upper-level classifier staying feasible
(`classifier(x) <= 0`).

The casadi path (previously `global_graph_upperlevel_NLP` in
`constraints/downstream.py`) encoded the objective as the integer sentinel
`obj_fn = -1`, which `build_objective_function` in the old solver utilities
interpreted as "minimise `x[-1]`".  We lift that into a plain JAX scalar
here — no sentinel indirection.

Classifier is read from the graph in preferred-live-then-rebuild order:
`graph.graph['post_process_upper_classifier']` if callable, otherwise
`graph.graph['post_process_upper_classifier_serialised']` reconstructed
via `mu_F.solvers.utilities.construct_model`.  Live case is normal during
a run; rebuild is the checkpoint-resume path.

### BaseEvaluator port

Graph-wide, so `_keys()` yields a single sentinel `[None]`.  No `p`
parameters — the problem is fully static, so `n_params=0`.  One factory
built in `_build_for_key`, reused across calls.  A midpoint-seeded
multi-start batch runs through `solve_batch`.
"""
from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
import numpy as np

from mu_F.constraints.evaluators.base import (
    BaseEvaluator,
    build_factory,
    pick_best,
    precompute_sobol_pool,
)


__all__ = ["PostProcessUpperLevelEvaluator", "post_process_upper_level"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_upper_classifier(graph, cfg):
    """
    Resolve the upper-level classifier to a live JAX callable.

    Preferred: the graph already holds a callable under
    `post_process_upper_classifier` (normal runs).  Fallback: rebuild from
    the serialised dict — the checkpoint-resume path.
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
    # Rebuild lazily to avoid paying the surrogate-reconstruction import cost
    # on happy-path runs.
    from mu_F.solvers.utilities import construct_model
    return construct_model(
        serialised, cfg,
        supervised_learner='classification',
        model_type=cfg.surrogate.classifier_selection,
        model_surrogate='live_set_surrogate',
    )


def _upper_level_bounds(graph, cfg) -> list[jnp.ndarray]:
    """
    Compute the upper-level decision bounds.

    Total graph-level vars are `n_design + n_aux`; drop the indices listed
    in `post_process_decision_indices` (those are the lower-level decisions
    held fixed here).  Bounds stay in real-world units — the upper-level
    classifier self-scales internally.
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
    bounds = [lb, ub]

    return [bounds[0][dec_ind], bounds[1][dec_ind]]


# =============================================================================
# PostProcessUpperLevelEvaluator — one instance per (cfg, graph)
# =============================================================================

class PostProcessUpperLevelEvaluator(BaseEvaluator):
    """
    Stateful upper-level post-processing evaluator.

    Graph-wide — no successors / predecessors / node iteration.  `_keys()`
    yields a single `[None]` sentinel so the base class's build loop runs
    exactly once and populates `self.factories[None]`.

    The problem is fully static (bounds and classifier come from the graph
    at construction), so `n_params=0` and `p` is an empty vector.
    """

    def __init__(self, cfg, graph, node=None):
        # Single graph-wide sub-problem.  `node` is accepted for signature
        # parity with other evaluators but unused.
        self.bounds: dict = {}
        self.n_d: dict    = {}

        super().__init__(cfg, graph, node)

        # Pin bound method — harmless for this evaluator (not pmap'd), but
        # keeps the pattern consistent across the evaluator family.
        self._shard = self.evaluate

    # ------------------------------------------------------------------
    # BaseEvaluator contract
    # ------------------------------------------------------------------

    def _keys(self) -> list:
        return [None]

    def _build_for_key(self, key) -> None:
        classifier = _load_upper_classifier(self.graph, self.cfg)
        bounds     = _upper_level_bounds(self.graph, self.cfg)
        lb         = jnp.asarray(bounds[0]).reshape(-1)
        ub         = jnp.asarray(bounds[1]).reshape(-1)
        n_d        = int(lb.size)

        # Epigraph reformulation: minimise the last decision variable.
        def objective(x, p):
            return x.reshape(-1)[-1]

        def constraint(x, p):
            return classifier(x.reshape(1, -1)).reshape(-1)

        # Probe the classifier at midpoint to discover n_g (usually 1 but
        # don't assume — some upper classifiers bundle multiple outputs).
        x_probe = 0.5 * (lb + ub)
        n_g = int(classifier(x_probe.reshape(1, -1)).reshape(-1).shape[0])

        self.factories[key] = build_factory(
            objective, constraint, [lb, ub],
            n_decision=n_d,
            n_params=0,
            n_constraints=n_g,
            tol=self.tol,
        )
        # Sobol pool kept for optional multi-start; midpoint seed is the
        # canonical behaviour preserved from the casadi path.
        self.sobol_pool[key] = precompute_sobol_pool(
            [lb, ub], n_d, self.n_sobol_screen,
        )

        self.bounds[key] = [lb, ub]
        self.n_d[key]    = n_d

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Solve the upper-level NLP via `solve_batch` (single midpoint start).

        Returns
        -------
        (objective, decision_variables) : (scalar, (n_d,))
            Matches the return shape the casadi path yielded, so downstream
            code (sampling schemes, visualisers) consumes the tuple unchanged.
        """
        lb, ub = self.bounds[None]
        x0 = 0.5 * (lb + ub)
        x0_batch = x0.reshape(1, -1)
        p_batch  = jnp.zeros((1, 0))

        result = self.factories[None].solve_batch(x0_batch, p_batch)
        best_f, _, best_x = pick_best(result)
        return best_f.reshape(()), best_x.reshape(-1)


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
    """
    Solve the upper-level global reconstruction NLP via septal.

    Arguments match `constraint_evaluator(..., constraint_type='post_process_upper_level')`.
    `node` is accepted for signature parity with other evaluators but unused —
    this problem is graph-wide, not node-local.

    Returns
    -------
    (objective, decision_variables) : (scalar, (n_d,))
    """
    return _get_evaluator(cfg, graph, node).evaluate()
