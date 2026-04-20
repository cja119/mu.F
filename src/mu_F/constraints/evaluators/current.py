"""
Current-node surrogate evaluators.

Two flavours, both operating on the current node's own live surrogates
(not a successor's / predecessor's):

  - `current_constraint_surrogate`  — box-only feasibility check: min
    classifier(assemble(x, inputs)) over (design, aux).  No general
    constraints.  Used for "rollout constraint" checks.

  - `current_cost_surrogate`  — constrained CTG: minimise the node's CTG
    surrogate subject to the node's classifier being feasible (g(x) <= 0).
    Single-problem (no successor fan-out) so batching happens entirely via
    vmap inside septal — no pmap wrapper.

Both use the reduced decision space (inputs + aux indices masked away from
the full NLP) since those positions are pinned by the supplied `inputs`.

### BaseEvaluator port

Each evaluator has a single sub-problem (`_keys() = [node]`).  The per-
sample fixed inputs are threaded as the `p` parameter instead of baked
into Python closures — one factory per node, reused across calls.

`evaluate(inputs, aux)` returns `(decision_variables.T, objective, converged)`
to match the legacy contract; the transpose is preserved so rollout
agents keep indexing as `v.T[i]`.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from mu_F.constraints.evaluators.base import (
    BaseEvaluator,
    build_factory,
    build_penalty_screener,
    pick_best,
    pick_x0_batch,
    precompute_sobol_pool,
)
from mu_F.constraints.utils import (
    mask_classifier,
)


__all__ = [
    "CurrentConstraintEvaluator",
    "CurrentCostEvaluator",
    "current_constraint_surrogate",
    "current_cost_surrogate",
]


# =============================================================================
# Shared index / bounds helper — used by both evaluators
# =============================================================================

def _current_reduced_space(graph, node, cfg):
    """
    Compute the reduced decision space for a current-node sub-problem.

    Returns
    -------
    ndim          : int                    — full NLP dimension
    input_indices : np.ndarray[int]        — fixed input slots
    aux_indices   : np.ndarray[int]        — fixed aux slots
    fix_indices   : np.ndarray[int]        — concatenation of the above
    decision_bounds : list[jnp.ndarray]    — [lb, ub] on the reduced space
    """
    n_design = graph.nodes[node]['n_design_args']
    n_input  = graph.nodes[node]['n_input_args']
    n_aux    = graph.graph['n_aux_args']
    ndim     = n_design + n_input + n_aux

    input_indices = np.arange(n_design, n_design + n_input).astype(int)
    aux_indices   = np.arange(ndim - n_aux, ndim).astype(int)
    fix_indices   = np.hstack([input_indices, aux_indices]).astype(int)

    # Bounds stay in real-world units.  The surrogates stored on the graph
    # (`classifier`, `ctg_surrogate`) are self-scaling — they standardise
    # their inputs internally using their own trained scaler.  The solver
    # runs in real space; scaling concerns live entirely inside each
    # surrogate callable.
    decision_bounds = graph.nodes[node]["extendedDS_bounds"]

    decision_bounds = [
        jnp.delete(bound, fix_indices, axis=1) for bound in decision_bounds
    ]
    lb = jnp.asarray(decision_bounds[0]).reshape(-1)
    ub = jnp.asarray(decision_bounds[1]).reshape(-1)

    return ndim, input_indices, aux_indices, fix_indices, [lb, ub]


def _prepare_inputs(inputs, graph, node, fix_indices, cfg):
    """
    Flatten `inputs` to a single fixed-slot vector (shape `(n_fix,)`).

    Inputs are passed through verbatim: every callable on the graph is
    self-scaling (takes real-world input, standardises internally), so
    there's no external scaling step to apply here any more.
    """
    if inputs is None:
        return jnp.zeros((0,))
    return inputs.reshape(-1)


# =============================================================================
# CurrentConstraintEvaluator — box-only feasibility on current node
# =============================================================================

class CurrentConstraintEvaluator(BaseEvaluator):
    """
    Stateful current-node feasibility evaluator.

    Box-only NLP: minimise classifier(assemble(x, p)) over the reduced
    decision space, with fixed inputs `p` threaded as the parametric
    problem's `p` argument.  No general constraints.

    Single sub-problem (`_keys() = [node]`) — build once, reuse across
    every `evaluate(inputs, aux)` call.
    """

    def __init__(self, cfg, graph, node):
        self.bounds: dict        = {}
        self.n_d: dict           = {}
        self.n_fix: dict         = {}
        self.fix_indices: dict   = {}
        self.ndim: dict          = {}
        self.objective_fn: dict  = {}

        super().__init__(cfg, graph, node)

        self._shard = self.evaluate

    def _keys(self) -> list:
        return [self.node]

    def _build_for_key(self, key) -> None:
        ndim, input_indices, aux_indices, fix_indices, bounds = \
            _current_reduced_space(self.graph, key, self.cfg)
        lb, ub = bounds
        n_d = int(lb.size)
        n_fix = int(len(fix_indices))

        classifier         = self.graph.nodes[key]['classifier']
        wrapped_classifier = mask_classifier(
            classifier, ndim, input_indices, aux_indices,
        )

        def objective(x, p):
            return wrapped_classifier(x, p.reshape(1, -1)).squeeze()

        self.factories[key] = build_factory(
            objective, None, bounds,
            n_decision=n_d,
            n_params=n_fix,
            n_constraints=0,
            tol=self.tol,
        )
        self.sobol_pool[key] = precompute_sobol_pool(
            bounds, n_d, self.n_sobol_screen,
        )

        self.bounds[key]       = bounds
        self.n_d[key]          = n_d
        self.n_fix[key]        = n_fix
        self.fix_indices[key]  = fix_indices
        self.ndim[key]         = ndim
        self.objective_fn[key] = objective

    def evaluate(self, inputs, aux):
        """
        Solve the box-only current-node NLP with `cfg.solvers.n_starts`
        Sobol-seeded starts.

        Returns
        -------
        (decision_variables.T, objective, converged)
            decision_variables : (n_d, n_starts) — transposed for `v.T[i]`.
            objective          : (n_starts, 1)
            converged          : (n_starts, 1), dtype bool
        """
        key = self.node
        if key is None:
            n = 0 if inputs is None else inputs.shape[0]
            return (
                jnp.zeros((0, n)),
                jnp.zeros((n, 1)),
                jnp.zeros((n, 1), dtype=bool),
            )

        p = _prepare_inputs(inputs, self.graph, key, self.fix_indices[key], self.cfg)
        # The classifier's `construct_input` only consumes the first
        # `n_fix` slots of `p`; pad or trim to that width for determinism.
        p = jnp.zeros(self.n_fix[key]).at[:min(p.size, self.n_fix[key])].set(
            p[:self.n_fix[key]]
        )

        # Box-only — no constraint to penalise against, so the L1-penalty
        # screener degenerates to "sort by objective".  Take the first
        # `n_starts` Sobol points directly; multi-start variance is
        # preserved by the deterministic pool layout.
        x0_batch = _sobol_topn(self.sobol_pool[key], self.n_starts)
        p_batch = jnp.broadcast_to(
            p.reshape(1, -1), (self.n_starts, self.n_fix[key]),
        )

        result = self.factories[key].solve_batch(x0_batch, p_batch)

        # Solver ran in real-world decision space (bounds are real, inputs
        # are real, surrogates self-scale) — no post-solve inverse
        # transform needed.
        #
        # Collapse the `n_starts` multi-start axis via `pick_best` so the
        # caller (rollout) receives exactly one decision per sample.  The
        # old return shape `(n_d_k, n_starts)` caused the multi-start axis
        # to be interpreted as a feature axis downstream — that poisoned
        # the rollout's forward pass at non-root nodes.
        best_f, best_c, best_x = pick_best(result)
        params    = jnp.asarray(best_x).reshape(-1, 1)   # (n_d_k, 1)
        objective = jnp.asarray(best_f).reshape(1, 1)    # (1, 1)
        converged = jnp.asarray(best_c).reshape(1, 1)    # (1, 1)
        return params, objective, converged


# =============================================================================
# CurrentCostEvaluator — constrained CTG on current node
# =============================================================================

class CurrentCostEvaluator(BaseEvaluator):
    """
    Stateful constrained-CTG evaluator for the current node.

    Minimises the node's CTG surrogate subject to its classifier staying
    feasible.  `p` = flattened fixed inputs (same as
    `CurrentConstraintEvaluator`).  Multi-start seeds use the L1-penalty
    screen over the Sobol pool.
    """

    def __init__(self, cfg, graph, node):
        self.bounds: dict         = {}
        self.n_d: dict            = {}
        self.n_fix: dict          = {}
        self.fix_indices: dict    = {}
        self.ndim: dict           = {}
        self.objective_fn: dict   = {}
        self.constraint_fn: dict  = {}

        super().__init__(cfg, graph, node)

        self._shard = self.evaluate

    def _keys(self) -> list:
        return [self.node]

    def _build_for_key(self, key) -> None:
        ndim, input_indices, aux_indices, fix_indices, bounds = \
            _current_reduced_space(self.graph, key, self.cfg)
        lb, ub = bounds
        n_d = int(lb.size)
        n_fix = int(len(fix_indices))

        ctg_surrogate      = self.graph.nodes[key]['ctg_surrogate']
        classifier         = self.graph.nodes[key]['classifier']
        wrapped_ctg        = mask_classifier(ctg_surrogate, ndim, input_indices, aux_indices)
        wrapped_classifier = mask_classifier(classifier,    ndim, input_indices, aux_indices)

        # CTG surrogate may return a per-uncertainty-sample vector — sum to
        # scalar to match the casadi `_ensure_scalar_objective` behaviour.
        def objective(x, p):
            return wrapped_ctg(x, p.reshape(1, -1)).reshape(-1).sum()

        def constraint(x, p):
            return wrapped_classifier(x, p.reshape(1, -1)).reshape(-1)

        self.factories[key] = build_factory(
            objective, constraint, bounds,
            n_decision=n_d,
            n_params=n_fix,
            n_constraints=1,
            tol=self.tol,
        )
        self.screeners[key] = build_penalty_screener(
            objective, constraint, self.screen_penalty,
        )
        self.sobol_pool[key] = precompute_sobol_pool(
            bounds, n_d, self.n_sobol_screen,
        )

        self.bounds[key]        = bounds
        self.n_d[key]           = n_d
        self.n_fix[key]         = n_fix
        self.fix_indices[key]   = fix_indices
        self.ndim[key]          = ndim
        self.objective_fn[key]  = objective
        self.constraint_fn[key] = constraint

    def evaluate(self, inputs, aux):
        """
        Solve the constrained-CTG current-node NLP.

        Returns `(decision_variables.T, objective, converged)` in the same
        shape as `CurrentConstraintEvaluator.evaluate`.
        """
        key = self.node
        if key is None:
            n = 0 if inputs is None else inputs.shape[0]
            return (
                jnp.zeros((0, n)),
                jnp.zeros((n, 1)),
                jnp.zeros((n, 1), dtype=bool),
            )

        p = _prepare_inputs(inputs, self.graph, key, self.fix_indices[key], self.cfg)
        p = jnp.zeros(self.n_fix[key]).at[:min(p.size, self.n_fix[key])].set(
            p[:self.n_fix[key]]
        )

        x0_batch = pick_x0_batch(
            self.sobol_pool[key], self.screeners[key], p, self.n_starts,
            warmstart=None,
        )
        p_batch = jnp.broadcast_to(
            p.reshape(1, -1), (self.n_starts, self.n_fix[key]),
        )

        result = self.factories[key].solve_batch(x0_batch, p_batch)

        # Solver ran in real-world decision space (bounds are real, inputs
        # are real, surrogates self-scale) — no post-solve inverse
        # transform needed.  Collapse the n_starts axis via `pick_best`
        # so the caller gets one decision per sample (see
        # `CurrentConstraintEvaluator.evaluate` for the rationale).
        best_f, best_c, best_x = pick_best(result)
        params    = jnp.asarray(best_x).reshape(-1, 1)   # (n_d_k, 1)
        objective = jnp.asarray(best_f).reshape(1, 1)    # (1, 1)
        converged = jnp.asarray(best_c).reshape(1, 1)    # (1, 1)
        return params, objective, converged


# =============================================================================
# Local helper — top-N without screening (box-only path has no constraint
# to penalise against, so screening degenerates to `sort by objective`,
# which equals the screener's output when penalty*sum(max(0, g)) is 0).
# Keep a tiny helper so we don't drag a no-op penalty screen through.
# =============================================================================

def _sobol_topn(pool: jnp.ndarray, n_starts: int) -> jnp.ndarray:
    """Return the first `n_starts` points from the Sobol pool — deterministic."""
    return pool[:n_starts]


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
    """
    Box-only feasibility check — classifier minimised over the reduced
    decision space with fixed `inputs` threaded as `p`.

    Returns `(decision_variables, objective, converged)` matching the
    rollout agent's contract.
    """
    if node is None:
        n = 0 if inputs is None else inputs.shape[0]
        return (
            jnp.zeros((0, n)),
            jnp.zeros((n, 1)),
            jnp.zeros((n, 1), dtype=bool),
        )
    return _get_constraint_evaluator(cfg, graph, node).evaluate(inputs, aux)


def current_cost_surrogate(inputs, aux, cfg, graph, node):
    """
    Constrained-CTG on the current node — minimise CTG s.t. classifier <= 0
    with fixed `inputs` threaded as `p`.

    Returns `(decision_variables, objective, converged)`.
    """
    if node is None:
        n = 0 if inputs is None else inputs.shape[0]
        return (
            jnp.zeros((0, n)),
            jnp.zeros((n, 1)),
            jnp.zeros((n, 1), dtype=bool),
        )
    return _get_cost_evaluator(cfg, graph, node).evaluate(inputs, aux)
