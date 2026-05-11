"""
Backward coupling constraint evaluator (box-only feasibility check).

For each successor of `node`, solves

    min  classifier_succ(x, succ_input)       (live-set feasibility score)
    s.t. lb <= x <= ub                        (reduced box; no general constraints)

where `x` is the successor's reduced decision vector (design + non-current-
edge inputs + aux) and `succ_input` is the current node's output slice feeding
that edge, threaded as the parametric `p` argument.

Also handles the graph-wide post-processing variant (one NLP over the
graph-level decision space with the upper-level classifier as objective),
dispatched when `graph.graph["solve_post_processing_problem"]` is set.

Exports `construct_solver` / `solve` — thin box-only aliases for
`construct_constrained_solver` / `solve_constrained` with `constraint=None`.
Kept because callers elsewhere import these names for box-only problems.

### BaseEvaluator port

Per-successor state (factory, sobol pool, reduced bounds) is built once in
`_build_for_key` and reused across every thread call.  The `succ_input` is
threaded as the parametric `p` so one `ParametricSQPFactory` per successor
serves all samples — the JIT cache stays warm.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import devices

from mu_F.constraints.evaluators.base import (
    BaseEvaluator,
    build_factory,
    cached_parallel_thread,
    parallel_thread,
    pick_best,
    precompute_sobol_pool,
    shard_dispatch,
    skip_if_masked,
)
from mu_F.constraints.utils import (
    mask_classifier,
    get_successor_inputs,
    shaping_function,
    pad_to_multiple,
    batch_mask,
    poison_padded,
)
from mu_F.solvers.septal import (
    construct_constrained_solver,
    solve_constrained,
)


__all__ = [
    # Box-only solver aliases — used by callers that want `g=None`.
    "construct_solver",
    "solve",
    # Stateful backward evaluator + top-level entry point.
    "BackwardEvaluator",
    "backward_constraint_evaluator",
]


# ---------------------------------------------------------------------------
# Box-only solver aliases.  Post-2b, `construct_solver` and `solve` are thin
# wrappers over septal with `constraint_func=None`.  Preserved here because
# callers elsewhere import these names.
# ---------------------------------------------------------------------------

def construct_solver(objective_func, bounds, tol, sobol_pts=None):
    """Box-only NLP solver — thin wrapper over septal with `g=None`."""
    return construct_constrained_solver(
        objective_func,
        constraint_func=None,
        bounds=bounds,
        tol=tol,
        sobol_pts=sobol_pts,
    )


def solve(solver, initial_guesses):
    """Alias for `solve_constrained` — unified dict-returning driver."""
    return solve_constrained(solver, initial_guesses)


# =============================================================================
# BackwardEvaluator — one instance per (cfg, graph, node)
# =============================================================================

class BackwardEvaluator(BaseEvaluator):
    """
    Stateful backward feasibility evaluator.

    Two modes, decided at construction from
    `graph.graph["solve_post_processing_problem"]`:

      - **graph-wide** (`_keys()` yields `[None]`): one box-only NLP over
        the graph-level decision space with the lower-level post-process
        classifier as objective.

      - **node-local** (`_keys()` yields `list(successors(node))`): one
        box-only NLP per successor of `node`.  The `succ_input` threaded
        as `p`.  Only the per-successor feasibility score is returned;
        warmstart plumbing to CTG was removed (the receiving end never
        consumed it — dead code in `integration.py`).
    """

    def __init__(self, cfg, graph, node):
        self._graph_wide = bool(graph.graph.get("solve_post_processing_problem", False))

        # Per-key static state — keyed by successor id, or `None` in
        # graph-wide mode.
        self.n_fix: dict         = {}
        self.n_d_k: dict         = {}
        self.bounds: dict        = {}
        self.input_indices: dict = {}
        self.aux_indices: dict   = {}
        self.fix_indices: dict   = {}
        self.ndim: dict          = {}
        self.objective_fn: dict  = {}

        super().__init__(cfg, graph, node)

        # Pin bound methods for stable pmap identity across calls.
        self._thread_node_local = self.evaluate_node_local
        self._thread_graph_wide = self.evaluate_graph_wide

    def evaluate(self, outputs_s, aux_s, mask_s=None):
        if self._graph_wide:
            # graph-wide is a single solve — no pmap, no mask applies.
            return self.evaluate_graph_wide(outputs_s, aux_s)
        if mask_s is None:
            # Legacy single-shot entry — treat as a real lane.
            mask_s = jnp.asarray(True)
        return self.evaluate_node_local(outputs_s, aux_s, mask_s)

    def _keys(self) -> list:
        if self._graph_wide:
            return [None]
        if self.node is None:
            return []
        return list(self.graph.successors(self.node))

    def _build_for_key(self, key) -> None:
        if self._graph_wide:
            self._build_graph_wide(key)
        else:
            self._build_node_local(key)

    # ------------------------------------------------------------------
    # Build — node-local (per-successor)
    # ------------------------------------------------------------------

    def _build_node_local(self, succ: int) -> None:
        """Per-successor reduced-space box NLP — `p` is the succ_input."""
        n_d_succ = self.graph.nodes[succ]['n_design_args']
        input_indices = np.array(
            [n_d_succ + inp for inp in self.graph.edges[self.node, succ]['input_indices']],
            dtype=int,
        )
        aux_indices = np.array(
            self.graph.edges[self.node, succ]['auxiliary_indices'],
            dtype=int,
        )
        fix_indices = np.hstack([input_indices, aux_indices]).astype(int)
        ndim = (
            self.graph.nodes[succ]['n_design_args']
            + self.graph.nodes[succ]['n_input_args']
            + self.graph.graph['n_aux_args']
        )
        n_fix = int(len(fix_indices))
        n_d_k = ndim - n_fix

        # Bounds in real-world units; classifier self-scales.
        decision_bounds = self.graph.nodes[succ]["extendedDS_bounds"].copy()
        lb = jnp.delete(
            jnp.asarray(decision_bounds[0]), fix_indices, axis=1,
        ).reshape(-1)
        ub = jnp.delete(
            jnp.asarray(decision_bounds[1]), fix_indices, axis=1,
        ).reshape(-1)
        bounds = [lb, ub]

        classifier         = self.graph.nodes[succ]["classifier"]
        wrapped_classifier = mask_classifier(classifier, ndim, input_indices, aux_indices)

        def objective(x, p):
            return wrapped_classifier(x, p.reshape(1, -1)).squeeze()

        self.factories[succ] = build_factory(
            objective, None, bounds,
            n_decision=n_d_k,
            n_params=n_fix,
            n_constraints=0,
            feasibility_tol=self.feasibility_tol,
            optimality_tol=self.optimality_tol,
            max_iter=self.max_iter,
        )
        self.sobol_pool[succ] = precompute_sobol_pool(
            bounds, n_d_k, self.n_sobol_screen,
        )

        self.n_fix[succ]         = n_fix
        self.n_d_k[succ]         = n_d_k
        self.bounds[succ]        = bounds
        self.input_indices[succ] = input_indices
        self.aux_indices[succ]   = aux_indices
        self.fix_indices[succ]   = fix_indices
        self.ndim[succ]          = ndim
        self.objective_fn[succ]  = objective

    # ------------------------------------------------------------------
    # Build — graph-wide (post-process lower classifier)
    # ------------------------------------------------------------------

    def _build_graph_wide(self, key) -> None:
        """Graph-wide box NLP — `p` is the fixed graph-level aux inputs."""
        n_d   = self.graph.graph['n_design_args']
        n_aux = self.graph.graph['n_aux_args']
        total_ind = np.arange(n_d + n_aux)
        dec_ind   = np.array(self.graph.graph['post_process_decision_indices'])
        fix_ind   = np.delete(total_ind, dec_ind).astype(int)

        lb = jnp.hstack([jnp.array(b[0]).reshape(-1)
                         for b in self.graph.graph['bounds'] if b[0] != 'None'])
        ub = jnp.hstack([jnp.array(b[1]).reshape(-1)
                         for b in self.graph.graph['bounds'] if b[1] != 'None'])
        # Graph-wide bounds in real units; upper-level classifier self-scales.
        bounds_full = [lb, ub]

        lb_red = jnp.delete(bounds_full[0], fix_ind)
        ub_red = jnp.delete(bounds_full[1], fix_ind)
        bounds = [lb_red, ub_red]
        ndim = n_d + n_aux
        n_d_k = int(lb_red.size)
        n_fix = int(len(fix_ind))

        classifier         = self.graph.graph['post_process_lower_classifier']
        wrapped_classifier = mask_classifier(
            classifier, ndim, fix_ind, np.empty((0,)).astype(int),
        )

        # Maximisation in casadi convention — negate so we minimise.
        def objective(x, p):
            return -wrapped_classifier(x, p.reshape(1, -1)).squeeze()

        self.factories[key] = build_factory(
            objective, None, bounds,
            n_decision=n_d_k,
            n_params=n_fix,
            n_constraints=0,
            feasibility_tol=self.feasibility_tol,
            optimality_tol=self.optimality_tol,
            max_iter=self.max_iter,
        )
        self.sobol_pool[key] = precompute_sobol_pool(
            bounds, n_d_k, self.n_sobol_screen,
        )

        self.n_fix[key]         = n_fix
        self.n_d_k[key]         = n_d_k
        self.bounds[key]        = bounds
        self.fix_indices[key]   = fix_ind
        self.ndim[key]          = ndim
        self.objective_fn[key]  = objective

    # ------------------------------------------------------------------
    # Thread entry points — pmap targets
    # ------------------------------------------------------------------

    def evaluate_node_local(self, outputs_s, aux_s, mask_s):
        """
        Node-local thread body.

        Parameters
        ----------
        outputs_s : (N_uncertainty, N_output_dim)
        aux_s     : (N_uncertainty, N_aux)
        mask_s    : scalar bool — True on real lanes, False on padded lanes

        Returns
        -------
        shaped_evals : (1, N_successors) after shaping_function sign flip.
                       Zero-filled for padded lanes (then poisoned to NaN
                       at the top level).
        """
        def real():
            succ_inputs = get_successor_inputs(self.graph, self.node, outputs_s)

            evals = []
            for succ in self._keys():
                # `y` passed through in real units; classifier self-scales.
                y = succ_inputs[succ][0]

                # Box-only — no constraint to penalise, so take the first
                # n_starts Sobol points directly.
                x0_batch = self.sobol_pool[succ][:self.n_starts]
                p_batch = jnp.broadcast_to(
                    y.reshape(1, -1), (self.n_starts, self.n_fix[succ]),
                )

                result = self.factories[succ].solve_batch(x0_batch, p_batch)
                best_f, _, _ = pick_best(result, self.factories[succ], self.feasibility_tol)

                evals.append(best_f.reshape(-1, 1))

            return shaping_function(jnp.hstack(evals), self.cfg)

        return skip_if_masked(mask_s, real)

    def evaluate_graph_wide(self, outputs_s, aux_s):
        """
        Graph-wide thread body.

        The caller passes `outputs_s` in the slot that plays the role of
        the graph-level fixed inputs.  Returns `shaped_evals`.
        """
        key = None
        if outputs_s.ndim < 2:
            outputs_s = outputs_s.reshape(-1, 1)
        y = outputs_s.reshape(-1)[: self.n_fix[key]]
        y = jnp.zeros(self.n_fix[key]).at[:y.size].set(y)

        x0_batch = self.sobol_pool[key][:self.n_starts]
        p_batch = jnp.broadcast_to(
            y.reshape(1, -1), (self.n_starts, self.n_fix[key]),
        )

        result = self.factories[key].solve_batch(x0_batch, p_batch)
        best_f, _, _ = pick_best(result, self.factories[key], self.feasibility_tol)

        # Maximisation — return -shaping_function so sign matches legacy.
        return -shaping_function(best_f.reshape(-1, 1), self.cfg)


# =============================================================================
# Evaluator cache + top-level entry point
# =============================================================================

_BACKWARD_EVALUATOR_CACHE: dict = {}


def _get_evaluator(cfg, graph, node) -> BackwardEvaluator:
    key = (id(graph), node, bool(graph.graph.get("solve_post_processing_problem", False)))
    evaluator = _BACKWARD_EVALUATOR_CACHE.get(key)
    if evaluator is None:
        evaluator = BackwardEvaluator(cfg, graph, node)
        _BACKWARD_EVALUATOR_CACHE[key] = evaluator
    return evaluator


def backward_constraint_evaluator(outputs, aux, cfg, graph, node):
    """
    Top-level backward feasibility evaluator — threads samples across CPU
    devices via pmap.

    Returns
    -------
    (evaluations, None)
        evaluations : (N_batch, N_successors) after sign-flipping.

    A second element is preserved in the return tuple for contract parity
    with the legacy call sites (`evals, _ = ...`), but it is now
    unconditionally `None`: the warmstart plumbing that previously filled
    this slot was dropped because the consumer (the CTG pass in
    `integration.py`) built but never forwarded the dict to
    `cost_to_go_evaluator`.
    """
    evaluator = _get_evaluator(cfg, graph, node)

    if evaluator._graph_wide:
        thread_out = evaluator.evaluate_graph_wide(outputs, aux)
        n_batch = outputs.shape[0]
        return jnp.broadcast_to(thread_out.reshape(1, -1), (n_batch, thread_out.size)), None

    if evaluator._keys() == []:
        # No successors — nothing to do.
        return jnp.zeros((outputs.shape[0], 1)), None

    cpu_devs = list(devices('cpu'))
    W = min(int(cfg.max_devices), len(cpu_devs))
    n_real = outputs.shape[0]

    aux_expanded = jnp.repeat(
        jnp.expand_dims(aux, axis=1), outputs.shape[1], axis=1,
    )
    padded_out, _, _ = pad_to_multiple(outputs, W, axis=0)
    padded_aux, _, _ = pad_to_multiple(aux_expanded, W, axis=0)
    total = padded_out.shape[0]
    mask = batch_mask(n_real, total)

    devs = cpu_devs[:W]
    pmap_fn = cached_parallel_thread(
        evaluator,
        '_pmap_cache',
        evaluator._thread_node_local,
        in_axes=(0, 0, 0),
        devices=devs,
        use_vmap=bool(getattr(cfg.solvers, "use_vmap", False)),
    )
    full_evals = shard_dispatch(pmap_fn, (padded_out, padded_aux, mask), W=W)
    full_evals = poison_padded(full_evals, mask, fill=jnp.nan)
    return full_evals[:n_real], None
