"""
Cost-to-go (CTG) evaluator.

For each successor of `node`, solves

    min   ctg_surrogate(x, succ_input)                (regression)
    s.t.  classifier(x, succ_input) <= 0              (live-set feasibility)
          lb <= x <= ub                               (reduced box)

where `x` is the successor's reduced decision vector (design + non-current-
edge inputs + aux) and `succ_input` is the current node's output slice
feeding that edge, held fixed.

### p-lifting

The per-sample `succ_input` is threaded through as the `p` parameter of
septal's `ParametricNLPProblem`, not baked into a Python closure.  One
`ParametricSQPFactory` per successor is built in `_build_for_key`; every
subsequent call feeds new `(x_batch, p_batch)` to the same compiled scan
body.  This is what makes the JIT cache hit across calls.

### Multi-start

Initial guesses for the SQP are chosen by an L1-penalty screen over a
pre-generated Sobol pool (see `base.build_penalty_screener`) — matching
the behaviour of the old `l1_sample_initial_guess` path.

Warmstart plumbing (accepting a per-successor `(N_batch, n_d_k)` dict from
the backward pass) was removed in the port: the call site in
`integration.py` built the dict but never forwarded it into
`cost_to_go_evaluator`, so the whole path was dead code.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import devices

from mu_F.constraints.evaluators.base import (
    BaseEvaluator,
    build_factory,
    build_penalty_screener,
    cached_parallel_thread,
    parallel_thread,
    pick_best_per_scenario,
    pick_x0_batch,
    precompute_sobol_pool,
    shard_dispatch,
    skip_if_masked,
)
from mu_F.constraints.utils import (
    get_successor_inputs,
    mask_classifier,
    pad_to_multiple,
    batch_mask,
    poison_padded,
)


__all__ = ["CTGEvaluator", "cost_to_go_evaluator"]


# =============================================================================
# CTG evaluator — one instance per (cfg, graph, node)
# =============================================================================

class CTGEvaluator(BaseEvaluator):
    """
    Stateful CTG evaluator.

    Owns the factory + screener + sobol pool for every successor.  The
    thread entry point `evaluate(outputs_s, aux_s)` is pinned to a stable
    bound-method attribute in `__init__` so pmap's compile cache keys on
    the same callable identity across every call.
    """

    def __init__(self, cfg, graph, node):
        # Per-successor static state populated in `_build_for_key`.
        self.n_fix: dict         = {}
        self.n_d_k: dict         = {}
        self.bounds: dict        = {}
        self.input_indices: dict = {}
        self.aux_indices: dict   = {}
        self.fix_indices: dict   = {}
        self.ndim: dict          = {}
        # Kept for inspection / testing — the actual solve goes through
        # `self.factories[succ]` which wraps these in an NLP problem.
        self.objective_fn: dict  = {}
        self.constraint_fn: dict = {}

        super().__init__(cfg, graph, node)

        # Pin bound methods for stable id() across calls.  Without pinning,
        # `instance.evaluate` creates a fresh bound-method object on every
        # access, breaking pmap's compile cache.
        self._thread = self.evaluate

    # ------------------------------------------------------------------
    # BaseEvaluator contract
    # ------------------------------------------------------------------

    def _keys(self) -> list:
        return list(self.graph.successors(self.node))

    def _build_for_key(self, succ: int) -> None:
        """
        Compute stable state for one successor.  Called once per successor
        from `BaseEvaluator.__init__`.

        Everything stored on `self` here is part of the compile cache's
        identity — no per-call closures, no rebuilt factories.
        """
        # --- indices, ndim, reduced decision dim --------------------------
        n_d = self.graph.nodes[succ]['n_design_args']
        input_indices = np.array(
            [n_d + inp for inp in self.graph.edges[self.node, succ]['input_indices']],
            dtype=int,
        )
        aux_indices = np.array(
            [n_d + idx for idx in self.graph.edges[self.node, succ]['auxiliary_indices']],
            dtype=int,
        )
        fix_indices = np.hstack([input_indices, aux_indices]).astype(int)
        ndim = (
            self.graph.nodes[succ]['n_design_args']
            + self.graph.nodes[succ]['n_input_args']
            + self.graph.graph['n_aux_args']
        )
        n_fix = len(fix_indices)
        n_d_k = ndim - n_fix

        # --- reduced-space bounds ----------------------------------------
        # Bounds stay in real-world units; the classifier + CTG surrogates
        # self-scale internally.
        decision_bounds = self.graph.nodes[succ]['extendedDS_bounds'].copy()
        lb = jnp.delete(
            jnp.asarray(decision_bounds[0]), fix_indices, axis=1,
        ).reshape(-1)
        ub = jnp.delete(
            jnp.asarray(decision_bounds[1]), fix_indices, axis=1,
        ).reshape(-1)
        bounds = [lb, ub]

        # --- live-callable wrappers --------------------------------------
        # `mask_classifier` returns a cached jit'd callable keyed on (fn,
        # ndim, fix_ind, aux_ind) — stable identity per (graph, node, succ).
        ctg_surrogate = self.graph.nodes[succ]['ctg_surrogate']
        classifier    = self.graph.nodes[succ]['classifier']
        wrapped_ctg   = mask_classifier(ctg_surrogate, ndim, input_indices, aux_indices)
        wrapped_clf   = mask_classifier(classifier,    ndim, input_indices, aux_indices)

        # --- (x, p) parametric objective + constraint -------------------
        # The CTG surrogate may return a per-uncertainty-sample vector —
        # collapse to scalar via sum (matches the casadi path's
        # `_ensure_scalar_objective`).
        def objective(x, p):
            return wrapped_ctg(x, p.reshape(1, -1)).reshape(-1).sum()

        def constraint(x, p):
            return wrapped_clf(x, p.reshape(1, -1)).reshape(-1)

        # --- factory + screener + sobol pool -----------------------------
        self.factories[succ] = build_factory(
            objective, constraint, bounds,
            n_decision=n_d_k,
            n_params=n_fix,
            n_constraints=1,          # single classifier feasibility scalar
            feasibility_tol=self.feasibility_tol,
            optimality_tol=self.optimality_tol,
            max_iter=self.max_iter,
        )
        self.screeners[succ] = build_penalty_screener(
            objective, constraint, self.screen_penalty,
        )
        self.sobol_pool[succ] = precompute_sobol_pool(
            bounds, n_d_k, self.n_sobol_screen,
        )

        # --- metadata used at thread-evaluate time ------------------------
        self.n_fix[succ]         = n_fix
        self.n_d_k[succ]         = n_d_k
        self.bounds[succ]        = bounds
        self.input_indices[succ] = input_indices
        self.aux_indices[succ]   = aux_indices
        self.fix_indices[succ]   = fix_indices
        self.ndim[succ]          = ndim
        self.objective_fn[succ]  = objective
        self.constraint_fn[succ] = constraint

    # ------------------------------------------------------------------
    # Shard entry point — pmap target
    # ------------------------------------------------------------------

    def evaluate(self, outputs_s, aux_s, mask_s):
        """
        Shard body — per-scenario CTG via septal's parametric batch.

        Stacks `(start, scenario)` pairs into a single flat batch for one
        `solve_batch` call.  Per-scenario start selection (the L1-penalty
        screen) is `vmap`-ed over the uncertainty axis since each scenario
        gets its own succ_input.

        Parameters
        ----------
        outputs_s : (N_uncertainty, N_output_dim)
        aux_s     : (N_uncertainty, N_aux)
        mask_s    : scalar bool — True on real lanes, False on padded lanes

        Returns
        -------
        (evals, flags) : each shape (N_uncertainty, N_successors)
            `evals[t, k]` is the minimised CTG for scenario t at successor k.
            `flags[t, k]` is the convergence flag for the chosen start.

        Deterministic case studies pass `N_uncertainty == 1`, in which case
        the per-scenario axis is a singleton and the call is equivalent to
        a single design × successor solve.
        """
        def real():
            succ_inputs = get_successor_inputs(self.graph, self.node, outputs_s)

            evals, flags = [], []
            for succ in self._keys():
                ys = succ_inputs[succ]
                if aux_s is not None and aux_s.size > 0:
                    ys = jnp.concatenate([ys, aux_s], axis=-1)
                n_unc = ys.shape[0]
                n_starts = self.n_starts
                sobol = self.sobol_pool[succ]
                screener = self.screeners[succ]

                # Per-scenario screen → (N_unc, n_starts, n_d).
                def _screen_one(y):
                    return pick_x0_batch(sobol, screener, y, n_starts)
                x0_per_scenario = jax.vmap(_screen_one)(ys)
                x0_stacked = x0_per_scenario.reshape(n_unc * n_starts, -1)
                p_stacked  = jnp.repeat(ys, n_starts, axis=0)

                result = self.factories[succ].solve_batch(x0_stacked, p_stacked)
                best_obj, best_viable = pick_best_per_scenario(
                    result, self.factories[succ], self.feasibility_tol,
                    n_unc, n_starts,
                )

                evals.append(best_obj.reshape(-1, 1))
                flags.append(best_viable.reshape(-1, 1))

            return jnp.hstack(evals), jnp.hstack(flags)

        return skip_if_masked(mask_s, real)


# =============================================================================
# Evaluator cache — one CTGEvaluator per (id(graph), node)
# =============================================================================

_CTG_EVALUATOR_CACHE: dict = {}


def _get_evaluator(cfg, graph, node) -> CTGEvaluator:
    """
    Look up or build the `CTGEvaluator` for this `(graph, node)` pair.

    Keyed on `id(graph)` so the cache tracks the graph object's lifetime —
    a new iterate with a fresh graph object gets a fresh evaluator.
    Within one iterate the graph is stable, so DEUS calls reuse the cached
    factories + screeners.
    """
    key = (id(graph), node)
    evaluator = _CTG_EVALUATOR_CACHE.get(key)
    if evaluator is None:
        evaluator = CTGEvaluator(cfg, graph, node)
        _CTG_EVALUATOR_CACHE[key] = evaluator
    return evaluator


# =============================================================================
# Public entry point — called by constraints/constructor.py
# =============================================================================

def cost_to_go_evaluator(outputs, aux, cfg, graph, node):
    """
    Top-level CTG evaluator.  Shards samples across CPU devices via pmap
    at a **fixed width** (`cfg.max_devices`), padding smaller batches up
    with row-0 replicas and masking the padded lanes inside the thread.

    Fixed pmap width means the pmap compile cache keys only on static
    shape, not on batch size — so calls with different real-batch sizes
    reuse the same compiled kernel after the first.

    Parameters
    ----------
    outputs : (N_batch, N_uncertainty, N_output_dim)
    aux     : (N_batch, N_aux)

    Returns
    -------
    evaluations   : (N_batch, N_uncertainty, N_successors) — NaN on any row
                    where the padded-lane mask applied (defensive poison).
    success_flags : (N_batch, N_uncertainty, N_successors)
    """
    evaluator = _get_evaluator(cfg, graph, node)

    # Fixed pmap width clamped to real CPU device count.
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
        evaluator._thread,
        in_axes=(0, 0, 0),
        devices=devs,
        use_vmap=bool(getattr(cfg.solvers, "use_vmap", False)),
    )
    full_evals, full_flags = shard_dispatch(
        pmap_fn, (padded_out, padded_aux, mask), W=W,
    )
    # Defensive poison: NaN on padded rows, False on the converged flag.
    # `poison_padded` reshapes the mask to match the array's rank so 3D
    # `(total, N_unc, N_succ)` broadcasts correctly (naive `mask[:, None]`
    # trips the 1-dimensioned axes against each other and blows up to
    # `(total, total, N_succ)`).
    full_evals = poison_padded(full_evals, mask, fill=jnp.nan)
    full_flags = poison_padded(full_flags, mask, fill=False)
    return full_evals[:n_real], full_flags[:n_real]
