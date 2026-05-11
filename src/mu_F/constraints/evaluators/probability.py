"""
Probabilistic feasibility evaluator.

`BackwardPmapEvaluator` queries each successor's `probability_map` surrogate,
maximising the predicted probability over the successor's own design space
subject to the upstream output as fixed `succ_input`.  The inner SQP is
vmapped over the current node's uncertainty axis so the per-scenario,
per-successor probability is returned (`(N_uncertainty, N_successors)`).

The CTG analogue lives in `cost_to_go.py` — `CTGEvaluator.evaluate` does
the equivalent vmap over the scenario axis, so a single class serves both
deterministic (n=1) and probabilistic (n>1) use without an extra subclass.
Aggregation across scenarios (weighted sum, product over successors) lives
at the call site in `integration.py`.
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
    precompute_sobol_pool,
    skip_if_masked,
)
from mu_F.constraints.utils import (
    mask_classifier,
    get_successor_inputs,
    pad_to_multiple,
    batch_mask,
    poison_padded,
)


__all__ = [
    "BackwardPmapEvaluator",
    "backward_pmap",
]


# =============================================================================
# BackwardPmapEvaluator — one instance per (cfg, graph, node)
# =============================================================================

class BackwardPmapEvaluator(BaseEvaluator):
    """
    Backward feasibility evaluator for probabilistic constraints.
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
        self.objective_fn: dict  = {}

        super().__init__(cfg, graph, node)

        # Pin bound method for stable pmap identity across calls.
        self._thread = self.evaluate

    def _keys(self) -> list:
        if self.node is None:
            return []
        return list(self.graph.successors(self.node))

    # ------------------------------------------------------------------
    # Build 
    # ------------------------------------------------------------------

    def _build_for_key(self, succ: int) -> None:
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

        surrogate         = self.graph.nodes[succ]["probability_map"]
        wrapped_surrogate = mask_classifier(surrogate, ndim, input_indices, aux_indices)

        def objective(x, p):
            # We want to maximise the probability that we are feasible
            return - wrapped_surrogate(x, p.reshape(1, -1)).squeeze()

        self.factories[succ] = build_factory(
            objective, None, bounds,
            n_decision=n_d_k,
            n_params=n_fix,
            n_constraints=0,
            tol=self.tol,
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
    # Thread entry points — pmap targets
    # ------------------------------------------------------------------

    def evaluate(self, outputs_s, aux_s, mask_s):
        """
        Node-local thread body — per-scenario evaluation via septal's
        parametric batch.

        Instead of vmapping the inner SQP over scenarios externally, we
        stack `(start, scenario)` pairs into a single flat batch and let
        `solve_batch` vmap over both axes in one call.

        Parameters
        ----------
        outputs_s : (N_uncertainty, N_output_dim) — current node's per-scenario outputs
        aux_s     : (N_uncertainty, N_aux)
        mask_s    : scalar bool — True on real lanes, False on padded lanes

        Returns
        -------
        probs : (N_uncertainty, N_successors)
            For each (scenario, successor) the maximum achievable downstream
            probability of feasibility, conditioned on this scenario's output.
        """
        def real():
            succ_inputs = get_successor_inputs(self.graph, self.node, outputs_s)

            evals = []
            for succ in self._keys():
                ys = succ_inputs[succ]                 # (N_unc, n_fix)
                n_unc = ys.shape[0]
                n_starts = self.n_starts
                sobol = self.sobol_pool[succ][:n_starts]   # (n_starts, n_d)

                # Stack (start, scenario): row k → start (k % n_starts),
                # scenario (k // n_starts).
                x0_stacked = jnp.tile(sobol, (n_unc, 1))             # (n_unc*n_starts, n_d)
                p_stacked  = jnp.repeat(ys, n_starts, axis=0)        # (n_unc*n_starts, n_fix)

                result = self.factories[succ].solve_batch(x0_stacked, p_stacked)

                # Reshape and argmin per scenario.
                objectives = result.objective.reshape(n_unc, n_starts)
                success    = jnp.asarray(result.success).reshape(n_unc, n_starts)
                ranked     = jnp.where(success, objectives, objectives + 1e10)
                best_idx   = jnp.argmin(ranked, axis=1)              # (n_unc,)
                best_obj   = jnp.take_along_axis(
                    objectives, best_idx[:, None], axis=1,
                ).squeeze(-1)                                        # (n_unc,)

                # objective = -P_feas → negate to recover the probability.
                evals.append((-best_obj).reshape(-1, 1))

            return jnp.hstack(evals)  # (n_unc, N_succ)

        return skip_if_masked(mask_s, real)


# =============================================================================
# Evaluator cache + top-level entry point
# =============================================================================

_BACKWARDPMAP_EVALUATOR_CACHE: dict = {}


def _get_pmap_evaluator(cfg, graph, node) -> BackwardPmapEvaluator:
    key = (id(graph), node)
    evaluator = _BACKWARDPMAP_EVALUATOR_CACHE.get(key)
    if evaluator is None:
        evaluator = BackwardPmapEvaluator(cfg, graph, node)
        _BACKWARDPMAP_EVALUATOR_CACHE[key] = evaluator
    return evaluator


def _drive_pmap(evaluator, outputs, aux, cfg, succ_count_fallback: int = 1):
    """
    Generic driver: chunked-pmap of the per-scenario thread.

    Parameters
    ----------
    evaluator : has `_thread(outputs_s, aux_s, mask_s)` returning shape
        `(N_uncertainty, N_succ)` per design.
    outputs   : (N_batch, N_uncertainty, N_output_dim)
    aux       : (N_batch, N_aux)

    Returns `(N_batch, N_uncertainty, N_succ)`.
    """
    if evaluator._keys() == []:
        # No successors — return ones so the caller's product / sum collapses
        # to a no-op contribution (probability=1, cost=0 picked at call site).
        return jnp.ones((outputs.shape[0], outputs.shape[1], succ_count_fallback))

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

    n_chunks = total // W
    out_chunks  = padded_out.reshape((n_chunks, W) + padded_out.shape[1:])
    aux_chunks  = padded_aux.reshape((n_chunks, W) + padded_aux.shape[1:])
    mask_chunks = mask.reshape(n_chunks, W)

    devs = cpu_devs[:W]
    pmap_fn = cached_parallel_thread(
        evaluator,
        '_pmap_cache',
        evaluator._thread,
        in_axes=(0, 0, 0),
        devices=devs,
        use_vmap=bool(getattr(cfg.solvers, "use_vmap", False)),
    )

    evals_chunks = []
    for i in range(n_chunks):
        evals_chunks.append(pmap_fn(out_chunks[i], aux_chunks[i], mask_chunks[i]))

    full_evals = jnp.concatenate(evals_chunks, axis=0)  # (total, N_unc, N_succ)
    full_evals = poison_padded(full_evals, mask, fill=jnp.nan)
    return full_evals[:n_real]


def backward_pmap(outputs, aux, cfg, graph, node):
    """
    Top-level per-scenario downstream-feasibility evaluator.

    Returns
    -------
    (evaluations, None)
        evaluations : (N_batch, N_uncertainty, N_successors) — maximised
        downstream P_feas per (design, scenario, successor).

    The trailing `None` keeps tuple parity with `backward_constraint_evaluator`.
    """
    evaluator = _get_pmap_evaluator(cfg, graph, node)
    return _drive_pmap(evaluator, outputs, aux, cfg), None
