"""Base class and shared helpers for septal-backed constraint evaluators."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from septal.jax.sqp import ParametricNLPProblem, ParametricSQPFactory, SQPResult

from mu_F.solvers.septal import DEFAULT_SQP_CONFIG
from mu_F.solvers.utilities import generate_initial_guess


__all__ = [
    "BaseEvaluator",
    "build_factory",
    "build_penalty_screener",
    "precompute_sobol_pool",
    "pick_best",
    "skip_if_masked",
    "cached_parallel_thread",
    "shard_dispatch",
]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def build_factory(
    objective: Callable,
    constraint: Optional[Callable],
    bounds,
    n_decision: int,
    n_params: int,
    n_constraints: int,
    feasibility_tol: float,
    optimality_tol: float,
    max_iter: int,
    constraint_lhs=None,
    constraint_rhs=None,
) -> ParametricSQPFactory:
    """
    Assemble a ParametricSQPFactory for one sub-problem.  Defaults to the
    one-sided feasibility constraint g(x, p) <= 0; set constraint_lhs/rhs
    explicitly for two-sided or equality constraints.
    """
    if constraint is None:
        problem = ParametricNLPProblem(
            objective=objective,
            bounds=[jnp.asarray(bounds[0]).reshape(-1),
                    jnp.asarray(bounds[1]).reshape(-1)],
            n_decision=n_decision,
            n_params=n_params,
        )
    else:
        lhs = constraint_lhs if constraint_lhs is not None else jnp.full((n_constraints,), -jnp.inf)
        rhs = constraint_rhs if constraint_rhs is not None else jnp.zeros((n_constraints,))
        problem = ParametricNLPProblem(
            objective=objective,
            constraints=constraint,
            bounds=[jnp.asarray(bounds[0]).reshape(-1),
                    jnp.asarray(bounds[1]).reshape(-1)],
            n_decision=n_decision,
            n_params=n_params,
            constraint_lhs=jnp.asarray(lhs).reshape(-1),
            constraint_rhs=jnp.asarray(rhs).reshape(-1),
            n_constraints=n_constraints,
        )

    sqp_cfg = replace(
        DEFAULT_SQP_CONFIG,
        tol_stationarity=float(optimality_tol),
        tol_feasibility=float(feasibility_tol),
        max_iter=int(max_iter),
    )
    return ParametricSQPFactory(problem, sqp_cfg)


def build_penalty_screener(
    objective: Callable,
    constraint: Callable,
    penalty: float,
) -> Callable:
    """
    Build an L1-penalty-augmented screener for multi-start selection.
    Scores each Sobol point by objective + penalty * constraint violation;
    low scores are feasible, low-objective seeds.
    """
    penalty = float(penalty)

    def _score(x, p):
        obj = jnp.asarray(objective(x, p)).reshape(())
        if constraint is None:
            return obj                                    # box-only problem
        g    = jnp.asarray(constraint(x, p)).reshape(-1)
        viol = jnp.maximum(0.0, penalty * g).sum()
        return obj + viol


    _score_pool = jax.vmap(_score, in_axes=(0, None))

    def screener(sobol_pool, p):
        return _score_pool(sobol_pool, p)

    return screener


def precompute_sobol_pool(bounds, n_d: int, n_sobol: int, seed: int = 42) -> jnp.ndarray:
    """
    Draw a fixed, reproducible Sobol pool for screening.  Stored at
    construction and passed to the jit'd screener so the cache keys only
    on abstract shape.
    """
    return generate_initial_guess(n_sobol, n_d, bounds, seed=seed)


def parallel_thread(thread_fn, *, in_axes, devices, dispatch: str):
    """
    Dispatch per-shard parallelism via pmap, jit(vmap(...)), or a serial
    Python loop over a jit'd body, all sharing the (*args) -> pytree
    contract with axis-0 mapping.
    """
    # Inner vmap fuses the per-shard sample loop into the compiled program.
    inner = jax.vmap(thread_fn, in_axes=in_axes, out_axes=0)

    if dispatch == "pmap":
        return jax.pmap(inner, in_axes=in_axes, out_axes=0, devices=devices)

    if dispatch == "vmap":
        return jax.jit(jax.vmap(inner, in_axes=in_axes, out_axes=0))

    if dispatch == "serial":
        return _serial_dispatch(inner, in_axes=in_axes)

    raise ValueError(
        f"Unknown dispatch={dispatch!r}; expected 'pmap', 'vmap', or 'serial'"
    )


def _serial_dispatch(inner, *, in_axes):
    """
    Sequential Python for-loop over the leading axis: slice each mapped
    arg, call a jit'd single-shard inner per iter, stack the results.
    """
    axes_tuple = in_axes if isinstance(in_axes, tuple) else (in_axes,)
    inner_jit  = jax.jit(inner)

    def _run(*args):
        # Width comes from any batched arg.
        W = next(a.shape[ax] for a, ax in zip(args, axes_tuple) if ax is not None)
        outs = [
            inner_jit(*(a[i] if ax is not None else a
                        for a, ax in zip(args, axes_tuple)))
            for i in range(W)
        ]
        return jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *outs
        )

    return _run


def shard_dispatch(pmap_fn, padded_inputs, *, W):
    """
    Reshape (total, ...) -> (W, total//W, ...), dispatch through pmap_fn,
    and collapse the leading two output axes back to (total, ...).  Caller
    pads padded_inputs so the leading axis is a multiple of W.
    """
    total = padded_inputs[0].shape[0]
    if total % W != 0:
        raise ValueError(
            f"shard_dispatch: leading axis {total} is not a multiple of W={W}"
        )
    n_per_shard = total // W

    sharded = tuple(
        a.reshape((W, n_per_shard) + a.shape[1:]) for a in padded_inputs
    )
    out = pmap_fn(*sharded)

    def _unshard(x):
        return x.reshape((total,) + x.shape[2:])

    if isinstance(out, tuple):
        return tuple(_unshard(x) for x in out)
    return _unshard(out)


def cached_parallel_thread(owner, attr, thread_fn, *, in_axes, devices, dispatch: str):
    """
    Memoised wrapper around parallel_thread.  Built once per
    (owner, n_devices, dispatch, in_axes) and reused rather than rebuilt
    each DEUS iteration.
    """
    cache = getattr(owner, attr, None)
    if cache is None:
        cache = {}
        setattr(owner, attr, cache)
    key = (len(devices), dispatch, in_axes)
    fn = cache.get(key)
    if fn is None:
        fn = parallel_thread(thread_fn, in_axes=in_axes, devices=devices, dispatch=dispatch)
        cache[key] = fn
    return fn


def skip_if_masked(mask, real_fn):
    """
    Conditionally execute real_fn() under a scalar boolean mask, skipping
    the SQP solve on padded lanes.  The zero-filled dummy branch matches
    real_fn's abstract output via jax.eval_shape so the two cannot drift.
    """
    out_struct = jax.eval_shape(real_fn)
    dummy_fn = lambda: jax.tree_util.tree_map(
        lambda s: jnp.zeros(s.shape, s.dtype), out_struct,
    )
    return lax.cond(mask, real_fn, dummy_fn)


def _viable_mask(result: SQPResult, factory, feasibility_tol: float):
    """
    Per-start viability mask: KKT-feasibility within tol and iterate in
    bounds.  No convergence filter, so feasible-but-unconverged iterates
    remain usable candidates for pick_best.
    """
    x = result.decision_variables                              # (..., n_d)
    lb = factory.problem.lb
    ub = factory.problem.ub
    in_bounds = jnp.all((x >= lb) & (x <= ub), axis=-1)        # (...,)
    feas_viol = jnp.asarray(result.kkt_feasibility)            # (...,)
    return (feas_viol <= feasibility_tol) & in_bounds


def pick_best(result: SQPResult, factory, feasibility_tol: float):
    """Argmin-where-viable over a flat multi-start `SQPResult`.

    Returns `(best_objective, best_viable, best_decision_variables)` with the
    multi-start axis removed.  `best_viable` replaces the old `best_success`.
    """
    viable = _viable_mask(result, factory, feasibility_tol)
    rank   = jnp.where(viable, result.objective, result.objective + 1e10)
    best_i = jnp.argmin(rank)
    return (
        result.objective[best_i],
        viable[best_i],
        result.decision_variables[best_i],
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseEvaluator(ABC):
    """Stateful, stable-id evaluator for septal-backed NLP sub-problems.

    One instance per (cfg, graph, node): all factories, screeners and Sobol
    pools are built once in __init__, and because the instance id is stable
    the pmap thread body on evaluate stays cached across calls.  Subclasses
    supply _keys, _build_for_key and evaluate; knobs resolve per-evaluator
    via _resolve_knob with a flat cfg.solvers fallback.

    """

    # Subclasses set this to enable per-evaluator yaml overrides; when None,
    # _resolve_knob falls straight back to the flat cfg.solvers.<knob>.
    _eval_name: Optional[str] = None

    # Sentinel for "no default supplied" — distinct from None.
    _MISSING = object()

    # ---- Private Methods ----

    def _resolve_knob(self, cfg, knob_name: str, *, default=_MISSING):
        """
        Resolve a solver knob, preferring the per-evaluator override
        cfg.solvers.<_eval_name>.<knob>, then the flat cfg.solvers.<knob>,
        then default; raises AttributeError if none are set.
        """
        if self._eval_name is not None:
            sub = getattr(cfg.solvers, self._eval_name, None)
            if sub is not None and hasattr(sub, knob_name):
                return getattr(sub, knob_name)
        if hasattr(cfg.solvers, knob_name):
            return getattr(cfg.solvers, knob_name)
        if default is not self._MISSING:
            return default
        raise AttributeError(
            f"cfg.solvers.{knob_name} not set and no per-evaluator override "
            f"under cfg.solvers.{self._eval_name!r}.{knob_name}"
        )

    # ---- External Methods ----

    def __init__(self, cfg, graph, node):
        self.cfg   = cfg
        self.graph = graph
        self.node  = node

        # Cached scalar knobs — read from OmegaConf once to avoid per-call cost.
        self.n_starts        = int(  self._resolve_knob(cfg, 'n_starts'))
        self.feasibility_tol = float(self._resolve_knob(cfg, 'feasibility_tol'))
        self.optimality_tol  = float(self._resolve_knob(cfg, 'optimality_tol'))
        self.max_iter        = int(  self._resolve_knob(cfg, 'max_iter'))
        self.n_sobol_screen  = int(  self._resolve_knob(cfg, 'n_sobol_screen'))
        self.screen_penalty  = float(self._resolve_knob(cfg, 'screen_penalty', default=1000.0))
        self.integer_backend = str(  self._resolve_knob(cfg, 'integer_backend', default='enumeration'))
        self.bb_max_nodes    = int(  self._resolve_knob(cfg, 'bb_max_nodes', default=0))

        # Deprecation warning for legacy configs that set the standardised flag.
        if bool(getattr(cfg.solvers, "standardised", False)):
            import logging
            logging.getLogger(__name__).warning(
                "cfg.solvers.standardised is deprecated and has no effect — "
                "surrogates on the graph are self-scaling, evaluators always "
                "run in real-world units."
            )

        # Per-key state — subclasses populate these dicts from _build_for_key.
        self.factories: dict  = {}
        self.screeners: dict  = {}
        self.sobol_pool: dict = {}

        # SQP outcome counters — plain Python ints, not JAX-traced.
        self.n_sqp_calls:     int = 0
        self.n_sqp_viable:    int = 0
        self.n_sqp_converged: int = 0
        self._last_warn_at:   int = 0

        # Dispatch state cached once so _build_dispatch_fn is a dict lookup.
        from jax import devices
        avail_devs = list(devices())
        self._dispatch_W       = min(int(cfg.max_devices), len(avail_devs))
        self._dispatch_devices = avail_devs[:self._dispatch_W]
        self._dispatch_mode    = str(cfg.dispatch)

        # Build all compile-relevant per-key state before any evaluate call.
        for key in self._keys():
            self._build_for_key(key)

    # ---- Private Methods ----

    def _build_dispatch_fn(self, thread_fn, *, in_axes):
        """
        Build a cached (*sharded_args) -> pytree wrapper around thread_fn,
        returning (W, fn).  Caller pads inputs to a multiple of W and
        dispatches via shard_dispatch.
        """
        fn = cached_parallel_thread(
            self, '_dispatch_cache', thread_fn,
            in_axes=in_axes, devices=self._dispatch_devices,
            dispatch=self._dispatch_mode,
        )
        return self._dispatch_W, fn

    def _record_sqp_outcome(
        self,
        *,
        viable_flags,
        converged_flags,
        node_label: str = None,
        warn_every: int = 500,
        viable_warn_threshold: float = 0.5,
        converged_warn_threshold: float = 0.5,
    ) -> None:
        """
        Accumulate SQP feasibility and KKT-convergence successes; emit a
        throttled warning whenever either rate drops at or below its
        threshold.  Caller passes only the real lanes (flags[:n_real]).
        """
        vf = np.asarray(viable_flags).reshape(-1)
        cf = np.asarray(converged_flags).reshape(-1)
        assert vf.size == cf.size, "viable / converged flag arrays must align"

        self.n_sqp_calls     += int(vf.size)
        self.n_sqp_viable    += int(vf.sum())
        self.n_sqp_converged += int(cf.sum())

        if self.n_sqp_calls >= self._last_warn_at + warn_every:
            v_rate = self.n_sqp_viable    / max(self.n_sqp_calls, 1)
            c_rate = self.n_sqp_converged / max(self.n_sqp_calls, 1)
            label = node_label or f"node={self.node}"
            if v_rate <= viable_warn_threshold:
                logging.warning(
                    f"SQP feasible {label} ({type(self).__name__}): "
                    f"{self.n_sqp_viable}/{self.n_sqp_calls} "
                    f"({v_rate*100:.1f}%) — feasibility region may be "
                    f"empty / warm-start pool too coarse."
                )
            if c_rate <= converged_warn_threshold:
                logging.warning(
                    f"SQP converged {label} ({type(self).__name__}): "
                    f"{self.n_sqp_converged}/{self.n_sqp_calls} "
                    f"({c_rate*100:.1f}%) — consider bumping "
                    f"cfg.solvers.max_iter / n_starts."
                )
            self._last_warn_at = self.n_sqp_calls

    # ---- Base Methods ----

    @abstractmethod
    def _keys(self) -> list:
        """
        Return the list of sub-problem keys this evaluator iterates over
        (successors, predecessors, [node], or [None] for graph-wide).
        """

    @abstractmethod
    def _build_for_key(self, key) -> None:
        """
        Populate the per-key factory, screener, Sobol pool and shape state.
        Called once per key from __init__; closes only over static
        per-(cfg, graph, node, key) data so the stored objects keep stable ids.
        """

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """
        Pmap thread body, evaluator-specific in signature.  Must be a pure
        function of its traced arguments and the static state on self — any
        fresh closure / jit / pmap object would break the compile cache.
        """
