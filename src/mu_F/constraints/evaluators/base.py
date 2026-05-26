"""
Base class + shared helpers for septal-backed constraint evaluators.

Every concrete evaluator (CTG, backward, forward, current, ...) inherits from
`BaseEvaluator` and provides three pieces:

  - `_keys()`               — what sub-problems to iterate (successors,
                              predecessors, just `[node]`, graph-wide, ...)
  - `_build_for_key(key)`   — build and cache the stable state for one
                              sub-problem (factory, screener, sobol pool,
                              wrapped callables, shapes)
  - `evaluate(...)`         — the pmap thread body; pure function of its
                              traced arguments, closing only over `self`

The base `__init__` reads the common cfg knobs, initialises the per-key
state dicts, then drives the `_build_for_key` loop.  Subclass state lives
on `self` under those dicts so the instance has a stable id() — necessary
for pmap to cache the compiled thread across calls.

All compile-relevant work happens in `__init__`.  `evaluate` does only
pure JAX operations on traced arrays (and on the precomputed JAX arrays
stored on `self`), which is what makes the JIT cache hit on every call
after the first.

Module-level helpers:
  - `build_factory`               ParametricSQPFactory assembly
  - `build_penalty_screener`      L1-augmented multi-start screener
  - `precompute_sobol_pool`       fixed Sobol draws for the screener
  - `pick_best`                   argmin-where-viable over a flat SQPResult
                                  (used by diagnostic scripts; evaluators
                                  pick best inside `solve_integer_nlp`)
"""
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


# =============================================================================
# Module-level helpers
# =============================================================================

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
    Assemble a `ParametricSQPFactory` for one sub-problem.

    Defaults:
      - `constraint_lhs = -inf`  (one-sided inequality)
      - `constraint_rhs =  0`    (feasibility: g(x, p) <= 0)

    Set both explicitly to encode a two-sided or equality constraint.

    The `objective` / `constraint` callables must have signature `f(x, p)`
    where `x` has shape `(n_decision,)` and `p` has shape `(n_params,)`.
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

    Returns a `screener(sobol_pool, p) -> scores` callable that scans
    across the Sobol pool:

        score(x, p) = objective(x, p) + penalty * sum(max(0, constraint(x, p)))

    Low scores correspond to points that are both feasible (small
    violation) and have low objective — good multi-start seeds.

    Calling pattern the thread should use:
        scores = screener(sobol_pool, p)           # shape (N_pool,)
        best   = sobol_pool[jnp.argsort(scores)[:n_starts]]

    
    """
    penalty = float(penalty)

    def _score(x, p):
        obj = jnp.asarray(objective(x, p)).reshape(())
        if constraint is None:
            return obj                                    # box-only problem
        g    = jnp.asarray(constraint(x, p)).reshape(-1)
        viol = jnp.maximum(0.0, penalty * g).sum()
        return obj + viol

    def screener(sobol_pool, p):
        # carry = () — per-row scores are independent, no state needed.
        # lax.scan stacks the per-step scalar outputs into a (N_pool,) array.
        def _step(_carry, x):
            return _carry, _score(x, p)
        _, scores = lax.scan(_step, (), sobol_pool)
        return scores

    return screener


def precompute_sobol_pool(bounds, n_d: int, n_sobol: int, seed: int = 42) -> jnp.ndarray:
    """
    Draw a fixed, reproducible Sobol pool for screening.

    Stored on the evaluator instance at construction; passed as an argument
    to the jit'd screener so the cache keys only on abstract shape.
    """
    return generate_initial_guess(n_sobol, n_d, bounds, seed=seed)


def parallel_thread(thread_fn, *, in_axes, devices, use_vmap: bool):
    """
    Pick between `pmap(thread_fn, devices=devices)` and `jit(vmap(thread_fn))`
    based on a single runtime flag.

    Both paths expose the same `(*args) -> pytree` signature with axis-0
    mapping semantics, so the caller's padding / masking / chunk-loop logic
    doesn't care which is returned.

    `pmap` genuinely distributes across `devices` (real multi-device
    parallelism, at the cost of a device-to-host gather at the end).

    `jit(vmap(...))` stays on a single device — XLA can still parallelise
    internally when the compute graph supports it, but there's no thread-
    args/gather round-trip.  On a single-host CPU with our small scalar-
    output threads, this is often strictly faster.

    Shape compile-cache behaviour is unchanged: vmap still keys on abstract
    argument shape, so callers that pad to a fixed batch width keep the
    cache warm in exactly the same way as the pmap path.
    """
    # Inner vmap fuses the per-shard sample loop into the compiled program,
    # so `shard_dispatch` can do one dispatch instead of a Python chunk loop.
    inner = jax.vmap(thread_fn, in_axes=in_axes, out_axes=0)
    if use_vmap:
        return jax.jit(jax.vmap(inner, in_axes=in_axes, out_axes=0))
    from jax import pmap
    return pmap(inner, in_axes=in_axes, out_axes=0, devices=devices)


def shard_dispatch(pmap_fn, padded_inputs, *, W):
    """
    Reshape `(total, *trailing) -> (W, total//W, *trailing)`, dispatch through
    `pmap_fn`, and reshape outputs back to `(total, *trailing)`.

    `pmap_fn` is expected to come from `parallel_thread` / `cached_parallel_thread`
    (so it already wraps `pmap(vmap(thread_fn))`).  Caller is responsible for
    padding `padded_inputs` so the leading axis is a multiple of `W`.

    Returns the same pytree type `pmap_fn` returns (tuple or single array),
    with its leading two axes collapsed.
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


def cached_parallel_thread(owner, attr, thread_fn, *, in_axes, devices, use_vmap: bool):
    """
    Memoised wrapper around `parallel_thread`.

    `jax.pmap(...)` allocates per-call dispatch state (incl. Mach-port-name-
    space slots on macOS) that is NOT reclaimed when the wrapper is GC'd.
    Rebuilding the wrapper every DEUS iteration leaks ~80–130 slots/call,
    which crosses the per-process limit (~262k) after ~90 minutes and the
    kernel kills the process with EXC_RESOURCE.

    Build the wrapper once per `(owner, n_devices, use_vmap)` triple and
    reuse it.  `owner` must be the cached evaluator instance so the cache
    lives as long as the evaluator does (one entry per `(graph, node)`).
    """
    cache = getattr(owner, attr, None)
    if cache is None:
        cache = {}
        setattr(owner, attr, cache)
    key = (len(devices), use_vmap, in_axes)
    fn = cache.get(key)
    if fn is None:
        fn = parallel_thread(thread_fn, in_axes=in_axes, devices=devices, use_vmap=use_vmap)
        cache[key] = fn
    return fn


def skip_if_masked(mask, real_fn):
    """
    Conditionally execute `real_fn()` under a scalar boolean mask.

    Used inside pmap threads to skip the SQP solve on padded lanes.  When
    `mask` is True the real branch runs; when False, a zero-filled pytree
    matching the real branch's abstract output is returned instead.

    Shapes and dtypes of the dummy branch are discovered via
    `jax.eval_shape` so the caller just writes `real_fn` once — no
    separate "dummy" declaration, no risk of the two branches drifting
    out of sync.

    Behaviour under pmap: each device evaluates its own `lax.cond`
    independently, so padded lanes genuinely skip the solve.  pmap still
    synchronises on the slowest lane, so wall-time doesn't drop below
    the real-lanes' critical path — but cache stability (fixed pmap
    width) is the win we're after here.
    """
    out_struct = jax.eval_shape(real_fn)
    dummy_fn = lambda: jax.tree_util.tree_map(
        lambda s: jnp.zeros(s.shape, s.dtype), out_struct,
    )
    return lax.cond(mask, real_fn, dummy_fn)


def _viable_mask(result: SQPResult, factory, feasibility_tol: float):
    """Per-start viability mask: KKT-feasibility within tol AND iterate in bounds.

    Used by `pick_best` below.  Drops the convergence-based filter so
    non-KKT-converged-but-feasible iterates remain candidates — septal's
    line search guarantees the iterate is at worst as good in merit as
    `x0`, so a feasible final iterate is still usable even when `max_iter`
    truncated the solve.
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


# =============================================================================
# Base class
# =============================================================================

class BaseEvaluator(ABC):
    """
    Stateful, stable-id evaluator for septal-backed NLP sub-problems.

    One instance per `(cfg, graph, node)` triple — all factories, screeners
    and Sobol pools are built once in `__init__`.  The pmap thread body lives
    on `evaluate(...)`; since the instance id is stable across calls, pmap
    caches the compiled thread and only re-uses it.

    Subclasses implement:

      - `_keys()`                 — list of sub-problem keys to iterate
                                    (successors, predecessors, `[node]`, ...)
      - `_build_for_key(key)`     — populate per-key state on `self`
                                    (factory, screener, sobol pool, wrapped
                                    callables, shapes).  Called once per
                                    `_keys()` entry from `__init__`.
      - `evaluate(...)`           — the pmap thread body.  Argument signature
                                    varies per evaluator; documented in the
                                    concrete subclass.  Must be a pure
                                    function of its traced inputs and `self`.

    Common state populated in `__init__`:

      self.cfg, self.graph, self.node
      self.n_starts, self.feasibility_tol, self.optimality_tol, self.max_iter,
      self.n_sobol_screen, self.screen_penalty
      self.factories : dict[key -> ParametricSQPFactory]
      self.screeners : dict[key -> Callable]
      self.sobol_pool: dict[key -> jnp.ndarray]

    `cfg.solvers.standardised` is deprecated — every surrogate stored on
    the graph is self-scaling, and evaluators always operate in real-world
    units.  Setting the flag to `True` still works (the value is read to
    warn) but has no effect.
    """

    def __init__(self, cfg, graph, node):
        self.cfg   = cfg
        self.graph = graph
        self.node  = node

        # Cached scalar knobs — pulled out of OmegaConf once.  Reading
        # `cfg.solvers.*` inside a tight loop in `evaluate` would add
        # measurable Python overhead on every call.
        self.n_starts        = int(cfg.solvers.n_starts)
        self.feasibility_tol = float(cfg.solvers.feasibility_tol)
        self.optimality_tol  = float(cfg.solvers.optimality_tol)
        self.max_iter        = int(cfg.solvers.max_iter)
        self.n_sobol_screen  = int(cfg.solvers.n_sobol_screen)
        self.screen_penalty  = float(getattr(cfg.solvers, "screen_penalty", 1000.0))

        # Deprecation warning for legacy configs that set the flag.  Fires
        # once per evaluator construction; harmless otherwise.  Remove the
        # warning (and the field) once every yaml has been cleaned up.
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

        # SQP non-convergence counters.  Plain Python ints, accumulated by
        # entry functions via `_record_sqp_outcome(...)` — not JAX-traced,
        # so they don't affect the pjit cache.
        self.n_sqp_calls:   int = 0
        self.n_sqp_failed:  int = 0
        self._last_warn_at: int = 0

        # Drive the build loop.  After this completes, the instance holds all
        # compile-relevant state and its __call__ / evaluate can safely pmap.
        for key in self._keys():
            self._build_for_key(key)

    def _record_sqp_outcome(
        self,
        success_flags,
        *,
        node_label: str = None,
        warn_every: int = 500,
        warn_threshold: float = 0.5,
    ) -> None:
        """Accumulate SQP non-convergence stats; emit a throttled warning.

        Caller filters real lanes (pass `flags[:n_real]`).  Warning fires
        each time `n_sqp_calls` crosses a `warn_every` boundary, only when
        the cumulative rate is at or above `warn_threshold`.
        """
        flags = np.asarray(success_flags).reshape(-1)
        self.n_sqp_calls  += int(flags.size)
        self.n_sqp_failed += int((~flags).sum())

        if self.n_sqp_calls >= self._last_warn_at + warn_every:
            rate = self.n_sqp_failed / max(self.n_sqp_calls, 1)
            if rate >= warn_threshold:
                label = node_label or f"node={self.node}"
                logging.warning(
                    f"SQP non-conv {label} ({type(self).__name__}): "
                    f"{self.n_sqp_failed}/{self.n_sqp_calls} ({rate*100:.1f}%) "
                    f"— consider bumping cfg.solvers.max_iter / n_starts."
                )
            self._last_warn_at = self.n_sqp_calls

    # --- contract --------------------------------------------------------

    @abstractmethod
    def _keys(self) -> list:
        """
        Return the list of sub-problem keys this evaluator iterates over.

        Examples:
          CTGEvaluator              -> list(graph.successors(node))
          ForwardEvaluator          -> list(graph.predecessors(node))
          CurrentCostEvaluator      -> [node]
          PostProcessUpperLevel     -> [None]   (graph-wide)
        """

    @abstractmethod
    def _build_for_key(self, key) -> None:
        """
        Populate `self.factories[key]`, `self.screeners[key]`,
        `self.sobol_pool[key]` plus any per-key shape/indices state.

        Called once per key from `__init__`.  Should close only over static
        per-(cfg, graph, node, key) data so the stored factory / screener
        objects have stable identities across all subsequent `evaluate`
        calls.
        """

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """
        Pmap thread body.  Argument signature is evaluator-specific.

        Must be a pure function of:
          - its (traced) arguments, and
          - the static state stored on `self` at construction.

        No new Python closures, no fresh jit / pmap objects, no partial()
        construction — anything that would change function identity breaks
        the compile cache.
        """
