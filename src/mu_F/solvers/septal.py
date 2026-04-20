"""
septal.py — constrained NLP solver backend for the mu_F evaluators.

Wraps `septal.jax.sqp.ParametricSQPFactory` as a drop-in replacement for the
ad-hoc jaxopt / CasADi+IPOPT paths previously used by the constraints
evaluators.  The public surface mirrors `construct_solver` / `solve` in
`jax_evaluator` so the two backends can coexist during the phased migration:

    solver = construct_constrained_solver(f, g, bounds, tol)
    result = solve_constrained([solver], [initial_guesses])

`result` is a dict with keys `objective`, `error`, `converged`, `params`.

All objective / constraint callables handed in must be pure JAX — the live
`classifier` and `ctg_surrogate` callables attached to graph nodes by
`mu_F.integration` are already pure-JAX, so the evaluators can pass them
through directly without reconstruction from serialised weights.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from septal.jax.sqp import (
    ParametricNLPProblem,
    ParametricSQPFactory,
    SQPConfig,
)

# Default SQP settings — tuned during Phase 1 validation on the markov_process
# CTG sub-problem (septal vs IPOPT agreement to 8 sig figs). `max_iter=300` is
# well above the iteration count needed at our typical n_d <= 50; exact
# Lagrangian Hessian is on because Phase 1 showed it matters for constrained
# convergence on non-convex CTG surfaces.
DEFAULT_SQP_CONFIG = SQPConfig(
    max_iter=300,
    use_exact_hessian=True,
    tol_stationarity=1e-6,
    tol_feasibility=1e-6,
)


def _lift_to_parametric(objective_func, constraint_func):
    """
    Promote `(x,) -> scalar / (n_g,)` callables to septal's `(x, p)` signature.

    The evaluators currently bake sample-specific data into the closure via
    `partial(y=...)` rather than threading it through `p`, so `p` is always
    empty (`jnp.zeros(0)`).  Lifting those sample-varying parameters into a
    real `p` is a Phase 2 performance optimisation — doing it now would let
    one `ParametricSQPFactory` be reused across all samples for a given
    successor, cutting JIT cost from O(N_samples) to O(1).
    """
    def obj(x, p):
        return objective_func(x).reshape(())

    if constraint_func is None:
        return obj, None

    def g(x, p):
        return constraint_func(x).reshape(-1)

    return obj, g


def _resolve_config(tol: float, user_config: Optional[SQPConfig]) -> SQPConfig:
    """Use the caller's `SQPConfig` verbatim if provided, else apply `tol` to defaults."""
    if user_config is not None:
        return user_config
    return replace(
        DEFAULT_SQP_CONFIG,
        tol_stationarity=tol,
        tol_feasibility=tol,
    )


def build_problem(objective_func, constraint_func, bounds,
                  constraint_lhs=None, constraint_rhs=None) -> ParametricNLPProblem:
    """
    Construct a `ParametricNLPProblem` from mu_F evaluator-style inputs.

    Box-only: pass `constraint_func=None`.
    Constrained: defaults to one-sided inequality `g(x) <= 0`.
    Equality / two-sided: pass `constraint_lhs` and `constraint_rhs` explicitly.
    """
    lb = jnp.asarray(bounds[0]).reshape(-1)
    ub = jnp.asarray(bounds[1]).reshape(-1)
    n_d = int(lb.size)

    obj, g = _lift_to_parametric(objective_func, constraint_func)

    if g is None:
        return ParametricNLPProblem(
            objective=obj,
            bounds=[lb, ub],
            n_decision=n_d,
            n_params=0,
        )

    # Probe constraint dim at the midpoint so we can size lhs/rhs correctly.
    x_probe = 0.5 * (lb + ub)
    n_g = int(g(x_probe, jnp.zeros(0)).shape[0])
    lhs = constraint_lhs if constraint_lhs is not None else jnp.full((n_g,), -jnp.inf)
    rhs = constraint_rhs if constraint_rhs is not None else jnp.zeros((n_g,))

    return ParametricNLPProblem(
        objective=obj,
        bounds=[lb, ub],
        n_decision=n_d,
        n_params=0,
        constraints=g,
        constraint_lhs=jnp.asarray(lhs).reshape(-1),
        constraint_rhs=jnp.asarray(rhs).reshape(-1),
        n_constraints=n_g,
    )


def construct_constrained_solver(
    objective_func: Callable,
    constraint_func: Optional[Callable],
    bounds,
    tol: float,
    sobol_pts=None,
    constraint_lhs=None,
    constraint_rhs=None,
    config: Optional[SQPConfig] = None,
) -> Callable:
    """
    Build a septal-backed multi-start solver.

    Returns a closure `solver(initial_guesses) -> dict` where `initial_guesses`
    has shape `(n_starts, n_d)` and the dict carries scalar summaries of the
    best start: `{objective, error, converged, params}`.

    `sobol_pts` is accepted for signature parity with `construct_solver` but
    currently unused — septal's SQP handles multi-start by vmapping the
    fixed-iteration scan solver (`sqp_solve_scan`) across the supplied
    initial-guess batch.
    """
    problem = build_problem(
        objective_func, constraint_func, bounds,
        constraint_lhs=constraint_lhs, constraint_rhs=constraint_rhs,
    )
    sqp_cfg = _resolve_config(tol, config)
    factory = ParametricSQPFactory(problem, sqp_cfg)

    def _solver(initial_guesses):
        x0 = jnp.asarray(initial_guesses).reshape(-1, problem.n_decision)
        n_starts = x0.shape[0]
        # No parameters threaded through yet — pad to (n_starts, 0).
        p_batch = jnp.zeros((n_starts, 0))

        result = factory.solve_batch(x0, p_batch)

        converged = jnp.asarray(result.success)
        objectives = jnp.asarray(result.objective)
        # Penalise non-converged starts so argmin skips them when any start converged.
        ranked = jnp.where(converged, objectives, objectives + 1e10)
        best = jnp.argmin(ranked)

        return {
            "objective": objectives[best].reshape(1),
            "error": jnp.asarray(result.kkt_stationarity)[best].reshape(1),
            "converged": converged[best].reshape(1),
            "params": jnp.asarray(result.decision_variables)[best].reshape(-1),
        }

    return _solver


def solve_constrained(solvers, initial_guesses) -> dict:
    """
    Driver symmetric with `jax_evaluator.solve`.

    Runs each per-sample solver on its initial-guess batch and stacks the
    scalar summaries. Input lists must be the same length (one entry per
    sample in the pmap/vmap shard).
    """
    obj, err, conv, params = [], [], [], []
    for s, ig in zip(solvers, initial_guesses):
        r = s(ig)
        obj.append(r["objective"])
        err.append(r["error"])
        conv.append(r["converged"])
        params.append(r["params"])

    return {
        "objective": jnp.stack(obj).reshape(-1),
        "error":     jnp.stack(err).reshape(-1),
        "converged": jnp.stack(conv).reshape(-1),
        "params":    jnp.stack(params),
    }


# ---------------------------------------------------------------------------
# Monolithic NLP solver — replaces the old CasADi+IPOPT
# `callable_casadi_nlp_optimizer_mono` used by the direct-shooting modules.
#
# Returns septal's native `SQPResult` unchanged — no adapter shim.  Callers
# read `.success`, `.objective`, `.decision_variables`, `.constraints` etc.
# directly (see septal.jax.sqp.schema.SQPResult).
# ---------------------------------------------------------------------------

def septal_monolithic_solver(objective, constraints, bounds, initial_guess,
                             lhs, rhs, config: Optional[SQPConfig] = None):
    """
    Solve a monolithic NLP via septal's SQP factory.

    Parameters
    ----------
    objective : Callable
        JAX scalar objective `f(x) -> scalar`.
    constraints : Callable | list[Callable]
        Either a single vector-valued `g(x) -> (n_g,)` or a list of per-row
        callables, each returning a scalar / (1,) / (1, 1) contribution.
    bounds : list[jnp.ndarray]
        `[lb, ub]`, each shape `(n_d,)` or `(1, n_d)`.
    initial_guess : jnp.ndarray
        Starting point, shape `(n_d,)` or `(1, n_d)`.
    lhs, rhs : jnp.ndarray
        Constraint bounds, shape `(n_g,)` or `(n_g, 1)`.  Casadi convention:
        `lhs <= g(x) <= rhs` elementwise.
    config : SQPConfig, optional
        Override the default septal settings.

    Returns
    -------
    septal.jax.sqp.SQPResult
        Native septal result.  Key fields:
          `success`             : bool
          `objective`           : scalar
          `decision_variables`  : (n_d,)
          `constraints`         : (n_g,)      value of `g(x*)`
          `multipliers`         : (n_g,)
          `kkt_stationarity`    : scalar
          `kkt_feasibility`     : scalar
          `iterations`          : int
          `timing`              : float       wall-clock seconds
    """
    x0 = jnp.asarray(initial_guess).reshape(-1)
    n_d = int(x0.size)

    # Collapse a list-of-callables into a single vector-valued constraint.
    if isinstance(constraints, (list, tuple)):
        _cons_list = tuple(constraints)
        def g_stack(x, p):
            pieces = [jnp.asarray(c(x)).reshape(-1) for c in _cons_list]
            return jnp.concatenate(pieces)
    else:
        def g_stack(x, p):
            return jnp.asarray(constraints(x)).reshape(-1)

    def f_param(x, p):
        return jnp.asarray(objective(x)).reshape(())

    lb = jnp.asarray(bounds[0]).reshape(-1)
    ub = jnp.asarray(bounds[1]).reshape(-1)

    # Probe constraint width once (cheap — single JAX eval).
    n_g = int(g_stack(x0, jnp.zeros(0)).shape[0])

    problem = ParametricNLPProblem(
        objective=f_param,
        bounds=[lb, ub],
        n_decision=n_d,
        n_params=0,
        constraints=g_stack,
        constraint_lhs=jnp.asarray(lhs).reshape(-1),
        constraint_rhs=jnp.asarray(rhs).reshape(-1),
        n_constraints=n_g,
    )
    factory = ParametricSQPFactory(
        problem, config if config is not None else DEFAULT_SQP_CONFIG,
    )
    return factory.solve(x0, jnp.zeros(0))
