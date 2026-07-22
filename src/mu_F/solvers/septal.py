"""Constrained NLP solver backend wrapping septal's ParametricSQPFactory."""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from septal.jax.sqp import (
    ParametricNLPProblem,
    ParametricSQPFactory,
    SQPConfig,
)
from septal.jax.sqp.solver import make_sqp_step, init_sqp_state, state_to_result

DEFAULT_SQP_CONFIG = SQPConfig(
    max_iter=300,
    use_exact_hessian=True,
    tol_stationarity=1e-4,
    tol_feasibility=1e-4,
)


def _lift_to_parametric(objective_func, constraint_func):
    """
    Promote (x,) -> scalar / (n_g,) callables to septal's (x, p) signature.
    p is always empty; sample data is baked into the closures themselves.
    """
    def obj(x, p):
        return objective_func(x).reshape(())

    if constraint_func is None:
        return obj, None

    def g(x, p):
        return constraint_func(x).reshape(-1)

    return obj, g


def _resolve_config(tol: float, user_config: Optional[SQPConfig]) -> SQPConfig:
    """
    Use the caller's SQPConfig verbatim if provided, else apply tol to
    the module defaults.
    """
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
    Construct a ParametricNLPProblem from mu_F evaluator-style inputs.
    Box-only passes constraint_func=None; constrained defaults to g(x) <= 0;
    pass constraint_lhs / constraint_rhs for equality or two-sided bounds.
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

    # Probe constraint dim at the midpoint to size lhs/rhs correctly.
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
    Returns a closure mapping an (n_starts, n_d) initial-guess batch to the
    scalar summaries of the best start. sobol_pts is unused signature parity.
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
        # No parameters threaded through; pad to (n_starts, 0).
        p_batch = jnp.zeros((n_starts, 0))

        result = factory.solve_batch(x0, p_batch)

        converged = jnp.asarray(result.success)
        objectives = jnp.asarray(result.objective)
        # Penalise non-converged starts so argmin skips them when any converged.
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
    Driver symmetric with jax_evaluator.solve.
    Runs each per-sample solver on its initial-guess batch and stacks the
    scalar summaries; the two input lists must be the same length.
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
# Monolithic NLP solver
# ---------------------------------------------------------------------------
# Used by the direct-shooting modules; returns septal's native SQPResult,
# whose fields callers read directly (see septal.jax.sqp.schema.SQPResult).

def septal_monolithic_solver(objective, constraints, bounds, initial_guess,
                             lhs, rhs, config: Optional[SQPConfig] = None):
    """
    Solve a monolithic NLP via septal's SQP factory under the convention
    lhs <= g(x) <= rhs. constraints may be a single vector-valued callable or
    a list of per-row callables. Returns septal's native SQPResult.
    """
    x0 = jnp.asarray(initial_guess).reshape(-1)
    n_d = int(x0.size)

    # Collapse a list of callables into a single vector-valued constraint.
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

    # Probe constraint width once with a single JAX eval.
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
    cfg = config if config is not None else DEFAULT_SQP_CONFIG
    return _run_logged_sqp(problem, x0, cfg)


def septal_monolithic_bb_solver(objective, constraints, bounds, initial_guess,
                                lhs, rhs, config: Optional[SQPConfig] = None,
                                int_slots=None, int_domains=None,
                                bb_options=None):
    """
    Integer-aware monolithic solve: DFS branch-and-bound over the decision
    slots in `int_slots` (each restricted to the values in `int_domains`),
    with septal solving each relaxation.

    The integer box is enforced through 2*n_int parametric inequality rows
    (x_i - lo_i >= 0, hi_i - x_i >= 0 with (lo, hi) carried in p), so a single
    compiled SQP step serves every B&B node — branching just re-inits the
    state with clamped (lo, hi).  Returns septal's native SQPResult for the
    incumbent, so callers are agnostic to the backend.
    """
    from mu_F.solvers.branch_and_bound import BBConfig, dfs_branch_bound
    import numpy as np

    x0 = jnp.asarray(initial_guess).reshape(-1)
    n_d = int(x0.size)
    slots = [int(s) for s in (int_slots or [])]
    domains = [list(map(float, d)) for d in (int_domains or [])]
    n_int = len(slots)
    assert n_int > 0, "septal_monolithic_bb_solver called without integer slots"
    slot_arr = jnp.asarray(slots, dtype=jnp.int32)

    if isinstance(constraints, (list, tuple)):
        _cons_list = tuple(constraints)
        def g_base(x):
            return jnp.concatenate([jnp.asarray(c(x)).reshape(-1) for c in _cons_list])
    else:
        def g_base(x):
            return jnp.asarray(constraints(x)).reshape(-1)

    def g_aug(x, p):
        lo, hi = p[:n_int], p[n_int:]
        xi = x[slot_arr]
        return jnp.concatenate([g_base(x), xi - lo, hi - xi])

    def f_param(x, p):
        return jnp.asarray(objective(x)).reshape(())

    lb = jnp.asarray(bounds[0]).reshape(-1)
    ub = jnp.asarray(bounds[1]).reshape(-1)
    n_g = int(g_base(x0).shape[0])

    lhs_aug = jnp.concatenate([jnp.asarray(lhs).reshape(-1), jnp.zeros(2 * n_int)])
    rhs_aug = jnp.concatenate([jnp.asarray(rhs).reshape(-1), jnp.full(2 * n_int, jnp.inf)])

    problem = ParametricNLPProblem(
        objective=f_param,
        bounds=[lb, ub],
        n_decision=n_d,
        n_params=2 * n_int,
        constraints=g_aug,
        constraint_lhs=lhs_aug,
        constraint_rhs=rhs_aug,
        n_constraints=n_g + 2 * n_int,
    )
    cfg = config if config is not None else DEFAULT_SQP_CONFIG
    step = jax.jit(make_sqp_step(problem, cfg))
    dom_lo = np.asarray([min(d) for d in domains])
    dom_hi = np.asarray([max(d) for d in domains])

    def _solve_relaxation(fixed: dict, warm):
        lo, hi = dom_lo.copy(), dom_hi.copy()
        for slot, val in fixed.items():
            j = slots.index(slot)
            lo[j] = hi[j] = val
        p = jnp.asarray(np.concatenate([lo, hi]))

        x_start = jnp.asarray(warm).reshape(-1) if warm is not None else x0
        x_start = x_start.at[slot_arr].set(
            jnp.clip(x_start[slot_arr], jnp.asarray(lo), jnp.asarray(hi)))

        state = init_sqp_state(x_start, p, problem, cfg)
        for _ in range(int(cfg.max_iter)):
            state, _ = step(state, None)
            if bool(state.converged):
                break
        feasible = bool(state.feasibility <= cfg.tol_feasibility * 10)
        return (np.asarray(state.x), float(state.f_val),
                feasible, float(state.feasibility))

    opts = dict(bb_options or {})
    bb_cfg = BBConfig(
        time_budget=float(opts.get('time_budget', 600.0)),
        max_nodes=int(opts.get('max_nodes', 2000)),
        integrality_tol=float(opts.get('integrality_tol', 1.0e-3)),
        bound_slack=float(opts.get('bound_slack', 0.0)),
        log_every=int(opts.get('log_every', 25)),
    )
    seed_fixing = _seed_leaf(x0, slots, domains) if opts.get('seed_incumbent', False) else None
    logging.info(f"B&B monolithic solve: n_d={n_d} n_int={n_int} "
                 f"budget={bb_cfg.time_budget:.0f}s max_nodes={bb_cfg.max_nodes} "
                 f"seed_leaf={'on' if seed_fixing else 'off'}")

    stats = dfs_branch_bound(_solve_relaxation, slots, domains, np.asarray(x0), bb_cfg,
                             seed_fixing=seed_fixing)

    if stats.best_x is None:
        logging.warning("B&B found no feasible incumbent — returning the hull relaxation")
        fixing = {}
        warm = np.asarray(x0)
    else:
        fixing = stats.best_fixing
        warm = stats.best_x

    # Final logged solve at the incumbent fixing produces the returned SQPResult.
    lo, hi = dom_lo.copy(), dom_hi.copy()
    for slot, val in fixing.items():
        j = slots.index(slot)
        lo[j] = hi[j] = val
    p_final = jnp.asarray(np.concatenate([lo, hi]))
    x_start = jnp.asarray(warm).reshape(-1)
    x_start = x_start.at[slot_arr].set(
        jnp.clip(x_start[slot_arr], jnp.asarray(lo), jnp.asarray(hi)))
    return _run_logged_sqp(problem, x_start, cfg, p=p_final)


def _seed_leaf(x0, slots, domains) -> dict:
    """Snap the warm start's integer slots onto their nearest allowed values, so
    the assignment it already carries (e.g. the decomposition's) can be solved as
    a leaf and seed the incumbent.  Read by septal_monolithic_bb_solver."""
    return {slot: min(domains[j], key=lambda c: abs(c - float(x0[slot])))
            for j, slot in enumerate(slots)}


def _run_logged_sqp(problem, x0, cfg, p=None):
    """
    Drive septal's SQP step on the host so each iteration's KKT residuals are
    logged and the loop exits the instant it converges (the scan solver always
    runs every iteration). The compiled step is identical to factory.solve's.
    """
    step = jax.jit(make_sqp_step(problem, cfg))
    state = init_sqp_state(x0, jnp.zeros(0) if p is None else p, problem, cfg)
    lb, ub = problem.bounds
    logging.info(f"SQP start: n_d={problem.n_decision} n_g={problem.n_constraints} "
                 f"max_iter={cfg.max_iter} tol_stat={cfg.tol_stationarity:.1e} "
                 f"tol_feas={cfg.tol_feasibility:.1e}")

    last, hits = 0, {}
    for k in range(int(cfg.max_iter)):
        state, _ = step(state, None)
        last = k
        idx, val = _worst_stationary_component(state, lb, ub)
        hits[idx] = hits.get(idx, 0) + 1
        logging.info(f"SQP iter {k:3d}: stationarity={float(state.stationarity):.3e} "
                     f"feasibility={float(state.feasibility):.3e} "
                     f"obj={float(state.f_val):.5f} penalty={float(state.penalty):.2e} "
                     f"worst@{idx}={val:.2e}")
        if bool(state.converged):
            break

    _log_stationarity_breakdown(state, lb, ub, hits)
    status = "converged" if bool(state.converged) else "max_iter reached"
    logging.info(f"SQP {status} at iter {last}: obj={float(state.f_val):.5f}")
    return state_to_result(state, problem)


def _worst_stationary_component(state, lb, ub):
    """Index/value of the projected-gradient component that sets septal's stationarity."""
    pg = jnp.abs(jnp.clip(state.x - state.grad_lag, lb, ub) - state.x)
    idx = int(jnp.argmax(pg))
    return idx, float(pg[idx])


def _log_stationarity_breakdown(state, lb, ub, hits):
    """Final top projected-gradient components and which indices dominated across iters."""
    pg = jnp.abs(jnp.clip(state.x - state.grad_lag, lb, ub) - state.x)
    order = [int(i) for i in jnp.argsort(pg)[::-1][:5]]
    top = ", ".join(f"{i}:{float(pg[i]):.2e}" for i in order)
    dom = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)[:5]
    dom_str = ", ".join(f"{i}({n})" for i, n in dom)
    logging.info(f"SQP stationarity top-5 [final] {top}")
    logging.info(f"SQP worst-index frequency [idx(count)] {dom_str}")
