"""Depth-first branch-and-bound over integer design slots.

Generic driver: the relaxation solve is injected, so the same loop serves the
monolithic septal path (integers relaxed to their hull, branched by parametric
bound-clamping) and any future backend.  Pruning compares local NLP objectives
against the incumbent — a heuristic bound for nonconvex problems, so results
are reported as best-found, not certified optima.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class BBConfig:
    time_budget: float    = 600.0    # wall-clock seconds
    max_nodes: int        = 2000     # visited-node cap
    integrality_tol: float = 1.0e-3  # |x - round(x)| below this counts as integral
    prune_tol: float      = 1.0e-6   # bound-prune margin against the incumbent
    log_every: int        = 25       # progress line every N visited nodes


@dataclass
class BBStats:
    nodes_visited: int      = 0
    pruned_bound: int       = 0
    pruned_infeasible: int  = 0
    incumbents_found: int   = 0
    best_objective: float   = float('inf')
    best_fixing: dict       = field(default_factory=dict)
    best_x: Optional[np.ndarray] = None
    start_time: float       = field(default_factory=time.time)

    def elapsed(self) -> float:
        return time.time() - self.start_time


def dfs_branch_bound(
    solve_relaxation: Callable,
    int_slots: Sequence[int],
    int_domains: Sequence[Sequence[float]],
    warm_start: Optional[np.ndarray],
    bb_cfg: BBConfig,
) -> BBStats:
    """
    DFS over the integer slots in order.  `solve_relaxation(fixed, warm)` must
    return (x, objective, feasible, infeas_measure) with the un-fixed slots
    relaxed to their domain hull.  Branch values are ordered by proximity to
    the relaxation's value at the branching slot; children warm-start from the
    parent's solution.  A relaxation whose free slots all come back integral is
    accepted as an incumbent immediately (round-and-accept shortcut).
    """
    stats = BBStats()
    slots = list(int_slots)
    domains = [list(d) for d in int_domains]

    def _all_integral(x, depth) -> bool:
        for j in range(depth, len(slots)):
            v = float(x[slots[j]])
            if min(abs(v - c) for c in domains[j]) > bb_cfg.integrality_tol:
                return False
        return True

    def _register_incumbent(x, obj, fixed):
        stats.best_objective = obj
        stats.best_fixing = dict(fixed)
        stats.best_x = np.asarray(x).copy()
        stats.incumbents_found += 1
        logging.info(
            f"B&B incumbent #{stats.incumbents_found}: obj={obj:.4f} "
            f"(visited={stats.nodes_visited}, "
            f"pruned={stats.pruned_bound + stats.pruned_infeasible}, "
            f"t={stats.elapsed():.1f}s)"
        )

    def _recurse(fixed: dict, warm):
        if stats.elapsed() > bb_cfg.time_budget or stats.nodes_visited >= bb_cfg.max_nodes:
            return
        depth = len(fixed)
        stats.nodes_visited += 1

        x, obj, feasible, infeas = solve_relaxation(fixed, warm)

        if not feasible:
            stats.pruned_infeasible += 1
            return
        if obj >= stats.best_objective - bb_cfg.prune_tol:
            stats.pruned_bound += 1
            return

        if stats.nodes_visited % bb_cfg.log_every == 0:
            logging.info(
                f"B&B depth={depth} visited={stats.nodes_visited} "
                f"pruned_bnd={stats.pruned_bound} pruned_inf={stats.pruned_infeasible} "
                f"LB={obj:.4f} best={stats.best_objective:.4f}"
            )

        if depth == len(slots) or _all_integral(x, depth):
            # leaf, or the relaxation landed integral on all free slots
            full_fixing = dict(fixed)
            for j in range(depth, len(slots)):
                v = float(x[slots[j]])
                full_fixing[slots[j]] = min(domains[j], key=lambda c: abs(c - v))
            _register_incumbent(x, obj, full_fixing)
            return

        slot = slots[depth]
        branch_val = float(x[slot])
        for cand in sorted(domains[depth], key=lambda c: abs(c - branch_val)):
            child = dict(fixed)
            child[slot] = float(cand)
            _recurse(child, x)
            if stats.elapsed() > bb_cfg.time_budget or stats.nodes_visited >= bb_cfg.max_nodes:
                return

    _recurse({}, warm_start)

    outcome = "exhausted" if (stats.elapsed() <= bb_cfg.time_budget
                              and stats.nodes_visited < bb_cfg.max_nodes) else "budget hit"
    logging.info(
        f"B&B done ({outcome}): visited={stats.nodes_visited} "
        f"pruned_bnd={stats.pruned_bound} pruned_inf={stats.pruned_infeasible} "
        f"incumbents={stats.incumbents_found} best={stats.best_objective:.4f} "
        f"t={stats.elapsed():.1f}s"
    )
    return stats
