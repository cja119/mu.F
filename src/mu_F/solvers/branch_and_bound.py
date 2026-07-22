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
    bound_slack: float    = 0.0      # fraction of |incumbent| a node may be worse by
                                     # and still be explored; a local NLP value is not
                                     # a valid bound, so 0 prunes good subtrees
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
    seed_fixing: Optional[dict] = None,
) -> BBStats:
    """
    DFS over the integer slots.  `solve_relaxation(fixed, warm)` must return
    (x, objective, feasible, infeas_measure) with the un-fixed slots relaxed to
    their domain hull.  The branching slot is the most-fractional free one and
    its values are ordered by proximity to the relaxation; children warm-start
    from the parent's solution.  A relaxation whose free slots all come back
    integral is accepted as an incumbent immediately (round-and-accept shortcut).

    `seed_fixing` (slot -> value, all integer slots) is solved first as a leaf,
    so a known-good assignment becomes the incumbent before the search starts
    and bound-pruning is live from the first branch.  A root warm start alone
    cannot do this: the root relaxes the very integers the seed carries.
    """
    stats = BBStats()
    slots = list(int_slots)
    domains = [list(d) for d in int_domains]
    dom_of = {slot: domains[j] for j, slot in enumerate(slots)}

    def _distance_to_domain(x, slot) -> float:
        """How un-integral this slot's relaxed value is, in domain units."""
        v = float(x[slot])
        return min(abs(v - c) for c in dom_of[slot])

    def _all_integral(x, free) -> bool:
        return all(_distance_to_domain(x, s) <= bb_cfg.integrality_tol for s in free)

    def _prune_threshold() -> float:
        """Objective a node must be worse than to be discarded. The relaxation is a
        local NLP value, not a valid bound, so bound_slack buys back subtrees a
        strict comparison would wrongly cut."""
        best = stats.best_objective
        if not np.isfinite(best):
            return best
        return best + abs(best) * bb_cfg.bound_slack - bb_cfg.prune_tol

    def _register_incumbent(x, obj, fixed):
        """Keep only improvements: with bound_slack > 0 a node worse than the
        incumbent can survive the prune and reach here, and an unguarded write
        would ratchet best_objective the wrong way."""
        if obj >= stats.best_objective:
            return
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
        if obj >= _prune_threshold():
            stats.pruned_bound += 1
            return

        if stats.nodes_visited % bb_cfg.log_every == 0:
            logging.info(
                f"B&B depth={depth} visited={stats.nodes_visited} "
                f"pruned_bnd={stats.pruned_bound} pruned_inf={stats.pruned_infeasible} "
                f"LB={obj:.4f} best={stats.best_objective:.4f}"
            )

        free = [s for s in slots if s not in fixed]
        if not free or _all_integral(x, free):
            # leaf, or the relaxation landed integral on all free slots
            full_fixing = dict(fixed)
            for s in free:
                v = float(x[s])
                full_fixing[s] = min(dom_of[s], key=lambda c: abs(c - v))
            _register_incumbent(x, obj, full_fixing)
            return

        # most-fractional branching: split the decision the relaxation is least
        # sure of, rather than walking the slots in index order
        slot = max(free, key=lambda s: _distance_to_domain(x, s))
        branch_val = float(x[slot])
        for cand in sorted(dom_of[slot], key=lambda c: abs(c - branch_val)):
            child = dict(fixed)
            child[slot] = float(cand)
            _recurse(child, x)
            if stats.elapsed() > bb_cfg.time_budget or stats.nodes_visited >= bb_cfg.max_nodes:
                return

    if seed_fixing:
        stats.nodes_visited += 1
        x_s, obj_s, feasible_s, infeas_s = solve_relaxation(seed_fixing, warm_start)
        if feasible_s:
            _register_incumbent(x_s, obj_s, seed_fixing)
        else:
            logging.info(f"B&B seed leaf infeasible (infeas={infeas_s:.2e}); "
                         f"searching without a seeded incumbent.")

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
