"""Integer-aware parametric NLP: discrete spec, continuous factory, swappable backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable, NamedTuple, Optional, Protocol

import jax
import jax.numpy as jnp
import numpy as np


__all__ = [
    "LinearEq",
    "LinearIneq",
    "DesignIntegerDim",
    "IntegerProblem",
    "IntegerNLPSpec",
    "SolveResult",
    "IntegerBackend",
    "EnumerationBackend",
    "solve_integer_nlp",
    "solve_integer_nlp_batched",
    "splice_design_integers",
]


# ---------------------------------------------------------------------------
# Discrete-side description
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinearEq:
    """Linear equality on integer variables: Σ coeffs[i] · y_i = rhs.

    coeffs is indexed over the structural-integer variables of the parent
    IntegerProblem (not the design-integer dims).

    """
    coeffs: tuple
    rhs: float

    # ---- External Methods ----

    def satisfied(self, structural_values) -> bool:
        """Read by feasible_assignments when filtering integer rows."""
        return abs(sum(c * v for c, v in zip(self.coeffs, structural_values))
                   - self.rhs) < 1e-9


@dataclass(frozen=True)
class LinearIneq:
    """Linear inequality on structural integers: Σ coeffs[i] · y_i ≤ rhs.

    Companion to LinearEq, applied by feasible_assignments as a row filter.

    """
    coeffs: tuple
    rhs: float

    # ---- External Methods ----

    def satisfied(self, structural_values) -> bool:
        """Read by feasible_assignments when filtering integer rows."""
        return sum(c * v for c, v in zip(self.coeffs, structural_values)) <= self.rhs + 1e-9


@dataclass(frozen=True)
class DesignIntegerDim:
    """One integer-valued dim of the continuous design vector.

    slot is the design-vector position where this dim's value gets spliced
    after the SQP returns; domain is the tuple of allowed integer values.

    """
    slot: int
    domain: tuple


@dataclass(frozen=True)
class IntegerProblem:
    """Discrete structure of an integer-aware sub-problem.

    feasible_assignments enumerates the Cartesian product of every integer
    domain and filters by sos1_groups + linear_constraints, yielding the
    column-encoded parametric tail consumed at solve-time.

    """
    design_dims: tuple = ()                     # tuple[DesignIntegerDim, ...]
    structural_domains: tuple = ()              # tuple[tuple[int, ...], ...]
    sos1_groups: tuple = ()                     # tuple[tuple[int, ...], ...]
    linear_constraints: tuple = ()              # tuple[LinearEq | LinearIneq, ...]

    # ---- External Methods ----

    @property
    def n_design_int(self) -> int:
        """Count of integer-valued design-vector dims."""
        return len(self.design_dims)

    @property
    def n_structural(self) -> int:
        """Count of structural integer variables (parametric-tail only)."""
        return len(self.structural_domains)

    @property
    def n_int_total(self) -> int:
        """Total integer variables across design and structural sets."""
        return self.n_design_int + self.n_structural

    def feasible_assignments(self) -> jnp.ndarray:
        """
        Pre-filtered integer assignment table, shape (n_feasible, n_int_total),
        design-integer columns first then structural. An integer-free problem
        returns shape (1, 0) so the downstream pipeline stays shape-uniform.
        """
        design_domains = tuple(d.domain for d in self.design_dims)
        all_domains = design_domains + self.structural_domains

        if not all_domains:
            return jnp.zeros((1, 0), dtype=jnp.float32)

        rows = list(product(*all_domains))
        n_design = self.n_design_int

        # SOS1: at most one non-zero per group; group indices offset by n_design.
        for group in self.sos1_groups:
            rows = [r for r in rows
                    if sum(1 for i in group if r[n_design + i] != 0) <= 1]

        # Linear constraints over structural variables only.
        for cons in self.linear_constraints:
            rows = [r for r in rows if cons.satisfied(r[n_design:])]

        if not rows:
            return jnp.zeros((0, len(all_domains)), dtype=jnp.float32)
        return jnp.asarray(rows, dtype=jnp.float32)

    @classmethod
    def from_cfg(cls, design_domain, n_heads: int = 0) -> "IntegerProblem":
        """
        Build from a cfg.case_study.design_domain spec plus an optional K-head
        selector. Each domain entry is 'real' (skipped) or a list of integers;
        n_heads > 0 appends K SOS1 binaries with a Σ=1 active-head equality.
        """
        design_dims = tuple(
            DesignIntegerDim(slot=i, domain=tuple(d))
            for i, d in enumerate(design_domain)
            if d != 'real'
        )
        structural_domains: tuple = ()
        sos1_groups: tuple = ()
        linear_constraints: tuple = ()
        if n_heads > 0:
            structural_domains = tuple((0, 1) for _ in range(n_heads))
            sos1_groups = (tuple(range(n_heads)),)
            linear_constraints = (LinearEq(coeffs=tuple(1.0 for _ in range(n_heads)),
                                           rhs=1.0),)
        return cls(
            design_dims=design_dims,
            structural_domains=structural_domains,
            sos1_groups=sos1_groups,
            linear_constraints=linear_constraints,
        )


# ---------------------------------------------------------------------------
# Spec + result + backend
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class IntegerNLPSpec:
    """Static state for an integer-aware parametric SQP sub-problem.

    Built once per node in _build_for_key and reused across solve calls, so
    the pjit cache keys on id(spec) (eq=False) for one compile per node.
    __post_init__ attaches cached _solve_batched / _screen_batched vmaps.

    """
    integer_problem: IntegerProblem
    continuous_factory: object                  # septal.jax.sqp.ParametricSQPFactory
    screener: Optional[Callable]
    sobol_pool: jnp.ndarray
    n_starts: int
    feasibility_tol: float
    backend: str = 'enumeration'                # 'enumeration' | 'dfs_bb'
    bb_max_nodes: int = 0                       # dfs_bb: assignment-visit cap (0 = all)

    _solve_batched:  Optional[Callable] = field(
        default=None, init=False, repr=False, compare=False,
    )
    _screen_batched: Optional[Callable] = field(
        default=None, init=False, repr=False, compare=False,
    )

    # ---- External Methods ----

    def __post_init__(self):
        """Attach the cached assignment/start solve and screen vmaps."""
        # Nested vmap: outer over assignment rows, inner over warm-starts.
        solve_one = self.continuous_factory._solve_jit
        object.__setattr__(self, "_solve_batched", jax.vmap(
            jax.vmap(solve_one, in_axes=(0, None)),
            in_axes=(0, 0),
        ))

        if self.screener is not None:
            pool, n_starts, screener = self.sobol_pool, self.n_starts, self.screener
            def _screen_one(p_for_screen):
                scores  = screener(pool, p_for_screen)         # (n_pool,)
                top_idx = jnp.argsort(scores)[:n_starts]
                return pool[top_idx]                           # (n_starts, n_d)
            object.__setattr__(self, "_screen_batched", jax.vmap(_screen_one))


class SolveResult(NamedTuple):
    """One solve's output, scalar or 1-D for a single (sample, theta).

    success is the viability flag (feasibility within tol and iterate in
    bounds) of the best multistart pick; kkt_converged is septal's KKT flag.
    Reporting uses viability for the CTG mask and convergence as a diagnostic.

    """
    objective: jnp.ndarray
    success: jnp.ndarray
    kkt_converged: jnp.ndarray
    x: jnp.ndarray                              # (n_d_continuous,)
    assignment_idx: jnp.ndarray                 # scalar, row of feasible_assignments


class IntegerBackend(Protocol):
    """Interface every integer backend implements.

    Given a spec and one upstream input y, return a SolveResult; theta and
    outer-batch axes belong to the caller, which vmaps the backend across
    its batch dimensions.

    """

    # ---- External Methods ----

    def solve(self, spec: "IntegerNLPSpec", y) -> SolveResult: ...


class EnumerationBackend:
    """Full enumeration of feasible integer assignments.

    For each feasible_assignments row it runs the continuous NLP with the
    assignment in the parametric tail, vmapping septal's core over
    (assignment × start); the best feasible row wins via _pick_best.

    """

    # ---- External Methods ----

    def solve(self, spec, y) -> SolveResult:
        """Read by solve_integer_nlp as the default backend."""
        x0_per_asn = _screen_warmstarts(spec, y)            # (n_asn, n_starts, n_d)
        p_per_asn  = _augment_p(spec, y)                    # (n_asn, n_p_total)

        # Cached nested vmap from spec.__post_init__.
        state = spec._solve_batched(x0_per_asn, p_per_asn)
        # state.x: (n_asn, n_starts, n_d); state.f_val / feasibility: (n_asn, n_starts)

        # Per-start viability: KKT feasibility within tol AND iterate in bounds.
        lb = spec.continuous_factory.problem.lb
        ub = spec.continuous_factory.problem.ub
        in_bounds = jnp.all((state.x >= lb) & (state.x <= ub), axis=-1)
        viable    = (jnp.asarray(state.feasibility) <= spec.feasibility_tol) & in_bounds
        # Septal's per-iter KKT-convergence flag, carried alongside viability.
        converged = jnp.asarray(state.converged)

        return _pick_best(spec, state.f_val, viable, converged, state.x)


class DFSBranchBoundBackend:
    """Anytime sequential search over integer assignments.

    At this layer the integers live in the parametric tail (not the decision
    vector), so no hull relaxation exists to bound internal nodes — the search
    is a screener-ordered depth-one enumeration with incumbent pruning and an
    optional visit cap (spec.bb_max_nodes).  Its value over EnumerationBackend
    is memory (one assignment in flight instead of all at once) and early
    termination on large assignment spaces; the true relaxation-based B&B is
    the monolithic path (septal_monolithic_bb_solver).  Python-side control
    flow: NOT vmap/jit-compatible, so batched theta falls back to a host loop.

    """

    # ---- External Methods ----

    def solve(self, spec, y) -> SolveResult:
        """Read by solve_integer_nlp when spec.backend == 'dfs_bb'."""
        x0_per_asn = _screen_warmstarts(spec, y)            # (n_asn, n_starts, n_d)
        p_per_asn  = _augment_p(spec, y)                    # (n_asn, n_p_total)
        n_asn = int(p_per_asn.shape[0])
        cap = spec.bb_max_nodes if spec.bb_max_nodes > 0 else n_asn

        lb = spec.continuous_factory.problem.lb
        ub = spec.continuous_factory.problem.ub

        # Order assignments by their best screener score (cheapest-looking first).
        if spec.screener is not None:
            scores = jnp.stack([
                jnp.min(jax.vmap(spec.screener, in_axes=(0, None))(spec.sobol_pool, p_per_asn[a]))
                for a in range(n_asn)
            ])
            order = [int(i) for i in jnp.argsort(scores)]
        else:
            order = list(range(n_asn))

        best = None
        for count, a in enumerate(order):
            if count >= cap:
                break
            state = spec._solve_batched(x0_per_asn[a:a+1], p_per_asn[a:a+1])
            in_bounds = jnp.all((state.x >= lb) & (state.x <= ub), axis=-1)
            viable = (jnp.asarray(state.feasibility) <= spec.feasibility_tol) & in_bounds
            row = _pick_best(spec, state.f_val, viable, jnp.asarray(state.converged), state.x)
            row = SolveResult(row.objective, row.success, row.kkt_converged,
                              row.x, jnp.asarray(a))
            if bool(row.success) and (best is None or float(row.objective) < float(best.objective)):
                best = row
        if best is None:
            # nothing viable: return the first row so shapes/dtypes stay uniform
            state = spec._solve_batched(x0_per_asn[0:1], p_per_asn[0:1])
            in_bounds = jnp.all((state.x >= lb) & (state.x <= ub), axis=-1)
            viable = (jnp.asarray(state.feasibility) <= spec.feasibility_tol) & in_bounds
            best = _pick_best(spec, state.f_val, viable, jnp.asarray(state.converged), state.x)
        return best


_BACKENDS = {
    'enumeration': EnumerationBackend(),
    'dfs_bb':      DFSBranchBoundBackend(),
}
_DEFAULT_BACKEND = _BACKENDS['enumeration']


def solve_integer_nlp(spec: IntegerNLPSpec, y,
                      backend: Optional[IntegerBackend] = None) -> SolveResult:
    """
    Solve one integer-aware parametric NLP for upstream input y. Theta-blind;
    for batched theta use solve_integer_nlp_batched, which caches the vmap
    wrapper at module import.  Backend resolves from spec.backend unless
    overridden explicitly.
    """
    if backend is None:
        backend = _BACKENDS.get(getattr(spec, 'backend', 'enumeration'), _DEFAULT_BACKEND)
    return backend.solve(spec, y)


def _solve_one(spec: IntegerNLPSpec, y) -> SolveResult:
    """Top-level trampoline so the vmap below has a stable function id."""
    return solve_integer_nlp(spec, y, backend=_DEFAULT_BACKEND)


# jax.jit(static_argnums=(0,)) compiles the batched program once per spec.
_solve_one_batched = jax.jit(
    jax.vmap(_solve_one, in_axes=(None, 0)),
    static_argnums=(0,),
)


def solve_integer_nlp_batched(spec: IntegerNLPSpec, ys) -> SolveResult:
    """
    Batched solve over the theta axis of ys, shape (n_theta, n_y). spec is
    static and hashed via id(spec) for the pjit cache: one compile per spec,
    later calls hit the cache. Returns a SolveResult with leading n_theta axis.

    'dfs_bb' specs run a host loop over theta rows (Python control flow can't
    live inside vmap); expect it to be slower for large theta batches —
    enumeration remains the right default for the decomposition.
    """
    if getattr(spec, 'backend', 'enumeration') == 'enumeration':
        return _solve_one_batched(spec, ys)

    rows = [solve_integer_nlp(spec, ys[i]) for i in range(int(ys.shape[0]))]
    return SolveResult(
        objective=jnp.stack([r.objective for r in rows]),
        success=jnp.stack([r.success for r in rows]),
        kkt_converged=jnp.stack([r.kkt_converged for r in rows]),
        x=jnp.stack([r.x for r in rows]),
        assignment_idx=jnp.stack([r.assignment_idx for r in rows]),
    )


# ---------------------------------------------------------------------------
# Internal helpers — pure jnp, JIT-traceable, vmap-composable
# ---------------------------------------------------------------------------

def _augment_p(spec: IntegerNLPSpec, y):
    """
    Build the per-assignment parametric tail [y | assignment row], shape
    (n_assignments, n_p_base + n_int_total). No n_starts replication: the
    inner vmap over starts shares p via in_axes=(0, None).
    """
    assignments = spec.integer_problem.feasible_assignments()
    n_asn = assignments.shape[0]

    y_arr = jnp.asarray(y).reshape(-1)
    return jnp.concatenate(
        [
            jnp.broadcast_to(y_arr.reshape(1, -1), (n_asn, y_arr.size)),
            assignments,
        ],
        axis=-1,
    )


def _screen_warmstarts(spec: IntegerNLPSpec, y):
    """
    Per-assignment Sobol screen to the top-n_starts warm-starts, shape
    (n_assignments, n_starts, n_d_continuous). With no screener, falls back
    to the first n_starts pool points broadcast across assignments.
    """
    pool = spec.sobol_pool                                   # (n_pool, n_d_continuous)
    assignments = spec.integer_problem.feasible_assignments()
    n_asn = assignments.shape[0]
    n_starts = spec.n_starts

    if spec.screener is None:
        x0_row = pool[:n_starts]                             # (n_starts, n_d_continuous)
        return jnp.broadcast_to(x0_row, (n_asn, n_starts, x0_row.shape[-1]))

    y_arr = jnp.asarray(y).reshape(-1)
    p_per_asn = jnp.concatenate(
        [
            jnp.broadcast_to(y_arr.reshape(1, -1), (n_asn, y_arr.size)),
            assignments,
        ],
        axis=-1,
    )                                                        # (n_asn, n_p_total)

    # Cached jax.vmap(_screen_one); the screener scans the pool internally.
    return spec._screen_batched(p_per_asn)                   # (n_asn, n_starts, n_d_continuous)


def _pick_best(spec: IntegerNLPSpec, obj, viable, converged, x):
    """
    Argmin chain: best warm-start per assignment, then best feasible
    assignment, with a +big penalty enforcing feasibility priority. converged
    rides the same path to report the chosen iterate's KKT flag.
    """
    # obj, viable, converged: (n_asn, n_starts); x: (n_asn, n_starts, n_d_continuous)
    big = jnp.asarray(1.0e10, dtype=obj.dtype)
    ranked = jnp.where(viable, obj, obj + big)
    best_start = jnp.argmin(ranked, axis=-1)                 # (n_asn,)
    asn_obj       = jnp.take_along_axis(obj,       best_start[:, None],       axis=-1).squeeze(-1)
    asn_viable    = jnp.take_along_axis(viable,    best_start[:, None],       axis=-1).squeeze(-1)
    asn_converged = jnp.take_along_axis(converged, best_start[:, None],       axis=-1).squeeze(-1)
    asn_x         = jnp.take_along_axis(x,         best_start[:, None, None], axis=-2).squeeze(-2)

    ranked_asn = jnp.where(asn_viable, asn_obj, asn_obj + big)
    best_asn = jnp.argmin(ranked_asn)                        # scalar

    return SolveResult(
        objective=asn_obj[best_asn],
        success=asn_viable[best_asn],
        kkt_converged=asn_converged[best_asn],
        x=asn_x[best_asn],
        assignment_idx=best_asn,
    )


# ---------------------------------------------------------------------------
# Post-solve helper — caller composes this with solve_integer_nlp
# ---------------------------------------------------------------------------

def splice_design_integers(x_continuous, spec: IntegerNLPSpec,
                           assignment_idx, n_design_full: int) -> jnp.ndarray:
    """
    Reinsert the chosen design-integer values into the continuous decision
    vector, returning the full n_design_full design vector for downstream
    consumers. Returns x_continuous unchanged when there are no design ints.
    """
    design_dims = spec.integer_problem.design_dims
    if not design_dims:
        return x_continuous

    assignments = spec.integer_problem.feasible_assignments()
    int_vals = assignments[assignment_idx, : len(design_dims)]   # (n_design_int,)
    int_slots = jnp.asarray([d.slot for d in design_dims], dtype=jnp.int32)

    cont_mask = np.ones(n_design_full, dtype=bool)
    for d in design_dims:
        cont_mask[d.slot] = False
    cont_slots = jnp.asarray(np.where(cont_mask)[0], dtype=jnp.int32)

    full = jnp.zeros((n_design_full,), dtype=x_continuous.dtype)
    full = full.at[cont_slots].set(x_continuous)
    full = full.at[int_slots].set(int_vals.astype(x_continuous.dtype))
    return full
