"""Integer-aware parametric NLP.

Backend-agnostic description of the discrete part of a sub-problem
(`IntegerProblem`), paired with the continuous-NLP factory (septal) via
`IntegerNLPSpec`, solved by a swappable `IntegerBackend`.

The module is **theta-blind by design**: `solve_integer_nlp(spec, y)`
processes one upstream input `y` at a time. Theta threading lives in the
caller (an outer `vmap` in the evaluator); JAX composes the two vmaps into
a single XLA program at trace time.

Discrete-side structure:

  design_dims         integer-valued dims of the continuous design vector
                      (e.g. n_active). Their values get spliced into the
                      design vector at `slot` after the SQP returns.
  structural_domains  integer variables that route ONLY through the
                      parametric tail of the SQP (e.g. K head-selector
                      binaries). Never appear in the design vector.
  sos1_groups         index tuples within structural_domains forming
                      SOS1 sets ("at most one is non-zero"). Combined
                      with a Σ=1 linear constraint they give the "exactly
                      one is 1" encoding used for active-head selection.
  linear_constraints  LinearEq / LinearIneq on structural variables.

For the K-head classifier we encode:

  structural_domains = ((0,1), (0,1), ..., (0,1))     # K binaries
  sos1_groups        = (tuple(range(K)),)
  linear_constraints = (LinearEq((1,)*K, 1),)         # exactly one is 1

`feasible_assignments` enumerates the Cartesian product and applies the
side-constraint filter, so SOS1 prunes the 2^K binary combos down to the K
one-hots at construction time. The integer enumeration that reaches septal
is already pre-filtered.

JIT / XLA contract:

  • feasible_assignments runs Python-side at construction and returns a
    static-shape jnp array captured by closure.
  • IntegerNLPSpec is frozen with identity-based hash
    (`@dataclass(frozen=True, eq=False)`); each evaluator builds one
    per node and reuses the same instance, so the pjit cache fires
    exactly one compile per node.
  • All internal helpers (_augment_p, _screen_warmstarts, _pick_best) are
    pure jnp ops on static-shape arrays.
  • We use septal's JAX-pure single-instance core (`factory._solve_jit`)
    directly and own the vmap structure ourselves — no flat-batch
    replication, no traversal through `solve_batch`.

Single-XLA-compile property: on the first call for a given spec, JAX
traces the whole composition (outer-theta vmap + outer-asn vmap +
inner-start vmap + the JIT'd single solve) into one fused XLA program.
Subsequent calls with matching shape signature hit the cache.

Shape contract within solve_integer_nlp(spec, y):

  y                 (n_p_base,)                    one upstream input
  assignments       (n_assignments, n_int_total)   static; from spec.integer_problem
  p_per_asn         (n_assignments, n_p_total)     one parametric tail per assignment
  x0_per_asn        (n_assignments, n_starts,
                     n_d_continuous)                warm-starts per assignment
  SQPState fields   (n_assignments, n_starts, …)   nested vmap into septal's
                                                    JAX-pure single-instance core
  SolveResult       scalars + (n_d_continuous,) x  argmin chain result

We bypass septal's `solve_batch` and vmap its JIT'd single-instance solver
(`factory._solve_jit`) directly: inner vmap over starts with `p` shared
(`in_axes=(0, None)`), outer vmap over assignments (`in_axes=(0, 0)`). This
keeps the natural (n_asn, n_starts) axes — no flat-batch replication of p,
no reshape on the way out. Outer-theta vmap composes cleanly with the two
inner vmaps into a single fused XLA program.
"""

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


# =============================================================================
# Discrete-side description
# =============================================================================

@dataclass(frozen=True)
class LinearEq:
    """Linear equality on integer variables: Σ coeffs[i] · y_i  =  rhs.

    `coeffs` is indexed over the structural-integer variables of the
    parent `IntegerProblem` (NOT the design-integer dims).
    """
    coeffs: tuple
    rhs: float

    def satisfied(self, structural_values) -> bool:
        return abs(sum(c * v for c, v in zip(self.coeffs, structural_values))
                   - self.rhs) < 1e-9


@dataclass(frozen=True)
class LinearIneq:
    """Linear inequality on structural integers: Σ coeffs[i] · y_i  ≤  rhs."""
    coeffs: tuple
    rhs: float

    def satisfied(self, structural_values) -> bool:
        return sum(c * v for c, v in zip(self.coeffs, structural_values)) <= self.rhs + 1e-9


@dataclass(frozen=True)
class DesignIntegerDim:
    """One integer-valued dim of the continuous design vector.

    `slot` is the position in the full design vector where this dim's
    value gets spliced after the SQP returns. `domain` is the tuple of
    allowed integer values.
    """
    slot: int
    domain: tuple


@dataclass(frozen=True)
class IntegerProblem:
    """Discrete structure of an integer-aware sub-problem.

    `feasible_assignments` enumerates the Cartesian product of every
    integer variable's domain and filters by sos1_groups + linear_constraints.
    The resulting (n_feasible, n_design_ints + n_structural) table is
    consumed at solve-time as the column-encoded parametric tail.
    """
    design_dims: tuple = ()                     # tuple[DesignIntegerDim, ...]
    structural_domains: tuple = ()              # tuple[tuple[int, ...], ...]
    sos1_groups: tuple = ()                     # tuple[tuple[int, ...], ...]
    linear_constraints: tuple = ()              # tuple[LinearEq | LinearIneq, ...]

    @property
    def n_design_int(self) -> int:
        return len(self.design_dims)

    @property
    def n_structural(self) -> int:
        return len(self.structural_domains)

    @property
    def n_int_total(self) -> int:
        return self.n_design_int + self.n_structural

    def feasible_assignments(self) -> jnp.ndarray:
        """Pre-filtered integer assignment table, shape (n_feasible, n_int_total).

        Column layout: design-integer columns first, then structural columns.
        Empty integer problem (no integers anywhere) returns shape (1, 0) —
        one trivial assignment with no columns. That keeps the rest of the
        pipeline shape-uniform when there are no integers to enumerate.
        """
        design_domains = tuple(d.domain for d in self.design_dims)
        all_domains = design_domains + self.structural_domains

        if not all_domains:
            return jnp.zeros((1, 0), dtype=jnp.float32)

        rows = list(product(*all_domains))
        n_design = self.n_design_int

        # SOS1 filter: at most one non-zero among each group's structural
        # variables. `group` indices are into structural_domains, so we
        # offset by n_design when reading the assignment row.
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
        """Build from a `cfg.case_study.design_domain` spec + optional K-head selector.

        Each entry of `design_domain` is either the string `'real'` (skipped)
        or a list of allowed integer values. The position in the original
        `design_domain` list becomes the `slot` of the corresponding
        DesignIntegerDim. If `n_heads > 0` we append K structural binaries
        with an SOS1 group + Σ=1 linear equality — the SOS1 active-head
        encoding.
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


# =============================================================================
# Spec + result + backend
# =============================================================================

@dataclass(frozen=True, eq=False)
class IntegerNLPSpec:
    """Static state for an integer-aware parametric SQP sub-problem.

    `frozen=True` for immutability; `eq=False` uses object-identity for
    `__eq__` / `__hash__` so the pjit cache keys on `id(spec)`.  (The
    auto-generated `__hash__` would otherwise raise on the
    `sobol_pool: jnp.ndarray` field.)

    Built once per node in `_build_for_key` and reused across every
    solve call — one pjit cache entry per node.

    `__post_init__` attaches two cached `jax.vmap` wrappers via
    `object.__setattr__`:

      • `_solve_batched`  — nested (assignment × start) vmap of
        septal's `_solve_jit`; used by `EnumerationBackend.solve`.
      • `_screen_batched` — per-assignment vmap of the screener
        post-processing (argsort + gather); used by
        `_screen_warmstarts` when a screener is configured.

    Both are declared `field(init=False, repr=False, compare=False)`
    so they sit outside the constructor signature and don't enter
    any auto-generated dunder.
    """
    integer_problem: IntegerProblem
    continuous_factory: object                  # septal.jax.sqp.ParametricSQPFactory
    screener: Optional[Callable]
    sobol_pool: jnp.ndarray
    n_starts: int
    feasibility_tol: float

    _solve_batched:  Optional[Callable] = field(
        default=None, init=False, repr=False, compare=False,
    )
    _screen_batched: Optional[Callable] = field(
        default=None, init=False, repr=False, compare=False,
    )

    def __post_init__(self):
        # Nested vmap: outer over assignment rows, inner over warm-starts
        # (with `p` shared across the n_starts axis).
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
    """One solve's output. All fields are scalars or 1-D for one (sample, theta)."""
    objective: jnp.ndarray
    success: jnp.ndarray
    x: jnp.ndarray                              # (n_d_continuous,)
    assignment_idx: jnp.ndarray                 # scalar, row of feasible_assignments


class IntegerBackend(Protocol):
    """Interface every integer backend implements.

    The contract is: given a spec and one upstream input `y`, return a
    SolveResult. Theta and outer-batch axes belong to the caller — backends
    process a single (sample, theta) at a time and rely on the caller's
    vmap to broadcast across batch dimensions.
    """
    def solve(self, spec: "IntegerNLPSpec", y) -> SolveResult: ...


class EnumerationBackend:
    """Full enumeration of feasible integer assignments.

    For each row of `spec.integer_problem.feasible_assignments()` we run
    the continuous NLP with the assignment encoded in the parametric tail.
    Batching is owned at this layer: we vmap septal's JAX-pure core
    `factory._solve_jit` over (assignment × start) directly, keeping the
    natural axes — no flat-batch replication, no reshape gymnastics. The
    best feasible row wins via the argmin chain in `_pick_best`.

    SOS1 / linear-constraint filtering happens at construction time, so
    the SQP never sees a-priori infeasible discrete rows.
    """
    def solve(self, spec, y) -> SolveResult:
        x0_per_asn = _screen_warmstarts(spec, y)            # (n_asn, n_starts, n_d)
        p_per_asn  = _augment_p(spec, y)                    # (n_asn, n_p_total)

        # Cached nested vmap from spec.__post_init__ — just dispatch.
        state = spec._solve_batched(x0_per_asn, p_per_asn)
        # state.x:           (n_asn, n_starts, n_d)
        # state.f_val:       (n_asn, n_starts)
        # state.feasibility: (n_asn, n_starts)

        # Per-start viability: KKT feasibility within tol AND iterate in bounds.
        # Inline the check — keeps integer_nlp free of evaluator-layer deps.
        lb = spec.continuous_factory.problem.lb
        ub = spec.continuous_factory.problem.ub
        in_bounds = jnp.all((state.x >= lb) & (state.x <= ub), axis=-1)
        viable    = (jnp.asarray(state.feasibility) <= spec.feasibility_tol) & in_bounds

        return _pick_best(spec, state.f_val, viable, state.x)


_DEFAULT_BACKEND = EnumerationBackend()


def solve_integer_nlp(spec: IntegerNLPSpec, y,
                      backend: IntegerBackend = _DEFAULT_BACKEND) -> SolveResult:
    """Solve one integer-aware parametric NLP for upstream input `y`.

    Theta-blind.  For batched-theta input use `solve_integer_nlp_batched`
    instead — that path caches the vmap wrapper at module import.
    """
    return backend.solve(spec, y)


def _solve_one(spec: IntegerNLPSpec, y) -> SolveResult:
    """Top-level trampoline so the vmap below has a stable function id."""
    return solve_integer_nlp(spec, y)


# Wrap the batched vmap in `jax.jit(static_argnums=(0,))` so the whole
# batched program compiles into one XLA executable per spec.  Without
# the jit, sub-routines (notably the screener's `lax.scan`) re-trace
# on every call — one "Compiling scan ..." event per DEUS iter.
# `static_argnums=(0,)` keys the cache on `id(spec)`.
_solve_one_batched = jax.jit(
    jax.vmap(_solve_one, in_axes=(None, 0)),
    static_argnums=(0,),
)


def solve_integer_nlp_batched(spec: IntegerNLPSpec, ys) -> SolveResult:
    """Batched solve over the theta axis of `ys`.

    Parameters
    ----------
    spec : IntegerNLPSpec
        Static across rows; hashed via `id(spec)` for the pjit cache.
    ys : jnp.ndarray, shape (n_theta, n_y)

    Returns
    -------
    SolveResult with shape (n_theta, …) on every field.

    First call per spec triggers one compile; subsequent calls hit the
    cached compiled program.
    """
    return _solve_one_batched(spec, ys)


# =============================================================================
# Internal helpers — pure jnp, JIT-traceable, vmap-composable
# =============================================================================

def _augment_p(spec: IntegerNLPSpec, y):
    """Per-assignment parametric tail: [y | assignment row].

    No `n_starts` replication — the inner vmap over starts shares `p`
    via `in_axes=(0, None)`.

    Returns
    -------
    p_per_asn : (n_assignments, n_p_base + n_int_total)
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
    """Per-assignment Sobol screen → top-n_starts warm-starts per assignment.

    For each row of `feasible_assignments`, evaluate the screener on every
    pool point with that row's parametric tail, take the n_starts pool
    points with the lowest screener score.

    If `spec.screener is None`, fall back to the first n_starts pool points
    broadcast across assignments (no per-assignment differentiation
    possible without an objective/penalty signal).

    Returns
    -------
    x0_per_asn : (n_assignments, n_starts, n_d_continuous)
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

    # spec._screen_batched is the cached `jax.vmap(_screen_one)` from
    # spec.__post_init__.  spec.screener itself uses `lax.scan` over the
    # pool, so the inner _screen_one calls it directly — wrapping the
    # screener in another vmap would double-iterate the pool axis.
    return spec._screen_batched(p_per_asn)                   # (n_asn, n_starts, n_d_continuous)


def _pick_best(spec: IntegerNLPSpec, obj, viable, x):
    """Argmin-chain: best start per assignment, then best feasible assignment.

    obj, viable : (n_asn, n_starts)
    x           : (n_asn, n_starts, n_d_continuous)

    Inner argmin selects the best warm-start per assignment (feasibility
    priority via the +big penalty). Outer argmin selects the best feasible
    assignment across the integer enumeration — that's the union-
    feasibility semantics encoded by SOS1 + the per-assignment NLP.
    """
    big = jnp.asarray(1.0e10, dtype=obj.dtype)
    ranked = jnp.where(viable, obj, obj + big)
    best_start = jnp.argmin(ranked, axis=-1)                 # (n_asn,)
    asn_obj    = jnp.take_along_axis(obj,    best_start[:, None],       axis=-1).squeeze(-1)
    asn_viable = jnp.take_along_axis(viable, best_start[:, None],       axis=-1).squeeze(-1)
    asn_x      = jnp.take_along_axis(x,      best_start[:, None, None], axis=-2).squeeze(-2)

    ranked_asn = jnp.where(asn_viable, asn_obj, asn_obj + big)
    best_asn = jnp.argmin(ranked_asn)                        # scalar

    return SolveResult(
        objective=asn_obj[best_asn],
        success=asn_viable[best_asn],
        x=asn_x[best_asn],
        assignment_idx=best_asn,
    )


# =============================================================================
# Post-solve helper — caller composes this with `solve_integer_nlp`
# =============================================================================

def splice_design_integers(x_continuous, spec: IntegerNLPSpec,
                           assignment_idx, n_design_full: int) -> jnp.ndarray:
    """Reinsert design-integer values into the continuous decision vector.

    `x_continuous` covers the non-integer slots of the design vector (the
    SQP only optimises over continuous dims). This helper places the
    chosen integer values from `feasible_assignments()[assignment_idx]`
    at the slots indicated by `spec.integer_problem.design_dims[i].slot`,
    producing the full `n_design_full`-length design vector for downstream
    consumers (e.g. the simulator's unit_op call).

    If there are no design integers, returns `x_continuous` unchanged.
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
