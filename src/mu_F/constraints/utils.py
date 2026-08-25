"""Constraint utilities: generic helpers plus evaluator scaffolding."""
from typing import Callable
import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import lru_cache
from scipy.stats import beta

from mu_F.solvers.utilities import (
    generate_initial_guess,
    determine_batches,
    create_batches,
)


# ---------------------------------------------------------------------------
# Pmap padding helpers — keep pmap width constant across calls.
# ---------------------------------------------------------------------------

def pad_to_multiple(arr, W: int, axis: int = 0):
    """
    Pad arr along axis up to the next multiple of W using replicas of row 0,
    so padded lanes carry a genuine sample and never trip tracing.
    """
    n_real = arr.shape[axis]
    remainder = n_real % W
    if remainder == 0:
        return arr, n_real, 0
    n_pad = W - remainder

    slicer = [slice(None)] * arr.ndim
    slicer[axis] = slice(0, 1)
    row0 = arr[tuple(slicer)]

    pad_shape = list(arr.shape)
    pad_shape[axis] = n_pad
    pad_block = jnp.broadcast_to(row0, tuple(pad_shape))
    return jnp.concatenate([arr, pad_block], axis=axis), n_real, n_pad


def batch_mask(n_real: int, total: int) -> jnp.ndarray:
    """Boolean mask of length `total`: True on `[0, n_real)`, False after."""
    return jnp.arange(total) < n_real


def poison_padded(arr: jnp.ndarray, mask: jnp.ndarray, fill=jnp.nan) -> jnp.ndarray:
    """
    Set arr[i, ...] = fill wherever mask[i] is False, reshaping the 1D mask
    to broadcast over the trailing dims of arr at any rank.
    """
    bcast = mask.reshape((mask.shape[0],) + (1,) * (arr.ndim - 1))
    return jnp.where(bcast, arr, fill)


# ---------------------------------------------------------------------------
# Scaling helpers (deprecated)
# ---------------------------------------------------------------------------
# Identity passthroughs retained for call-site parity; surrogates self-scale.

def standardise_inputs(graph, succ_inputs, out_node, input_indices):
    """Deprecated — identity passthrough.  Surrogates self-scale now."""
    return succ_inputs


def standardise_model_decisions(graph, decisions, out_node):
    """Deprecated — identity passthrough.  Surrogates self-scale now."""
    return decisions


def destandardise_model_decisions(decisions, graph, node, cfg):
    """Deprecated — identity passthrough.  Solver already runs in real space."""
    return decisions
    

@lru_cache(maxsize=None)
def _cached_masked_surrogate(callable_, n_heads, ndim,
                             fix_ind_tuple, aux_ind_tuple, int_ind_tuple,
                             aggregator, n_y):
    """
    Cached factory building the jit'd masked-surrogate closure for septal.
    """
    fix_ind = np.array(fix_ind_tuple, dtype=int)
    aux_ind = np.array(aux_ind_tuple, dtype=int)
    int_ind = np.array(int_ind_tuple, dtype=int)
    n_fix_aux = int(fix_ind.size + aux_ind.size)
    n_int     = int(int_ind.size)
    # Size of the y slot in p_aug: fix/aux count by default, explicit n_y
    # (surrogate output dim) for the vector_diff equality target.
    n_y_eff = int(n_y) if n_y is not None else n_fix_aux

    # Full-space slots that are not fix/aux, split into continuous (optimised
    # by septal) and integer (read from the parametric tail at solve time).
    opt_ind_full = np.delete(
        np.arange(ndim),
        np.concatenate([fix_ind, aux_ind]).astype(int),
    )
    cont_ind = np.setdiff1d(opt_ind_full, int_ind)
    pos_cont_in_opt = np.array(
        [np.where(opt_ind_full == i)[0][0] for i in cont_ind], dtype=int,
    )
    pos_int_in_opt = np.array(
        [np.where(opt_ind_full == i)[0][0] for i in int_ind], dtype=int,
    )

    @jit
    def masked_surrogate(x_red, p_aug):
        # Parametric tail layout: [ y (n_y_eff) | integers (n_int) |
        # structural binaries (n_heads, one-hot under SOS1) ].
        # Promote 1-D p_aug (screener / one-call paths) to (1, n_p_total).
        p_aug = jnp.asarray(p_aug)
        if p_aug.ndim == 1:
            p_aug = p_aug.reshape(1, -1)

        y_upstream = p_aug[:, :n_y_eff]
        x_orig = jnp.zeros(opt_ind_full.size)
        x_orig = x_orig.at[pos_cont_in_opt].set(x_red.squeeze())
        if n_int:
            x_orig = x_orig.at[pos_int_in_opt].set(
                p_aug[0, n_y_eff:n_y_eff + n_int]
            )
        input_ = construct_input(x_orig, y_upstream, fix_ind, aux_ind, ndim)

        out = jnp.asarray(callable_(input_.reshape(1, -1))).reshape(-1)

        # Aggregator dispatch resolved at construction time (no traced control flow).
        if aggregator == 'scalar':
            return out.reshape(())
        if aggregator == 'vector':
            return out.reshape(-1)                           # raw K-vector (no reduction)
        if aggregator == 'onehot_sum':
            y_struct = p_aug[0, n_y_eff + n_int : n_y_eff + n_int + n_heads]
            return jnp.sum(y_struct * out)                   # Σ_k y_k · head[k]
        if aggregator == 'vector_diff':
            target = p_aug[0, :n_y_eff]                      # y IS the target here
            return out - target                              # equality residual
        raise ValueError(f"Unknown aggregator: {aggregator!r}")

    return masked_surrogate


def mask_surrogate(callable_: Callable, ndim,
                   fix_ind, aux_ind, int_ind=(),
                   n_heads: int = 0,
                   aggregator: str = None,
                   n_y: int = None) -> Callable:
    """
    Wrap any Surrogate into septal's (x_red, p_aug) signature, dispatching on
    the aggregator (scalar, vector, onehot_sum, vector_diff) and parametric tail layout.
    """
    if aggregator is None:
        aggregator = 'scalar' if int(n_heads) == 0 else 'onehot_sum'
    return _cached_masked_surrogate(
        callable_,
        int(n_heads),
        int(ndim),
        tuple(int(i) for i in fix_ind),
        tuple(int(i) for i in aux_ind),
        tuple(int(i) for i in int_ind),
        str(aggregator),
        None if n_y is None else int(n_y),
    )


# ---------------------------------------------------------------------------
# Forward aux masking — match coupling inputs at the output, pin shared aux
# ---------------------------------------------------------------------------

def mask_aux(callable_: Callable, ndim,
             aux_ind, int_ind=(),
             n_heads: int = 0,
             aggregator: str = None,
             n_g: int = None) -> Callable:
    """
    Forward masker built per predecessor by ForwardEvaluator: the coupling
    inputs are matched at the surrogate output, the shared aux is pinned at its
    input slots. Leaves the backward mask_surrogate path untouched.
    """
    if aggregator is None:
        aggregator = 'scalar' if int(n_heads) == 0 else 'onehot_sum'
    return _cached_masked_aux(
        callable_,
        int(ndim),
        tuple(int(a) for a in aux_ind),
        tuple(int(i) for i in int_ind),
        int(n_heads),
        str(aggregator),
        int(n_g),
    )


@lru_cache(maxsize=None)
def _cached_masked_aux(callable_, ndim, aux_ind_tuple, int_ind_tuple,
                       n_heads, aggregator, n_g):
    """
    Cached factory for the forward masker; aux is pinned from its own block of
    the parametric tail, so it never collides with the match target.
    """
    aux_ind = np.array(aux_ind_tuple, dtype=int)
    int_ind = np.array(int_ind_tuple, dtype=int)
    n_aux, n_int = int(aux_ind.size), int(int_ind.size)

    # Free slots: design + pred's own inputs (neither pinned nor tail-valued).
    opt_ind = np.delete(np.arange(ndim), np.concatenate([aux_ind, int_ind]).astype(int))

    @jit
    def masked_aux(x_red, p_aug):
        # Tail layout: [ target (n_g) | aux (n_aux) | integers (n_int) | heads (n_heads) ].
        p_aug = jnp.atleast_2d(jnp.asarray(p_aug))
        assert p_aug.shape[-1] == n_g + n_aux + n_int + n_heads, "forward tail width mismatch"
        a, b = n_g, n_g + n_aux
        target = p_aug[0, :a]

        x_full = jnp.zeros(ndim).at[opt_ind].set(x_red.squeeze())
        if n_aux: x_full = x_full.at[aux_ind].set(p_aug[0, a:b])              # pin shared aux
        if n_int: x_full = x_full.at[int_ind].set(p_aug[0, b:b + n_int])      # integers from tail

        out = jnp.asarray(callable_(x_full.reshape(1, -1))).reshape(-1)

        if aggregator == 'scalar':
            return out.reshape(())
        if aggregator == 'vector_diff':
            return out - target                                              # match inputs at output
        if aggregator == 'onehot_sum':
            heads = p_aug[0, b + n_int:b + n_int + n_heads]
            return jnp.sum(heads * out)
        raise ValueError(f"Unknown aggregator: {aggregator!r}")

    return masked_aux


def construct_input(
        x: jnp.ndarray,
        y: jnp.ndarray,
        fix_ind: jnp.ndarray,
        aux_ind: jnp.ndarray,
        ndim: int
    ) -> jnp.ndarray:

    """
    Build the ndim classifier input, placing optimised values x at the free
    slots and fixed values y at the fix/aux slots.
    """
    input_ = jnp.zeros(ndim)

    # Free (optimised) slots are everything not held by fix_ind / aux_ind.
    total_indices = np.arange(ndim)
    opt_ind = np.delete(total_indices, np.concatenate([fix_ind, aux_ind]).astype(int))

    input_ = input_.at[opt_ind].set(x.squeeze())

    # y only present when not solving a graph-wide problem at the node.
    if (y.shape[1] >= len(fix_ind)):
        if (fix_ind.size != 0): input_ = input_.at[fix_ind].set(y[0,:len(fix_ind)])
        if aux_ind.size != 0: input_ = input_.at[aux_ind].set(y[0,len(fix_ind):])

    return input_

def get_successor_inputs(graph, node, outputs):
    """
    Extract each successor's inputs from this node's outputs via the edge map.
    """
    lead = outputs.shape[:-1]  # real leading axes, captured before any promotion
    if outputs.ndim < 2:
        outputs = outputs.reshape(-1, 1)
    if outputs.ndim < 3:
        outputs = jnp.expand_dims(outputs, axis=0)  # the doubly-vmapped edge_fn needs (N, S, F)

    succ_inputs = {}
    for succ in graph.successors(node):
        edge_fn = graph.edges[node, succ]['edge_fn']
        succ_inputs[succ] = edge_fn(outputs).reshape(*lead, -1)  # restore real leading axes; flatten features
    return succ_inputs




def lower_bound_fn(
    constraint_evals: jnp.ndarray, samples: int, confidence: float
) -> jnp.ndarray:
    """
    Beta-distribution lower confidence bound on the satisfaction likelihood.
    """
    assert confidence <= 1, "Confidence level must be equal to or less than 1"
    assert confidence >= 0, "Confidence level must be equal to or greater than 0"

    F_vioSA = jnp.mean(constraint_evals)

    # Beta distribution shape parameters.
    alpha = samples + 1 - samples * F_vioSA
    b_ta = samples * F_vioSA + 1e-8

    conf = confidence
    betaDist = beta(alpha, b_ta)
    F_LB = betaDist.ppf(conf)

    return 1 - F_LB


def upper_bound_fn(
    constraint_evals: jnp.ndarray, samples: int, confidence: float
) -> jnp.ndarray:
    """
    Beta-distribution upper confidence bound on the satisfaction likelihood.
    """
    assert confidence <= 1, "Confidence level must be equal to or less than 1"
    assert confidence >= 0, "Confidence level must be equal to or greater than 0"

    F_vioSA = jnp.mean(constraint_evals)

    # Beta distribution shape parameters.
    alpha = samples - samples * F_vioSA
    b_ta = samples * F_vioSA + 1

    conf = confidence
    betaDist = beta(alpha, b_ta)
    F_LB = betaDist.ppf(1 - conf)

    return 1 - F_LB



def shaping_function(x, cfg):
    """
    Sign-flip the objective per cfg.samplers.notion_of_feasibility so that
    minimisation drives towards the feasible region.
    """
    if cfg.samplers.notion_of_feasibility == 'positive':
        return -x
    elif cfg.samplers.notion_of_feasibility == 'negative':
        return x
    raise ValueError(
        f"Unknown notion_of_feasibility: {cfg.samplers.notion_of_feasibility!r}"
    )


def initial_guess(cfg_solvers, bounds):
    """Draw `cfg_solvers.n_starts` Sobol points inside `bounds`."""
    n_d = len(bounds[0])
    return generate_initial_guess(cfg_solvers.n_starts, n_d, bounds)


def get_backward_bounds(graph, node, cfg):
    """
    Per-successor reduced-space decision bounds, dropping the indices the
    (node -> succ) edge holds fixed. Static w.r.t. sample data.
    """
    if node is None:
        return None
    backward_bounds = {}
    for succ in graph.successors(node):
        n_d = graph.nodes[succ]['n_design_args']
        input_indices = np.copy(np.array(
            [n_d + inp for inp in graph.edges[node, succ]['input_indices']]
        ))
        aux_indices = np.copy(np.array(
            [inp for inp in graph.edges[node, succ]['auxiliary_indices']]
        ))
        decision_bounds = graph.nodes[succ]["extendedDS_bounds"].copy()
        # Bounds stay in real-world units; the classifier callable self-scales.
        decision_bounds = [
            jnp.delete(bound, np.hstack([input_indices, aux_indices]).astype(int), axis=1)
            for bound in decision_bounds
        ]
        backward_bounds[succ] = decision_bounds
    return backward_bounds


def get_forward_bounds(graph, node, cfg):
    """
    Per-predecessor decision bounds in the predecessor's full NLP space.
    The forward-direction counterpart to get_backward_bounds.
    """
    if node is None:
        return None
    forward_bounds = {}
    for pred in graph.predecessors(node):
        decision_bounds = graph.nodes[pred]["extendedDS_bounds"].copy()
        # Real-world units; forward Surrogate self-scales.
        lb = jnp.asarray(decision_bounds[0]).reshape(-1)
        ub = jnp.asarray(decision_bounds[1]).reshape(-1)
        forward_bounds[pred] = [lb, ub]
    return forward_bounds
