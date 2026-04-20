"""
Utility functions for constraints.

Two groups in this file:

1. **Generic helpers** — standardisation, classifier masking, successor-input
   extraction, likelihood-bound computations.  Read by every evaluator.

2. **Evaluator scaffolding** — config glue (`shaping_function`), cfg-driven
   Sobol starts (`initial_guess`), per-direction bounds extraction
   (`get_backward_bounds`, `get_forward_bounds`).  Plus re-exports of
   `generate_initial_guess`, `determine_batches`, `create_batches` from
   `mu_F.solvers.utilities` so evaluator code has a single import site.

The evaluator scaffolding lived in `constraints/pmap_scaffolding.py` until
this file absorbed it.
"""
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

def standardise_inputs(graph, succ_inputs, out_node, input_indices):
    """
    Standardises the inputs
    """
    if out_node is not None:
        standardiser = graph.nodes[out_node]['classifier_x_scalar']
    else:
        standardiser = graph.graph['classifier_x_scalar']

    if standardiser is None: return succ_inputs
    else:
        mean, std = standardiser.mean, standardiser.std
        return (succ_inputs - mean[input_indices].reshape(1,-1)) / std[input_indices].reshape(1,-1)


def standardise_model_decisions(graph, decisions, out_node):
    """
    Standardises the decisions
    """
    if out_node is None:
        standardiser = graph.graph['classifier_x_scalar']
    else:
        standardiser = graph.nodes[out_node]['classifier_x_scalar']
    if standardiser is None: return decisions
    else:
        mean, std = standardiser.mean, standardiser.std
        return [(decision - mean[:].reshape(1,-1)) / std[:].reshape(1,-1) for decision in decisions]
    

@lru_cache(maxsize=None)
def _cached_masked_classifier(classifier, ndim, fix_ind_tuple, aux_ind_tuple):
    fix_ind = np.array(fix_ind_tuple)
    aux_ind = np.array(aux_ind_tuple)
    def masked_classifier(x, y):
        input_ = construct_input(x, y, fix_ind, aux_ind, ndim)
        return classifier(input_.reshape(1,-1)).squeeze()
    return jit(masked_classifier)


def mask_classifier(classifier: Callable, ndim, fix_ind, aux_ind):
    """
    Masks the classifier
    - y corresponds to those indices that are fixed
    - x corresponds to those indices that are optimised
    """
    return _cached_masked_classifier(
        classifier,
        ndim,
        tuple(int(i) for i in fix_ind),
        tuple(int(i) for i in aux_ind)
    )


def construct_input(
        x: jnp.ndarray, 
        y: jnp.ndarray, 
        fix_ind: jnp.ndarray, 
        aux_ind: jnp.ndarray, 
        ndim: int
    ) -> jnp.ndarray:

    """
    Constructs the input to the classifier
    - y corresponds to those indices that are fixed
        assumed to be of shape (1, len(fix_ind) + len(aux_ind))
    - x corresponds to those indices that are optimised
        assumed to be of shape (len(decision_vars), ), such that 
        len(x) + len(y) = ndim
    - fix_ind are the indices of the fixed (design or input) variables
    - aux_ind are the indices of the fixed auxiliary variables 
    - ndim is the total number of dimensions 
      assumed to be len(x) + len(y)
    :return: the constructed input of shape (ndim, )
    """
    # Initialize input_ with zeros or any placeholder value
    input_ = jnp.zeros(ndim)
    
    # Create a mask for positions not in fix_ind and aux_ind
    total_indices = np.arange(ndim)
    opt_ind = np.delete(total_indices, np.concatenate([fix_ind, aux_ind]).astype(int))
    
    # Assign values from x to input_ at positions not in fix_ind and aux_ind
    input_ = input_.at[opt_ind].set(x.squeeze()) 
    
    # Assign values from y to input_ at positions in fix_ind and aux_ind
    if (y.shape[1] >= len(fix_ind)): # note that this will not be the case if graph_wide_problem is solved at a Node
        if (fix_ind.size != 0): input_ = input_.at[fix_ind].set(y[0,:len(fix_ind)])
        if aux_ind.size != 0: input_ = input_.at[aux_ind].set(y[0,len(fix_ind):])
        
    return input_

def get_successor_inputs(graph, node, outputs):
    """
    Gets the inputs from the predecessors
    """
    succ_inputs = {}
    for succ in graph.successors(node):
        output_indices = graph.edges[node, succ]['edge_fn']
        if outputs.ndim < 2: outputs = outputs.reshape(-1, 1)
        if outputs.ndim < 3: outputs = jnp.expand_dims(outputs, axis=0)
        succ_inputs[succ]= output_indices(outputs).reshape(outputs.shape[1],-1)
    return succ_inputs




def lower_bound_fn(
    constraint_evals: jnp.ndarray, samples: int, confidence: float
) -> jnp.ndarray:
    # compute the lower bound of the likelihood
    # constraint_evals: the constraint evaluations
    # samples: the number of samples we used
    # confidence: the desired confidence level

    assert confidence <= 1, "Confidence level must be equal to or less than 1"
    assert confidence >= 0, "Confidence level must be equal to or greater than 0"

    # compute the average constraint evaluation
    F_vioSA = jnp.mean(constraint_evals)

    # compute alpha and beta for the beta distribution
    alpha = samples + 1 - samples * F_vioSA
    b_ta = samples * F_vioSA + 1e-8

    # compute the lower bound of the likelihood as the inverse of the CDF of the beta distribution
    conf = confidence
    betaDist = beta(alpha, b_ta)
    F_LB = betaDist.ppf(conf)

    return 1 - F_LB


def upper_bound_fn(
    constraint_evals: jnp.ndarray, samples: int, confidence: float
) -> jnp.ndarray:
    # compute the lower bound of the likelihood
    # constraint_evals: the constraint evaluations
    # samples: the number of samples we used
    # confidence: the desired confidence level

    assert confidence <= 1, "Confidence level must be equal to or less than 1"
    assert confidence >= 0, "Confidence level must be equal to or greater than 0"

    # compute the average constraint evaluation
    F_vioSA = jnp.mean(constraint_evals)

    # compute alpha and beta for the beta distribution
    alpha = samples - samples * F_vioSA
    b_ta = samples * F_vioSA + 1

    # compute the upper bound of the likelihood as the inverse of the CDF of the beta distribution
    conf = confidence
    betaDist = beta(alpha, b_ta)
    F_LB = betaDist.ppf(1 - conf)

    return 1 - F_LB



def shaping_function(x, cfg):
    """
    Sign-flip the objective according to `cfg.samplers.notion_of_feasibility`.

    "positive = feasible" negates so that minimisation drives towards the
    feasible region; "negative = feasible" leaves the sign alone.
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
    Per-successor reduced-space decision bounds.

    Drops the indices held fixed by the (node -> succ) edge (inputs + aux)
    from `extendedDS_bounds`, optionally applying `classifier_x_scalar`
    standardisation first.  Safe to hoist outside pmap (static w.r.t. sample
    data).
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
        if cfg.solvers.standardised:
            decision_bounds = standardise_model_decisions(graph, decision_bounds, succ)
        decision_bounds = [
            jnp.delete(bound, np.hstack([input_indices, aux_indices]).astype(int), axis=1)
            for bound in decision_bounds
        ]
        backward_bounds[succ] = decision_bounds
    return backward_bounds


def get_forward_bounds(graph, node, cfg):
    """
    Per-predecessor decision bounds (pred's full NLP space — no reduction).

    Static w.r.t. the inputs flowing through the edge; same role as
    `get_backward_bounds` but for the forward direction.
    """
    if node is None:
        return None
    forward_bounds = {}
    for pred in graph.predecessors(node):
        decision_bounds = graph.nodes[pred]["extendedDS_bounds"].copy()
        if cfg.solvers.standardised:
            decision_bounds = standardise_model_decisions(graph, decision_bounds, pred)
        lb = jnp.asarray(decision_bounds[0]).reshape(-1)
        ub = jnp.asarray(decision_bounds[1]).reshape(-1)
        forward_bounds[pred] = [lb, ub]
    return forward_bounds
