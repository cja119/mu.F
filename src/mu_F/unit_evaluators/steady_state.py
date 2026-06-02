"""Steady-state unit evaluator dispatching to the case-study vector field."""
import jax.numpy as jnp
from functools import partial
from mu_F.unit_evaluators.explicit_fn import case_studies

def unit_steady_state(design_params, u, aux, dd_params, uncertain_params, cfg, node, graph=None):
    """
    Evaluate a node's steady-state response by calling its
    case-study term on the broadcast design, input and aux arrays.
    """
    if design_params.ndim < 2:
        design_params = jnp.expand_dims(design_params, axis=0)

    if u.ndim < 1:
        u = jnp.expand_dims(u, axis=0)

    if u.ndim < 2:
        u = jnp.expand_dims(u, axis=0)

    if aux.ndim < 1:
        aux = jnp.expand_dims(aux, axis=0)

    if aux.ndim < 2:
        aux = jnp.expand_dims(aux, axis=0)

    if dd_params.ndim < 2:
        dd_params = jnp.expand_dims(dd_params, axis=0)

    # params passed to the vector field
    collected_p = jnp.concatenate([u, dd_params], axis=-1)

    # select the dynamics for this case study
    if cfg.case_study.eval_cost:
        # cfg-driven case studies: factory takes cfg, returns the case-study term
        term = partial(case_studies[cfg.case_study.case_study](cfg), node=node)
    else:
        term = case_studies[cfg.case_study.case_study][node]

    return term(cfg, design_params, collected_p, aux, uncertain_params).squeeze()
