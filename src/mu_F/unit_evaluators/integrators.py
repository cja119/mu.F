"""Diffrax-based integration of the case-study ODE dynamics."""

from typing import List

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import time
import logging
from omegaconf import DictConfig, OmegaConf

from functools import partial
import diffrax
from diffrax import ODETerm, SaveAt, diffeqsolve, DirectAdjoint
from diffrax import Tsit5

from mu_F.unit_evaluators.ode import case_studies


def unit_dynamics(design_params, u, aux, decision_dependent, uncertainty_params, cfg, node, graph=None):
    """
    Integrate a system of ODEs whose initial conditions are the input args.
    Design, decision-dependent and uncertainty params are passed to the field;
    redefine per case study if these assumptions do not hold.
    """

    if design_params.ndim < 2:
        design_params = jnp.expand_dims(design_params, axis=0)

    if decision_dependent.ndim < 2:
        decision_dependent = jnp.expand_dims(decision_dependent, axis=0)

    # defining the params to pass to the vector field
    params = jnp.hstack([design_params, decision_dependent, aux, uncertainty_params.reshape(1,-1)])

    # defining the dynamics
    if cfg.case_study.eval_cost:
        term = ODETerm(partial(case_studies[cfg.case_study.case_study](cfg), node=node))
        sizes = cfg.case_study.sizes
        pad = int(sizes.G_SIZE) + int(sizes.L_SIZE) + int(sizes.PHI_SIZE)
        u = jnp.concatenate([u, jnp.zeros(u.shape[:-1] + (pad,))], axis=-1) # Path constraints + stage cost both start at 0.
    else:
        term = ODETerm(case_studies[cfg.case_study.case_study][node])

    # defining the diffrax solver
    solver = dispatcher[cfg.model.integration.scheme]

    # defining saveat 
    saveat = SaveAt(t1=True) # just return the final time step

    # define step size controller for solver
    step_size_controller = dispatcher[cfg.model.integration.step_size_controller]

    try:
        start = time.time()
        return diffeqsolve(
        term,
        solver,
        cfg.model.integration.t0,
        cfg.model.integration.tf,
        cfg.model.integration.dt0,
        y0=u,
        args=params,
        max_steps=cfg.model.integration.max_steps,
        stepsize_controller=step_size_controller,
        saveat=saveat,
        adjoint=DirectAdjoint(),
    ).ys[
        :, :
    ][-1,:]  # t x n_components
    except:
        try:
            return diffeqsolve(
            term,
            solver,
            cfg.model.integration.t0,
            cfg.model.integration.tf,
            cfg.model.integration.dt0,
            y0=u,
            args=params,
            max_steps=cfg.model.integration.max_steps * 500,
            stepsize_controller=step_size_controller,
            saveat=saveat,
            adjoint=DirectAdjoint(),
            )

        except: # case study specific splodge
            return diffeqsolve(
            term,
            solver,
            cfg.model.integration.t0,
            cfg.model.integration.tf,
            cfg.model.integration.dt0,
            y0=jnp.hstack([u.reshape(1,-1), jnp.zeros(1).reshape(1,1)]).squeeze(),
            args=params,
            max_steps=cfg.model.integration.max_steps,
            stepsize_controller=step_size_controller,
            saveat=saveat,
            adjoint=DirectAdjoint(),
        ).ys[
            :, :
        ][-1,:]  # t x n_components
    finally:
        logging.info(f"Integration took {time.time() - start} seconds")

# defining the dispatcher for the dynamics

dispatcher = {
    "tsit5": Tsit5(),
    "dopri8": diffrax.Dopri8(),
    "Kvaerno5": diffrax.Kvaerno5(),
    "pid": diffrax.PIDController(rtol=1e-2, atol=1e-2),
}

