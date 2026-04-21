"""
A general interface for pc-gym problems
"""

from functools import partial
from dataclasses import dataclass
from flax.core import FrozenDict, unfreeze
import jax.numpy as jnp
import numpy as np
import jax


# ---------------------------------------------------------------------------
# Piecewise-smooth log:  log(z)  for  z >= z0,  C²-smooth quadratic for  z < z0.
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(1,))
def _smooth_log(z, z0=0.02):
    f0      = jnp.log(z0)
    f1      = 1.0 / z0
    f2      = -1.0 / (z0 * z0)
    delta   = z - z0
    fallback = f0 + f1 * delta + 0.5 * f2 * delta * delta
    # jnp.maximum guards against NaN in the non-taken branch during JAX tracing
    return jnp.where(z >= z0, jnp.log(jnp.maximum(z, 1e-30)), fallback)

ENV_PARAMS = FrozenDict(
    {
        "cstr_ode": {
            "case_study": "cstr_ode",
            "N": 100,
            "tsim": 26,
            "SP": {"Ca": tuple([0.85] * 10 + [0.9] * 10 + [0.8] * 10)},
            "a_space": {"low": (295,), "high": (302,)},
            "o_space": {"low": (0.7, 300, 0.8), "high": (1.0, 350, 0.9)},
            "x0": (0.8, 330, 0.8),
            "model": "cstr_ode",
            "integration_method": "jax",
            "maximise_reward": False,
            "normalise_a": False,
            "normalise_o": False,
            "disturbance_bounds": {"low": (325, 0.95), "high": (375, 1.05)},
            "disturbance_keys": ('Ti', 'Caf'),
            't_lower': 315,
            't_upper': 331,
        }
    }
)

SHAPE_DICTS = {
    "cstr_ode": {
        "X_SIZE": 2,
        "U_SIZE": 1,
        "F_SIZE": 2,
        "G_SIZE": 2,
        "Z_SIZE": 2,
        "L_SIZE": 1,
        "PHI_SIZE": 0,
    },
}

@partial(jax.jit, static_argnums=(0,))
def simulator(
    param_dict, node, x: jnp.ndarray, u: jnp.ndarray, z: jnp.ndarray = None
) -> jnp.ndarray:
    import importlib

    mod = importlib.import_module("pcgym.model_classes")
    ModelCls = getattr(mod, param_dict["case_study"])   # e.g. "cstr"
    model = ModelCls(int_method="jax")

    x = jnp.ravel(x)
    u = jnp.ravel(u)

    dxdt = model(x, u).squeeze()
    dgdt = jnp.concatenate([jnp.atleast_1d(c(param_dict, x, u)) for c in CONS_HOLDER[param_dict["case_study"]]], axis=0)
    sp_ca = jnp.asarray(param_dict["SP"]["Ca"])
    rwd = _smooth_log(jnp.abs(jnp.take(sp_ca, node) - x[0]))

    return jnp.concatenate([jnp.ravel(dxdt), jnp.ravel(dgdt), jnp.ravel(rwd)], axis=0)



# -------------------------------------------------------------------------------- #
# ----------------------------- Constraint Functions ----------------------------- #
# -------------------------------------------------------------------------------- #
@partial(jax.jit, static_argnums=(0,))
def cstr_cons_1(param_dict, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    return - jnp.square(jnp.minimum(0.0, (x[1] - param_dict["t_lower"]) / param_dict["t_upper"])) 

@partial(jax.jit, static_argnums=(0,))
def cstr_cons_2(param_dict, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    return - jnp.square(jnp.minimum(0.0, (param_dict["t_upper"] - x[1]) / param_dict["t_upper"]))


# -------------------------------------------------------------------------------- #
# ------------------------------ Constraint Helpers ------------------------------ #
# -------------------------------------------------------------------------------- #


CONS_HOLDER = {"cstr_ode": [cstr_cons_1, cstr_cons_2]}
