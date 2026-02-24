"""
A general interface for pc-gym problems
"""
from functools import partial

from pcgym import make_env
from flax.core import FrozenDict, unfreeze
import jax.numpy as jnp
import numpy as np
import jax

ENV_PARAMS = FrozenDict({
    "cstr": {
        "case_study": "cstr",
        "N": 100,
        "tsim": 25,
        "SP": {
            "Ca": tuple([0.85]*10 + [0.9]*10)
        },
        "a_space": {
            "low": (295,),
            "high": (302,)
        },
        "o_space": {
            "low": (0.7, 300, 0.8),
            "high": (1.0, 350, 0.9)
        },
        "x0": (0.8, 330, 0.8),
        "model": "cstr_ode",
    }
})

SHAPE_DICTS = {
    'cstr': {
        "X_SIZE": 3,
        "U_SIZE": 1,
        "F_SIZE": 3,
        "G_SIZE": 2,
        "Z_SIZE": 1,
        "L_SIZE": 1,
        "PHI_SIZE": 0
        },
}    

@partial(jax.jit, static_argnums=(0,))
def simulator(param_dict, node, x: jnp.ndarray, u: jnp.ndarray, z: jnp.ndarray = None
) -> jnp.ndarray:
    
    constraint_list =  CONS_HOLDER[param_dict['case_study']]
    pc_constriants = make_pc_constraints(constraint_list, param_dict)
    env_params = unfreeze(param_dict)

    env_params['x0'] = np.array(env_params["x0"])
    env_params["a_space"]["low"] = np.array(env_params["a_space"]["low"])
    env_params["a_space"]["high"] = np.array(env_params["a_space"]["high"])
    env_params["o_space"]["low"] = np.array(env_params["o_space"]["low"])
    env_params["o_space"]["high"] = np.array(env_params["o_space"]["high"])

    for k in env_params["SP"]:
        env_params["SP"][k] = np.array(env_params["SP"][k])

    pc_env = make_env(env_params)
    pc_env.reset()
    pc_env.state = x
    pc_env.t = node

    # Injecting disturbances < - Need to check injecting at right time(?)
    #for i, k in enumerate(pc_env.model.info()["disturbances"]):
    #    pc_env.disturbances[k][pc_env.t + 1] = z[:, i]

    x_n, rew, done, term, info = pc_env.step(u)
    cons_vals = pc_constriants(x_n, u)

    out = jnp.concatenate([x_n, cons_vals, rew.reshape(-1, 1)], axis=0)

    return out


# -------------------------------------------------------------------------------- #
# ----------------------------- Constraint Functions ----------------------------- #
# -------------------------------------------------------------------------------- #
def cstr_cons_1(param_dict, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    return (x[1] - param_dict['t_lower']) / param_dict['t_upper'] 

def cstr_cons_2(param_dict, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    return (param_dict['t_upper'] - x[1] ) / param_dict['t_upper']

# -------------------------------------------------------------------------------- #
# ------------------------------ Constraint Helpers ------------------------------ #
# -------------------------------------------------------------------------------- #

def make_pc_constraints(constraint_fns, param_dict):
    def cons(state, control):
        ctrl = jnp.ravel(control)
        vals = [cf(param_dict, state, ctrl) for cf in constraint_fns]
        return jnp.concatenate(vals, axis=0).reshape(-1, )
    return cons

CONS_HOLDER = {
    'cstr': [cstr_cons_1, cstr_cons_2]
}