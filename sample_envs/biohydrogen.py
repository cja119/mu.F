"""
Environment function definition for the biohydrogen production case study.
"""

from functools import partial

from flax.core import FrozenDict
import jax.numpy as jnp
import jax

ENV_PARAMS = FrozenDict({
    "mu_max": 0.04765,  # 1/h
    "k_q": 0.6281,      # -
    "K_c": 0.0,         # mmol/L
    "mu_d": 0.008559,   # l/g/h
    "K_N": 50.0,        # mg/L
    "Y_NX": 244.6,      # mg/g
    "Y_qX": 1.723,      # -
    "Y_OX": 14.6,        # %·L/g/h (paper units — O expressed in 0-100 %)
    "Y_d": 26.22,        # %·L/g   (paper units — O expressed in 0-100 %)
    "Y_CX": 20.83,      # mmol/g
    "F_max": 0.1,       # L
    "O_Fed": 20.0,      # %  (feed oxygen concentration in percent)
    "C_Fed": 50.0,      # mmol/L
    "Y_HX": 2.34,       # mL/g
    })

SHAPE_DICT = {
    "X_SIZE": 7, # X, C, N, q, O, H, F
    "U_SIZE": 2, # N_Fed, alpha (fraction of remaining feed budget)
    "F_SIZE": 7, # dxdt, dCdt, dNdt, dqdt, dOdt, dHdt, dFdt
    "G_SIZE": 2, # N <= 150, O <= 100
    "Z_SIZE": 0,
    "L_SIZE": 1, # H(tf)
    "PHI_SIZE": 0,
}

@partial(jax.jit, static_argnums=(0,))
def simulator(
    param_dict, node,  x: jnp.ndarray, u: jnp.ndarray, z: jnp.ndarray = None
) -> jnp.ndarray:
    
    assert x.shape[-1] == SHAPE_DICT["X_SIZE"]
    assert u.shape[-1] == SHAPE_DICT["U_SIZE"]

    for dim in x.shape[:-1]:
        assert dim == 1
    for dim in u.shape[:-1]:
        assert dim == 1

    x = jnp.ravel(x)
    u = jnp.ravel(u)

    _dxdt = dxdt(param_dict, x, u)
    _dgdt = jnp.atleast_1d(dgdt(param_dict, x, u))
    _dldt = jnp.atleast_1d(dldt(param_dict, x, u))

    #_dldt = jax.lax.cond(
    #    node == 29,
    #    lambda: jnp.atleast_1d(dphidt(param_dict, x, u)),
    #    lambda: _dldt,
    #)

    return jnp.concatenate([_dxdt, _dgdt, _dldt], axis=0)

@partial(jax.jit, static_argnums=(0,))
def dxdt(param_dict,  x: jnp.ndarray, u: jnp.ndarray):

    # Guard q (x[3]) against zero: k_q/q diverges as q→0, producing NaN in VJP
    _q_safe = jnp.maximum(x[3], 1e-8)
    # Guard C denominator: when K_c=0.0 (as set in ENV_PARAMS), x[1]/(0+x[1]) = 0/0 at C=0
    _t1 =  param_dict['mu_max'] * (1 - param_dict['k_q'] / _q_safe) * (x[1] / (param_dict['K_c'] + x[1] + 1e-8))
    _t2 =  x[2] / (param_dict['K_N'] + x[2])
    _t3 = x[4] / (x[4]**2 + 0.1) ** 0.5  # paper formulation: O in 0-100 %, smoothing 0.1
    # alpha reparametrisation: u[1] is fraction of remaining feed budget [0,1]
    _F_in = u[1] * jnp.maximum(param_dict['F_max'] - x[6], 0.0) / 24

    dXdt = (
        x[0] * _t1 - param_dict['mu_d'] * x[0] ** 2
    )

    dCdt = (
        - param_dict['Y_CX'] * x[0] * _t1 + _F_in * param_dict['C_Fed']
    )

    dNdt = (
        - param_dict['Y_NX'] * x[0] * _t2 * param_dict['mu_max'] + _F_in * u[0]
    )

    dqdt = (
        param_dict['Y_qX'] * _t2 * param_dict['mu_max'] - _t1 * x[3]
    )

    dOdt = (
        param_dict['Y_OX'] * x[0] * _t2 - param_dict['Y_d'] * x[0]**2 * _t3 + param_dict['O_Fed'] * _F_in
    )

    # f(N) switch: H₂ only produced when culture nitrate N < 100 mg/L (Eq 1g)
    _delta_N = x[2] - 100.0
    _f_N = 0.5 * (jnp.sqrt(_delta_N**2 + 1e-12) - _delta_N) / jnp.sqrt(_delta_N**2 + 0.1)

    dHdt = (
        param_dict['Y_HX'] * x[0] * (1 - _t3) * _f_N
    )

    dFdt = (
        _F_in
    )

    return jnp.concatenate([
        jnp.ravel(dXdt),jnp.ravel(dCdt), jnp.ravel(dNdt), jnp.ravel(dqdt), 
        jnp.ravel(dOdt), jnp.ravel(dHdt), jnp.ravel(dFdt)
    ], axis=0)

@partial(jax.jit, static_argnums=(0,))
def dgdt(param_dict,  x: jnp.ndarray, u: jnp.ndarray):
    g_nitrogen = - jnp.maximum(x[2] - 150.0, 0) / 150.0  # Culture nitrate N <= 150 mg/L
    g_oxygen = - jnp.maximum(x[4] - 100.0, 0) / 100.0  # O <= 100 %
    return jnp.concatenate([jnp.ravel(g_nitrogen), jnp.ravel(g_oxygen)], axis=0)

@partial(jax.jit, static_argnums=(0,))
def dldt(param_dict,  x: jnp.ndarray, u: jnp.ndarray):
    _delta_N = x[2] - 100.0
    _t3 = x[4] / (x[4]**2 + 0.1) ** 0.5
    _f_N = 0.5 * (jnp.sqrt(_delta_N**2 + 1e-12) - _delta_N) / jnp.sqrt(_delta_N**2 + 0.1)
    return -jnp.array([param_dict['Y_HX'] * x[0] * (1 - _t3) * _f_N / 1000])

@partial(jax.jit, static_argnums=(0,))
def dphidt(param_dict,  x: jnp.ndarray, u: jnp.ndarray):
    return  - jnp.expand_dims(x[5], axis=0) / 24.0 # Normalize by per hour. This is a bit hacky, need to integrate terminal reward logic (at the per-node level?)