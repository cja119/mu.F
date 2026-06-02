"""ODE systems whose feasibility is characterised across the case studies."""

from jax import jit
import jax.numpy as jnp


@jit
def serial_mechanism_vc_batch_dynamics_u1(
    t: float, state: jnp.ndarray, parameters: jnp.ndarray
):
    """
    Serial mechanism 2A ->_{k1} B ->_{k2} C, constant volume batch mode.
    Dynamics normalised to the selected batch time; design vars Temp and tf.
    """
    # component_concentrations
    Ca = state[0]
    Cb = state[1]
    Cc = state[2]

    # parameters
    k1, k2, tf1 = parameters[2], parameters[3], parameters[1]

    # normalised rates
    dCa = -2 * k1 * Ca**2
    dCb = k1 * Ca**2 - k2 * Cb
    dCc = k2 * Cb

    # differential equations
    dCa = dCa * tf1
    dCb = dCb * tf1
    dCc = dCc * tf1

    # jax.debug.print('dC {x}', x=[dCa, dCb, dCc])

    return jnp.array([dCa, dCb, dCc])



@jit
def serial_mechanism_vc_batch_dynamics_u2(
    t: float, state: jnp.ndarray, parameters: jnp.ndarray
):
    """
    Serial mechanism 2A ->_{k1} B ->_{k2} C, constant volume batch mode.
    Dynamics normalised to the selected batch time; design vars Temp and tf.
    """
    # component_concentrations
    Ca = state[0]
    Cb = state[1]
    Cc = state[2]

    # parameters
    k1, k2, tf1 = parameters[2], parameters[3], parameters[1]

    # normalised rates
    dCa = -2 * k1 * Ca**2
    dCb = k1 * Ca**2 - k2 * Cb
    dCc = k2 * Cb

    # differential equations
    dCa = dCa * tf1
    dCb = dCb * tf1
    dCc = dCc * tf1

    # jax.debug.print('dC {x}', x=[dCa, dCb, dCc])

    return jnp.array([dCa, dCb, dCc])

def reactor_network_4(t: float, state: jnp.ndarray, parameters: jnp.ndarray):

    # component_concentrations
    Ci = state[0]
    Cj = state[1]
    Ck = state[2]
    Cl = state[3]

    # parameters
    k1, k2, k3, k4, tf1 = parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]
    
    # normalised rates
    dCi = -(k1 + k2) * Ci
    dCj = k1 * Ci - k3 * Cj
    dCk = k2 * Ci + k3 * Cj - k4 * Ck
    dCl = - k4 * Ck 

    # differential equations
    dCi = dCi * tf1
    dCj = dCj * tf1
    dCk = dCk * tf1
    dCl = dCl * tf1

    return jnp.array([dCi, dCj, dCk, dCl])

def reactor_network_5(t: float, state: jnp.ndarray, parameters: jnp.ndarray):

    # component_concentrations
    Ci = state[0]
    Cj = state[1]
    Ck = state[2]
    Cl = state[3]
    Cb = state[4]
    Cm = state[5]
    Cn = state[6]

    # parameters
    k1, k2, k3, k4, k5, k6, tf1 = parameters[0], parameters[1], parameters[2]
    
    # normalised rates
    # reaction_network4
    dCi = -(k1 + k2) * Ci
    dCj = k1 * Ci - k3 * Cj
    dCk = k2 * Ci + k3 * Cj - k4 * Ck
    dCl = - k4 * Ck 

    # reaction_network5
    dCb = -(k5 + k6) * Cj * Cb
    dCj +=  -(k5 + k6) * Cj * Cb
    dCm = k5 * Cj * Cb
    dCn = k6 * Cj * Cb

    # differential equations
    dCb = dCb * tf1
    dCj = dCj * tf1
    dCm = dCm * tf1
    dCn = dCn * tf1
    dCi = dCi * tf1
    dCk = dCk * tf1 
    dCl = dCl * tf1


    return jnp.array([dCi, dCj, dCk, dCl, dCb, dCm, dCn])


def cstr_ode(cfg):
    """Factory: returns a diffrax-shaped ODE term for the cstr case study.

    Mirrors `markov_process(env)` but reads sizes from `cfg.case_study.sizes`
    and dispatches the underlying step through `_make_cstr_step` (defined
    alongside the steady-state factory in unit_evaluators/explicit_fn.py),
    which closes over cfg-resolved constants and JIT-compiles the inner body.
    """
    from mu_F.unit_evaluators.explicit_fn import _make_cstr_step
    X_size = int(cfg.case_study.sizes.X_SIZE)
    U_size = int(cfg.case_study.sizes.U_SIZE)
    step = _make_cstr_step(cfg)

    def cstr_ode_fn(t: float, state: jnp.ndarray, parameters: jnp.ndarray, node=None):
        x = state[..., :X_size]
        u = parameters[..., :U_size]
        return step(x, u, node)

    return cstr_ode_fn


def waste_water_ode(cfg):
    """Factory: returns a diffrax-shaped ODE term for the waste_water case study.

    Bernard et al. (2001) AM2 anaerobic digestion. Unlike cstr, this case study
    *consumes* the uncertainty / disturbance vector z = [S_1in, S_2in, Z_in, C_in],
    and (when global_n_aux_args > 0) the biomass retention fraction α from the
    global aux slot.
    Layout of `parameters` (set in unit_dynamics):
        [design_args | dd | aux | uncertainty]  with n_dd = 0
    where `aux` carries the global auxiliary variables (here: α).
    """
    from mu_F.unit_evaluators.explicit_fn import _make_waste_water_step
    X_size = int(cfg.case_study.sizes.X_SIZE)
    U_size = int(cfg.case_study.sizes.U_SIZE)
    Z_size = int(cfg.case_study.sizes.Z_SIZE)
    A_size = int(cfg.case_study.global_n_aux_args)
    step = _make_waste_water_step(cfg)

    def waste_water_ode_fn(t: float, state: jnp.ndarray, parameters: jnp.ndarray, node=None):
        x   = state[..., :X_size]
        u   = parameters[..., :U_size]
        aux = parameters[..., U_size:U_size + A_size]
        z   = parameters[..., U_size + A_size:U_size + A_size + Z_size]
        return step(x, u, aux, z, node)

    return waste_water_ode_fn


def biohydrogen_ode(cfg):
    """Factory: returns a diffrax-shaped ODE term for the biohydrogen case study.

    Fed-batch photobiological hydrogen culture (port of sample_envs/biohydrogen.py).
    No disturbance vector (Z_size = 0); `z` is sliced empty and ignored by step.
    """
    from mu_F.unit_evaluators.explicit_fn import _make_biohydrogen_step
    X_size = int(cfg.case_study.sizes.X_SIZE)
    U_size = int(cfg.case_study.sizes.U_SIZE)
    Z_size = int(cfg.case_study.sizes.Z_SIZE)
    AUX_size = int(cfg.case_study.get('global_n_aux_args', 0))
    step = _make_biohydrogen_step(cfg)

    # unit_dynamics packs params as [design | decision_dependent | aux | uncertainty].
    # biohydrogen has decision_dependent_size = 0, so aux sits at U_size+Z_size.
    aux_lo = U_size + Z_size
    aux_hi = aux_lo + AUX_size

    def biohydrogen_ode_fn(t: float, state: jnp.ndarray, parameters: jnp.ndarray, node=None):
        x = state[..., :X_size]
        u = parameters[..., :U_size]
        z = parameters[..., U_size:U_size + Z_size]
        aux = parameters[..., aux_lo:aux_hi]
        return step(x, u, z, aux, node)

    return biohydrogen_ode_fn


# define a dictionary that contains unit wise dynamics for each of the nodes in the graph in the case study
case_studies = {'serial_mechanism_batch': {0: serial_mechanism_vc_batch_dynamics_u1, 1: serial_mechanism_vc_batch_dynamics_u2},
                'batch_reaction_network': {0: reactor_network_4, 1: reactor_network_5},
                'cstr': cstr_ode,
                'waste_water': waste_water_ode,
                'biohydrogen': biohydrogen_ode,
                }