"""Explicit (steady-state) unit functions and ODE-step factories per case study."""

from functools import partial
import logging
from jax import jit
from omegaconf import DictConfig
import jax.numpy as jnp
from jax import hessian
from jax.nn import sigmoid


# ---------------------------------------------------------------------------
# Tablet press case study
# ---------------------------------------------------------------------------

@partial(jit, static_argnums=(0,))
def bulk_density_u1(cfg, design_args, input_args, *args):
    """
    Bulk density for the conical mill (unit 1).
    Design args: API inflow, excipient inflow, blade speed.
    """
    cfg_args = cfg.model.unit_1_args.bulk_density
    return (
        cfg_args[0]
        + cfg_args[1] * design_args[1]
        + cfg_args[2] * design_args[2]
        + cfg_args[3] * design_args[1] * design_args[1] * design_args[2]
    )


@partial(jit, static_argnums=(0,))
def mean_residence_time_u1(cfg, design_args, input_args, *args):
    """
    Mean residence time for the conical mill (unit 1).
    Design args: API inflow, excipient inflow, blade speed.
    """
    cfg_args = cfg.model.unit_1_args.mean_residence_time
    cfg_d_args = cfg.case_study.KS_bounds.design_args[0][2]
    exp_term = -cfg_args[1] / (design_args[2] - cfg_d_args[0]) - cfg_args[2] / (
        design_args[0] + design_args[1]
    )
    return cfg_args[0] * (1 - jnp.exp(exp_term))


@partial(jit, static_argnums=(0,))
def unit_1_dynamics(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, *args
):
    """
    Conical mill (unit 1) dynamics: returns hold-up mass and volume, mass
    outflow rate, and API / excipient mass fractions.
    """
    design_args = design_args.squeeze()
    bulk_density = bulk_density_u1(cfg, design_args, input_args, *args) #+ design_args[-1] - design_args[-2] 
    tau_cm = mean_residence_time_u1(cfg, design_args, input_args, *args)
    hold_up = (design_args[0] + design_args[1]) * tau_cm

    return jnp.array(
        [
            hold_up,
            hold_up / bulk_density,
            design_args[0] + design_args[1],
            design_args[0] / (design_args[0] + design_args[1]),
            design_args[1] / (design_args[0] + design_args[1]),
        ]
    ).reshape(1, -1)


@partial(jit, static_argnums=(0,))
def hold_up_mass_u2(cfg, design_args, input_args, *args):
    """
    Steady-state hold-up mass for the convective blender (unit 2).
    Design args: lubricant inflow, blade speed; inputs: API / excipient flow.
    """
    cfg_args = cfg.model.unit_2_args.hold_up_mass
    mass_flow_in = input_args[0] + input_args[1] + design_args[0]
    return (
        cfg_args[0]
        + cfg_args[1] * mass_flow_in
        + cfg_args[2] * design_args[1]
        + cfg_args[3] * design_args[1] * design_args[1]
        + cfg_args[4] * mass_flow_in * design_args[1]
    )


@partial(jit, static_argnums=(0,))
def bulk_density_u2(cfg, design_args, input_args, *args):
    """
    Bulk density for the convective blender (unit 2).
    Design args: lubricant inflow, blade speed; inputs: API / excipient flow.
    """
    cfg_args = cfg.model.unit_2_args.bulk_density
    mass_flow_in = input_args[0] + input_args[1] + design_args[0]
    if not cfg.model.blender_density.include_lubricant:
        return (
            cfg_args[0]
            + cfg_args[1] * (input_args[1] / mass_flow_in) * design_args[1]
            + cfg_args[2] * (input_args[1] / mass_flow_in)
        )
    else:
        return (
            cfg_args[0]
            + cfg_args[1]
            * (input_args[1] / mass_flow_in)
            * design_args[1]
            * (design_args[0])
            / mass_flow_in
            + cfg_args[2] * (input_args[1] / mass_flow_in)
            + cfg_args[3] * design_args[0] / mass_flow_in
        )


@partial(jit, static_argnums=(0,))
def porosity_estimate_u2(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, *args
):
    """
    Porosity estimate for the convective blender (unit 2).
    Design args: lubricant inflow, blade speed; inputs: API / excipient flow.
    """
    mass_flow_in = input_args[0] + input_args[1] + design_args[0]
    # particle density calculation
    cfg_args = cfg.model.unit_2_args.particle_density
    if cfg.model.blender_density.include_lubricant:
        p_d = (
            cfg_args[0]
            + cfg_args[1]
            * design_args[1]
            * input_args[1]
            / (mass_flow_in)
            * design_args[0]
            / mass_flow_in
            + cfg_args[2] * input_args[1] / (mass_flow_in)
            + cfg_args[3] * design_args[0] / mass_flow_in
        )
    else:
        p_d = (
            cfg_args[0]
            + cfg_args[1] * design_args[1] * input_args[1] / (mass_flow_in)
            + cfg_args[2] * input_args[1] / (mass_flow_in)
        )
    # bulk density calculation
    p_bulk = bulk_density_u2(cfg, design_args, input_args, *args)
    return 1 - p_bulk / p_d, p_bulk


@partial(jit, static_argnums=(0,))
def unit_2_dynamics(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, *args
):
    """
    Convective blender (unit 2) dynamics: returns hold-up mass and volume,
    mass outflow rate, API / excipient / lubricant mass fractions, porosity.
    """
    input_args = input_args.squeeze()
    design_args= design_args.squeeze()

    mass_hold_up = hold_up_mass_u2(cfg, design_args, input_args, *args) # + design_args[-1] - design_args[-2] 
    porosity, bulk_density = porosity_estimate_u2(cfg, design_args, input_args, *args)
    mass_flow_out = input_args[0] + input_args[1] + design_args[0]

    return jnp.array(
        [
            mass_hold_up,
            mass_hold_up / bulk_density,
            mass_flow_out,
            input_args[0] / (mass_flow_out),
            input_args[1] / (mass_flow_out),
            design_args[0] / mass_flow_out,
            porosity,
        ]
    ).reshape(1, -1)


@partial(jit, static_argnums=(0,))
def main_comp_volume_unit_3(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, *args
):
    """
    Main-compression volume for the tablet press (unit 3).
    Design args: pre- and main-compression pressure; input: initial porosity.
    """
    V_pre, pre_comp_psty = args[0], args[1]
    main_comp_kawakita = cfg.model.unit_3_args.main_comp_kawakita
    numerator = V_pre * (1 - design_args[1] * main_comp_kawakita * (pre_comp_psty - 1))
    denominator = 1 + design_args[1] * main_comp_kawakita
    return numerator / denominator


@partial(jit, static_argnums=(0,))
def pre_comp_volume_unit_3(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, *args
):
    """
    Pre-compression volume for the tablet press (unit 3).
    Design args: pre- and main-compression pressure; input: initial porosity.
    """
    V_0 = cfg.model.unit_3_args.initial_volume_in_die
    pre_comp_kawakita = cfg.model.unit_3_args.pre_comp_kawakita
    numerator = V_0 * (1 - design_args[0] * pre_comp_kawakita * (input_args - 1))
    denominator = 1 + design_args[0] * pre_comp_kawakita
    return numerator / denominator


@partial(jit, static_argnums=(0,))
def porosity_update_u3(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, *args
):
    """
    Porosity update for the tablet press (unit 3).
    Design args: pre- and main-compression pressure; input: initial porosity.
    """
    V_0 = cfg.model.unit_3_args.initial_volume_in_die
    V_pre = args[0]
    return 1 - (1 - input_args) * V_0 / V_pre


def hardness_estimate_u3(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, *args
):
    """
    Hardness estimate for the tablet press (unit 3).
    Design args: pre- and main-compression pressure; input: initial porosity.
    """
    relative_density = (
        (1 - input_args) * cfg.model.unit_3_args.initial_volume_in_die / args[0]
    )
    gamma = jnp.log((1 - relative_density) / (1 - cfg.model.unit_3_args.critical_density))
    h_0 = cfg.model.unit_3_args.hardness_zero_porosity
    exp_term = relative_density - cfg.model.unit_3_args.critical_density + gamma
    return h_0 * (1 - jnp.exp(exp_term))


@partial(jit, static_argnums=(0,))
def unit_3_dynamics(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, *args
):
    """
    Tablet press (unit 3) dynamics: returns tablet hardness, pre-compression
    volume and main-compression volume.
    """
    input_args = input_args.squeeze() #+ design_args[-1] - design_args[-2]
    design_args = design_args.squeeze() 
    V_pre = pre_comp_volume_unit_3(cfg, design_args, input_args, *args)
    porosity = porosity_update_u3(cfg, design_args, input_args, *(V_pre,))
    V_main = main_comp_volume_unit_3(cfg, design_args, input_args, *(V_pre, porosity))
    H = hardness_estimate_u3(cfg, design_args, input_args, *(V_main,))

    return jnp.array([H, V_pre, V_main]).reshape(1, -1)



# ---------------------------------------------------------------------------
# Convex estimator case study
# ---------------------------------------------------------------------------

@partial(jit, static_argnums=(0,))
def sub_fn_2_eval(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
):
    """
    Block 1 evaluation for the convex estimator.
    """

    log_terms = jnp.array([jnp.log(aux[i] + 1).squeeze() for i in range(aux.shape[0])])
    coefficients = jnp.array([design_args[i] for i in range(design_args.shape[0])])
    return -jnp.dot(coefficients, log_terms).squeeze()

jax_hessian_sub_fn_2 = hessian(sub_fn_2_eval, argnums=3, has_aux=False)

@partial(jit, static_argnums=(0,))
def sub_fn_2(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
    ):
    """
    Block 2 for the convex estimator (value with convexity property).
    """
    eval = sub_fn_2_eval(
    cfg, design_args, input_args, aux.squeeze(), args)
    hess = jax_hessian_sub_fn_2(
        cfg, design_args, input_args, aux.squeeze(), args)
    cvx_prop = aux @ hess @ aux.T
    return jnp.hstack([eval.reshape(1,-1), cvx_prop.reshape(1,-1), aux.reshape(1,-1)])  


@partial(jit, static_argnums=(0,))
def sub_fn_3_eval(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
    ):
    """
    Block 3 evaluation for the convex estimator.
    """
    log_terms = jnp.array([aux[i]*jnp.log(aux[i] + 1) for i in range(aux.shape[0])])
    coefficients = jnp.array([design_args[i] for i in range(design_args.shape[0])])
    return jnp.dot(coefficients, log_terms.T).squeeze()


jax_hessian_sub_fn_3 = hessian(sub_fn_3_eval, argnums=3, has_aux=False)


@partial(jit, static_argnums=(0,))
def sub_fn_3(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
):
    """
    Block 3 for the convex estimator (value with convexity property).
    """
    eval = sub_fn_3_eval(
    cfg, design_args, input_args, aux.squeeze(), args)
    hess = jax_hessian_sub_fn_3(
        cfg, design_args, input_args, aux.squeeze(), args)
    cvx_prop = aux @ hess @ aux.T
    return jnp.hstack([eval.reshape(1,-1), cvx_prop.reshape(1,-1), aux.reshape(1,-1)])


@partial(jit, static_argnums=(0,))
def sub_fn_1(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
):
    """
    Block 1 passthrough for the convex estimator.
    """
    return jnp.hstack([design_args.reshape(1,1), aux])

@partial(jit, static_argnums=(0,))
def sub_fn_4(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
):
    """
    Block 4 for the convex estimator (linear form in the aux variables).
    """
    return jnp.hstack([jnp.dot(design_args, aux.T).reshape(1,1), aux])

@partial(jit, static_argnums=(0,))
def sub_fn_5(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
):
    """
    Block 5 for the convex estimator (quadratic form in the aux variables).
    """
    Q = jnp.diag(design_args[0,:-1])
    Q= Q.at[0,1].set(design_args[0,-1])
    Q = Q.at[1,0].set(design_args[0,-1])

    return jnp.hstack([jnp.matmul(jnp.matmul(aux, Q), aux.T).reshape(1,1), aux])
    

@partial(jit, static_argnums=(0,))
def sub_fn_6(cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args):
    """
    Block 6 for the convex estimator (sum of upstream inputs).
    """
    return jnp.hstack([jnp.sum(input_args[:-2]).reshape(1,1), aux])

# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

@partial(jit, static_argnums=(0,))
def esub_fn_2_eval(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
):
    """
    Block 1 evaluation for the estimator.
    """

    log_terms = jnp.array([jnp.log(aux[i] + 1).squeeze() for i in range(aux.shape[0])])
    coefficients = jnp.array([design_args[i] for i in range(design_args.shape[0])])
    return -jnp.dot(coefficients, log_terms).squeeze()

jax_hessian_sub_fn_2 = hessian(esub_fn_2_eval, argnums=3, has_aux=False)

@partial(jit, static_argnums=(0,))
def esub_fn_2(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
    ):
    """
    Block 2 for the estimator.
    """
    z = aux[:,:-1]
    eval = esub_fn_2_eval(
    cfg, design_args, input_args, z.squeeze(), args)
    return jnp.hstack([eval.reshape(1,-1), aux.reshape(1,-1)])  


@partial(jit, static_argnums=(0,))
def esub_fn_3_eval(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
    ):
    """
    Block 3 evaluation for the estimator.
    """
    log_terms = jnp.array([aux[i]*jnp.log(aux[i] + 1) for i in range(aux.shape[0])])
    coefficients = jnp.array([design_args[i] for i in range(design_args.shape[0])])
    return jnp.dot(coefficients, log_terms.T).squeeze()


jax_hessian_sub_fn_3 = hessian(esub_fn_3_eval, argnums=3, has_aux=False)


@partial(jit, static_argnums=(0,))
def esub_fn_3(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
):
    """
    Block 3 for the estimator.
    """
    z = aux[:,:-1]
    eval = esub_fn_3_eval(
    cfg, design_args, input_args, z.squeeze(), args)
    return jnp.hstack([eval.reshape(1,-1), aux.reshape(1,-1)])


@partial(jit, static_argnums=(0,))
def esub_fn_1(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
):
    """
    Block 1 passthrough for the estimator.
    """
    return jnp.hstack([design_args.reshape(1,1), aux])

@partial(jit, static_argnums=(0,))
def esub_fn_4(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
):
    """
    Block 4 for the estimator (linear form in the aux variables).
    """
    z = aux[:,:-1]
    return jnp.hstack([jnp.dot(design_args, z.T).reshape(1,1), aux])

@partial(jit, static_argnums=(0,))
def esub_fn_5(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
):
    """
    Block 5 for the estimator (quadratic form in the aux variables).
    """
    z = aux[:,:-1]
    Q = jnp.diag(design_args[0,:-1])
    Q= Q.at[0,1].set(design_args[0,-1])
    Q = Q.at[1,0].set(design_args[0,-1])

    return jnp.hstack([jnp.matmul(jnp.matmul(z, Q), z.T).reshape(1,1), aux])
    

@partial(jit, static_argnums=(0,))
def esub_fn_6(cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args):
    """
    Block 6 for the estimator (sum of upstream inputs).
    """
    return jnp.hstack([jnp.sum(input_args[:-2]).reshape(1,1), aux])

# ---------------------------------------------------------------------------
# Affine case study
# ---------------------------------------------------------------------------

@partial(jit, static_argnums=(0,))
def affine_case_study_1(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
):
    """
    Affine map for the illustration case study (block 1-3).
    """

    A = jnp.array(cfg.model.affine_case_study_args.A[0])
    B = jnp.array(cfg.model.affine_case_study_args.B[0])

    return (A @ design_args.T + B).squeeze() 

@partial(jit, static_argnums=(0,))
def affine_case_study_2(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
):
    """
    Affine map for the illustration case study (block 2-3).
    """

    A = jnp.array(cfg.model.affine_case_study_args.A[1])
    B = jnp.array(cfg.model.affine_case_study_args.B[1])

    return A @ design_args.T + B 



@partial(jit, static_argnums=(0,))
def affine_case_study_3(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
):
    """
    Affine map for the illustration case study (block 3-[4,5]).
    """

    A = jnp.array(cfg.model.affine_case_study_args.A[2])
    B = jnp.array(cfg.model.affine_case_study_args.B[2])

    return A @ design_args.T + B @ input_args.T


@partial(jit, static_argnums=(0,))
def affine_case_study_4(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
):
    """
    Affine map for the illustration case study (block 4).
    """

    A = jnp.array(cfg.model.affine_case_study_args.A[3])
    B = jnp.array(cfg.model.affine_case_study_args.B[3])

    return A @ design_args.T + B @ input_args


@partial(jit, static_argnums=(0,))
def affine_case_study_5(
    cfg: DictConfig, design_args: jnp.ndarray, input_args, aux, *args
):
    """
    Affine map for the illustration case study (block 5).
    """

    A = jnp.array(cfg.model.affine_case_study_args.A[4])
    B = jnp.array(cfg.model.affine_case_study_args.B[4])

    return A @ design_args.T + B @ input_args

# ---------------------------------------------------------------------------
# CSTR (pcgym, jax)
# ---------------------------------------------------------------------------

def _smooth_log(z, z0=0.02):
    f0 = jnp.log(z0)
    f1 = 1.0 / z0
    f2 = -1.0 / (z0 * z0)
    delta = z - z0
    fallback = f0 + f1 * delta + 0.5 * f2 * delta * delta
    return jnp.where(z >= z0, jnp.log(jnp.maximum(z, 1e-30)), fallback)


def _make_cstr_step(cfg: DictConfig):
    """
    Factory for the JIT-compiled CSTR step.
    cfg-dependent constants are resolved eagerly so the inner body touches
    only JAX arrays, keeping the per-step trace XLA-friendly.
    """
    import importlib

    t_lower = float(cfg.model.t_lower)
    t_upper = float(cfg.model.t_upper)
    sp_ca = jnp.asarray(list(cfg.model.sp_ca))
    penalty = str(cfg.model.get('tracking_penalty', 'smooth_log'))

    # Smoothness for the constraint-violation softmax (beta -> inf recovers the hard max).
    beta = _resolve_beta(cfg, 'softmax_beta', 50.0)

    mod = importlib.import_module("pcgym.model_classes")
    model = getattr(mod, str(cfg.model.pcgym_model_class))(int_method="jax")

    @jit
    def _step(x: jnp.ndarray, u: jnp.ndarray, node):
        x = jnp.ravel(x)
        u = jnp.ravel(u)
        dxdt = model(x, u).squeeze()

        g_lower = jnp.atleast_1d((t_lower - x[1]) / t_upper)
        g_upper = jnp.atleast_1d((x[1] - t_upper) / t_upper)

        dgdt = -_softplus_centred(jnp.array([g_lower, g_upper]), beta)
        e = jnp.take(sp_ca, node) - x[0]
        rwd = (e * e) if penalty == 'quadratic' else _smooth_log(jnp.abs(e))

        return jnp.concatenate([jnp.ravel(dxdt), jnp.ravel(dgdt), jnp.ravel(rwd)], axis=0)

    return _step


def cstr_simulator(cfg: DictConfig):
    """
    Factory for the CSTR steady-state simulator (unit_op == 'steady_state').
    The step returns the bundled tensor [F | G | R]: state derivatives,
    lower / upper temperature path constraints and the setpoint stage cost.
    """
    step = _make_cstr_step(cfg)

    def cstr_simulator_fn(cfg_unused: DictConfig, design_args, input_args, aux, uncertainties, node):
        return step(input_args, design_args, node)

    return cstr_simulator_fn


# ---------------------------------------------------------------------------
# Waste water (Bernard et al. 2001 AM2 anaerobic digestion)
# ---------------------------------------------------------------------------

def _make_waste_water_step(cfg: DictConfig):
    """
    Factory for the JIT-compiled AM2 step, emitting [F | G | R]: state
    derivatives, five feasibility margins and the -q_m / Q_M_REF stage cost.
    cfg-dependent constants are resolved eagerly at factory time.
    """
    # Smoothness for the constraint-violation softmax (beta -> inf recovers the hard max).
    beta = _resolve_beta(cfg, 'softmax_beta', 50.0)

    # Kinetics
    mu_1_max = float(cfg.model.mu_1_max)
    k_s1     = float(cfg.model.k_s1)
    mu_2_max = float(cfg.model.mu_2_max)
    k_s2     = float(cfg.model.k_s2)
    k_i2     = float(cfg.model.k_i2)
    # Physical / transfer
    kl_a     = float(cfg.model.kl_a)
    p_t      = float(cfg.model.p_t)
    k_h      = float(cfg.model.k_h)
    K_a      = float(cfg.model.K_a)
    # Yields
    k_1 = float(cfg.model.k_1)
    k_2 = float(cfg.model.k_2)
    k_3 = float(cfg.model.k_3)
    k_4 = float(cfg.model.k_4)
    k_5 = float(cfg.model.k_5)
    k_6 = float(cfg.model.k_6)
    # Constraint thresholds
    GAMMA    = float(cfg.model.gamma)
    COD_MAX  = float(cfg.model.cod_max)
    S2_MAX   = float(cfg.model.s2_max)
    PH_MIN   = float(cfg.model.ph_min)
    PH_MAX   = float(cfg.model.ph_max)
    K_B      = float(cfg.model.k_b)
    EPS_Z_S2 = float(cfg.model.eps_z_s2)
    LOG_FLOOR = 1e-30

    # Reward scaling
    Q_M_REF  = float(cfg.model.q_m_ref)


    # Day-indexed
    CODin = jnp.array([  # g/l
        9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5,        # 1-10
        9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5,        # 11-20
        9.3, 9.3, 14.7, 14.7, 14.7, 14.7, 9.3, 9.3, 9.3, 9.3,    # 21-30
        4.8, 4.8, 4.8, 4.8, 4.8, 4.8, 4.8, 4.8, 15.0, 15.0,      # 31-40
        15.0, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.8, 10.7, 10.7,  # 41-50
        10.7, 10.7, 10.7, 10.7, 10.7, 10.7, 10.7, 10.7, 10.7, 10.7,  # 51-60
        10.7, 10.7, 10.7, 10.7, 10.7, 10.7, 10.7, 10.7, 10.7, 10.7,  # 61-70
        9.3, 9.3                                                  # 71-72
    ])

    VFAin = jnp.array([  # mmol/l
        93, 93, 93, 93, 93, 93, 93, 93, 93, 93,            # 1-10
        93, 93, 93, 93, 90, 90, 90, 90, 90, 90,            # 11-20
        90, 90, 114, 114, 114, 114, 73, 73, 73, 73,        # 21-30
        38, 38, 38, 38, 38, 38, 38, 38, 113, 113,          # 31-40
        113, 73, 73, 73, 73, 73, 73, 73, 72, 72,           # 41-50
        72, 72, 72, 72, 72, 72, 72, 72, 72, 72,            # 51-60
        72, 72, 72, 72, 72, 72, 72, 72, 70, 70,            # 61-70
        70, 70                                              # 71-72
    ])

    pHin = jnp.array([
        5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13, 5.13,  # 1-10
        5.13, 5.13, 5.13, 5.13, 5.13, 5.05, 5.05, 5.05, 5.05, 5.05,  # 11-20
        5.05, 5.05, 4.40, 4.40, 4.40, 4.40, 4.40, 4.40, 4.40, 4.40,  # 21-30
        4.40, 4.40, 4.50, 4.50, 4.50, 4.50, 4.50, 4.50, 4.50, 4.50,  # 31-40
        4.40, 4.40, 4.40, 4.40, 4.40, 4.40, 4.40, 4.40, 4.40, 4.40,  # 41-50
        4.40, 4.40, 4.40, 4.40, 4.40, 4.40, 4.40, 4.40, 4.40, 4.40,  # 51-60
        4.40, 4.40, 4.40, 4.40, 4.40, 4.40, 4.40, 4.40, 4.40, 5.30,  # 61-70
        5.30, 5.30                                                   # 71-72
    ])

    Cin = 55.0  # mmol/l, influent CO2 concentration (assumed constant)



    @jit
    def _step(x: jnp.ndarray, u: jnp.ndarray, aux: jnp.ndarray, z: jnp.ndarray, node):
        x   = jnp.ravel(x)
        u   = jnp.ravel(u)
        aux = jnp.ravel(aux)
        z   = jnp.ravel(z)

        X1, X2, Z, S1, S2, C = x[0], x[1], x[2], x[3], x[4], x[5]
        log_D = u[0]
        D = jnp.exp(log_D)
        alpha = aux[0]
        #S1_in, S2_in, pH_in, C_in = z[0], z[1], z[2], z[3]

        S1_in = jnp.take(CODin, node)
        S2_in = jnp.take(VFAin, node)
        pH_in = jnp.take(pHin, node)
        C_in = Cin

        # pH
        Z_in = (K_a / (K_a + 10.0 ** (-pH_in))) * S2_in

        # Kinetics
        mu_1 = mu_1_max * S1 / (k_s1 + S1)
        mu_2 = mu_2_max * S2 / (S2 + k_s2 + S2 * S2 / k_i2)

        # CO2 in liquid 
        co2 = _softplus(C + S2 - Z, beta=10.0)
        phi = co2 + k_h * p_t + (k_6 / kl_a) * mu_2 * X2
        p_c = (phi - jnp.sqrt(LOG_FLOOR + _softplus(phi * phi - 4.0 * k_h * p_t * co2, beta=10.0))) / (2.0 * k_h)
        q_c = kl_a * (co2 - k_h * p_c)
        pH = jnp.log10(_softplus(Z - S2, beta=20.0)+LOG_FLOOR) - jnp.log10(_softplus(C - Z + S2, beta=20.0)+LOG_FLOOR) - jnp.log10(K_B)

        # Mass balances (Eqs. 20-25)
        dX1 = (mu_1 - alpha * D) * X1
        dX2 = (mu_2 - alpha * D) * X2
        dZ  = D * (Z_in - Z)
        dS1 = D * (S1_in - S1) - k_1 * mu_1 * X1
        dS2 = D * (S2_in - S2) + k_2 * mu_1 * X1 - k_3 * mu_2 * X2
        dC  = D * (C_in - C)   - q_c           + k_4 * mu_1 * X1 + k_5 * mu_2 * X2
        dxdt = jnp.array([dX1, dX2, dZ, dS1, dS2, dC])

        # Path constraints. 
        g_cod   = ((S1 + GAMMA * S2) - COD_MAX) / COD_MAX
        g_s2    = (S2 - S2_MAX) / S2_MAX
        g_ph_hi = (pH - PH_MAX) / PH_MAX
        g_ph_lo = (PH_MIN - pH) / PH_MIN
        g_zs2   = (S2 + EPS_Z_S2 - Z) / (_smooth_abs(Z) + _smooth_abs(S2))
        dgdt = -_softplus_centred(jnp.array([g_cod, g_s2, g_ph_hi, g_ph_lo, g_zs2]), beta)

        # Stage cost
        q_m = k_6 * mu_2 * X2
        rwd = -q_m / Q_M_REF

        return jnp.concatenate([jnp.ravel(dxdt), jnp.ravel(dgdt), jnp.atleast_1d(rwd)], axis=0)

    return _step


def waste_water_simulator(cfg: DictConfig):
    """
    Factory for the waste-water steady-state simulator
    (unit_op == 'steady_state').
    """
    step = _make_waste_water_step(cfg)

    def waste_water_simulator_fn(cfg_unused: DictConfig, design_args, input_args, aux, uncertainties, node):
        return step(input_args, design_args, aux, uncertainties, node)

    return waste_water_simulator_fn


# ---------------------------------------------------------------------------
# Softplus-based smooth min/max helpers (smooth-dynamics models)
# ---------------------------------------------------------------------------
# beta -> inf recovers the hard operators; smoothing the kinks lets Newton SQP
# differentiate through the dynamics.

def _smooth_max(x, y, beta):
    """Softplus-smooth max(x, y).  beta -> inf recovers jnp.maximum."""
    return y + jnp.logaddexp(beta * (x - y), 0.0) / beta

def _smooth_min(x, y, beta):
    """Softplus-smooth min(x, y).  beta -> inf recovers jnp.minimum."""
    return -_smooth_max(-x, -y, beta)

def _softplus(x, beta):
    """Softplus function with smoothness parameter beta.  beta -> inf recovers relu."""
    return jnp.logaddexp(x * beta, 0.0) / beta


def _softplus_centred(x, beta):
    """
    Softplus shifted through the origin so a satisfied constraint contributes 0;
    keeps an integrated path penalty feasible at the bound (PC=0 -> g=0).
    """
    return _softplus(x, beta) - jnp.log(2.0) / beta


_MONO_METHODS = {'direct', 'single_shooting', 'multiple_shooting'}

def _resolve_beta(cfg, key, default):
    """
    Smoothing beta for the active method: a {decomposition, monolithic} mapping
    picks per pass, a scalar applies to both (decomposition runs near-hard).
    """
    value = cfg.model.get(key, default)
    if isinstance(value, (dict, DictConfig)):
        bucket = 'monolithic' if cfg.get('method', '') in _MONO_METHODS else 'decomposition'
        return float(value.get(bucket, default))
    return float(value)

def _smooth_abs(x, eps=1e-6):
    """Smooth |x| = sqrt(x^2 + eps^2); eps -> 0 recovers jnp.abs."""
    return jnp.sqrt(x * x + eps * eps)


# ---------------------------------------------------------------------------
# Hydrogen export (port of sample_envs/hydrogen3.py)
# ---------------------------------------------------------------------------

def _make_hydrogen_export_step(cfg: DictConfig):
    """
    Factory for the JIT-compiled hydrogen-export step (3-train ammonia vector).
    The renewable-energy disturbance z is consumed directly; the step emits
    [F | G | R]: next state, storage / energy-balance constraints, stage cost.
    """
    # Capacities / counts
    n_turbines              = float(cfg.model.n_turbines)
    n_trains_conversion     = float(cfg.model.n_trains_conversion)
    train_throughput_cap    = float(cfg.model.train_throughput_capacity)
    # Efficiencies / penalties
    vector_molar_efficiency = float(cfg.model.vector_molar_efficiency)
    electrolyser_efficiency = float(cfg.model.electrolyser_efficiency)
    fuelcell_efficiency     = float(cfg.model.fuelcell_efficiency)
    fixed_energy_penalty    = float(cfg.model.fixed_energy_penalty)
    variable_energy_penalty = float(cfg.model.variable_energy_penalty)
    vector_calorific_value  = float(cfg.model.vector_calorific_value)
    # Storage limits
    lower_storage_limit     = float(cfg.model.lower_storage_limit)
    upper_storage_limit     = float(cfg.model.upper_storage_limit)
    storage_lo = lower_storage_limit * upper_storage_limit
    storage_hi = upper_storage_limit

    # ramping limit
    lower_ramp_limit        = float(cfg.model.lower_ramp_limit)
    upper_ramp_limit        = float(cfg.model.upper_ramp_limit)
    minimum_train_throughput = float(cfg.model.minimum_train_throughput)

    # Smoothness for the fuel-cell / electrolyser split (softplus β).
    split_beta = float(cfg.model.get('smooth_beta_power_split', 20.0))

    @jit
    def _step(x: jnp.ndarray, u: jnp.ndarray, z: jnp.ndarray, node):
        x = jnp.ravel(x)
        u = jnp.ravel(u)
        z = jnp.ravel(z)

        _storage, _throughput, _n_active, = x
        n_active, delta_throughput, power_action = u
        # Weather realisation (0..11.88) sampled per scenario, passed via p.
        renewable_energy = jnp.ravel(z)[0]

        fuel_cell_energy    = _softplus(-power_action, beta=split_beta)
        electrolysis_energy = _softplus( power_action, beta=split_beta)

        # New absolute throughput = predecessor + delta. Same calorific units.
        throughput = _throughput + delta_throughput

        vector_production_energy = (
            fixed_energy_penalty * n_active * variable_energy_penalty * train_throughput_cap +
            (1 - fixed_energy_penalty) * throughput * (variable_energy_penalty / vector_calorific_value)
        )

        storage = (
            _storage - fuel_cell_energy / fuelcell_efficiency +
            electrolysis_energy * electrolyser_efficiency - throughput / vector_molar_efficiency
        )

        # all cons >= 0 is feasible; fc - el = -power_action via the softplus identity
        energy_balance = (renewable_energy * n_turbines - power_action - vector_production_energy) / (11.88 * n_turbines)
        lower_storage = (storage - storage_lo) / (storage_hi)
        upper_storage = (storage_hi - storage)  / (storage_hi)
        ramp_lo = (delta_throughput / vector_calorific_value + _n_active * train_throughput_cap * lower_ramp_limit) / train_throughput_cap
        ramp_hi = ((n_trains_conversion + 1 - _n_active) * train_throughput_cap * upper_ramp_limit
                   - delta_throughput / vector_calorific_value) / train_throughput_cap
        throughput_upper = (n_active * train_throughput_cap - throughput / vector_calorific_value) / train_throughput_cap
        throughput_lower = (throughput / vector_calorific_value - n_active * train_throughput_cap * minimum_train_throughput) / train_throughput_cap

        reward = -(throughput)

        outputs     = jnp.array([storage, throughput, n_active])      # F
        constraints = jnp.array([energy_balance, lower_storage, upper_storage,
                                 ramp_lo, ramp_hi,
                                 throughput_upper, throughput_lower])       # G
        cost        = jnp.atleast_1d(reward)                                # R

        return jnp.concatenate([outputs, constraints, cost], axis=0)

    return _step


def hydrogen_export_simulator(cfg: DictConfig):
    """
    Factory for the hydrogen-export steady-state simulator.
    """
    step = _make_hydrogen_export_step(cfg)

    def hydrogen_export_simulator_fn(cfg_unused: DictConfig, design_args, input_args, aux, uncertainties, node):
        return step(input_args, design_args, uncertainties, node)

    return hydrogen_export_simulator_fn


# ---------------------------------------------------------------------------
# Biohydrogen (port of sample_envs/biohydrogen.py, fed-batch H2 culture)
# ---------------------------------------------------------------------------

def _make_biohydrogen_step(cfg: DictConfig):
    """
    Factory for the JIT-compiled biohydrogen step (fed-batch H2 culture).
    Controls feed nitrate concentration and log feed flow; the per-trajectory
    aux caps the feed rate. Emits [F | G | R | Phi]; integration is in hours.
    """
    # Smoothness for the constraint-violation softmax (beta -> inf recovers the hard max).
    beta = _resolve_beta(cfg, 'softmax_beta', 50.0)

    mu_max = float(cfg.model.mu_max)
    k_q    = float(cfg.model.k_q)
    K_c    = float(cfg.model.k_c)
    mu_d   = float(cfg.model.mu_d)
    K_N    = float(cfg.model.k_n)
    Y_NX   = float(cfg.model.y_nx)
    Y_qX   = float(cfg.model.y_qx)
    Y_OX   = float(cfg.model.y_ox)
    Y_d    = float(cfg.model.y_d)
    Y_CX   = float(cfg.model.y_cx)
    F_max  = float(cfg.model.f_max)
    O_Fed  = float(cfg.model.o_fed)
    C_Fed  = float(cfg.model.c_fed)
    Y_HX   = float(cfg.model.y_hx)

    N_MAX    = float(cfg.model.n_max)
    O_MAX    = float(cfg.model.o_max)
    N_SWITCH = float(cfg.model.n_switch)
    N_SWITCH_BETA = float(cfg.model.get('n_switch_beta', 1.0))
    O_GATE_BETA = float(cfg.model.get('o_gate_beta', 2.0))
    TF       = float(cfg.model.integration.tf)
    REWARD_SCALE = float(cfg.model.get('reward_scale', 1.0))

    @jit
    def _step(x: jnp.ndarray, u: jnp.ndarray, z: jnp.ndarray, aux: jnp.ndarray, node):
        x = jnp.ravel(x)
        u = jnp.ravel(u)
        aux = jnp.ravel(aux)
        # z is unused — biohydrogen has Z_SIZE = 0.

        X, C, N, q, O, H, F = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        N_Fed, log_F_in = u[0], u[1]
        F_in = jnp.exp(log_F_in)                  # decision is in log space
        max_fr_per_node = aux[0]

        # Guards: k_q/q diverges as q→0, and C/(K_c+C)=0/0 at C=0 with K_c=0.
        q_safe = jnp.maximum(q, 1e-8)
        t1 = mu_max * (1.0 - k_q / q_safe) * (C / (K_c + C + 1e-8))
        t2 = N / (K_N + N)
        t3 = sigmoid(O * O_GATE_BETA)          # ~0 at O≈0, ~1 at O>0
        # numerically-stable (1 - σ(2·O)) avoiding float64 cancellation
        gate = sigmoid(-O * O_GATE_BETA)
        f_N = sigmoid((N_SWITCH - N) * N_SWITCH_BETA)    # ~1 at N<switch, ~0 otherwise

        # state derivatives; F_in is the decision directly (L/h)
        dX = X * t1 - mu_d * X ** 2
        dC = -Y_CX * X * t1 + F_in * C_Fed
        dN = -Y_NX * X * t2 * mu_max + F_in * N_Fed
        dq = Y_qX * t2 * mu_max - t1 * q
        dO = Y_OX * X * t2 - Y_d * X ** 2 * t3 + O_Fed * F_in
        dH = Y_HX * X * gate * f_N
        dF = F_in
        dxdt = jnp.array([dX, dC, dN, dq, dO, dH, dF])

        # Path constraints — framework convention (negative = violated, 0 feasible).
        f_in_cap = max_fr_per_node * F_max / TF
        g_N = (N - N_MAX) / N_MAX
        g_O = (O - O_MAX) / O_MAX
        g_F = (F - F_max) / F_max
        g_rate = (F_in - f_in_cap) / (F_max / TF)

        dgdt = -_softplus_centred(jnp.array([g_N, g_O, g_F, g_rate]), beta)

        # Cost — normalised by REWARD_SCALE to keep the monolithic objective O(1)
        rwd = -Y_HX * X * gate * f_N / REWARD_SCALE
        phi = -H / TF / REWARD_SCALE

        return jnp.concatenate([
            jnp.ravel(dxdt), jnp.ravel(dgdt),
            jnp.atleast_1d(rwd), jnp.atleast_1d(phi),
        ], axis=0)

    return _step


def biohydrogen_simulator(cfg: DictConfig):
    """
    Factory for biohydrogen used when unit_op == 'steady_state'.
    The real workflow is dynamic and routes through ode.biohydrogen_ode; this
    entry exists for symmetry with the other dynamic case studies.
    """
    step = _make_biohydrogen_step(cfg)

    def biohydrogen_simulator_fn(cfg_unused: DictConfig, design_args, input_args, aux, uncertainties, node):
        return step(input_args, design_args, uncertainties, aux, node)

    return biohydrogen_simulator_fn


case_studies = {'tablet_press': {0: unit_1_dynamics, 1: unit_2_dynamics, 2: unit_3_dynamics},
                'convex_estimator': {0: sub_fn_1, 1: sub_fn_2, 2: sub_fn_3, 3: sub_fn_4, 4: sub_fn_5, 5: sub_fn_6},
                'convex_underestimator': {0: sub_fn_1, 1: sub_fn_2, 2: sub_fn_3, 3: sub_fn_4, 4: sub_fn_5, 5: sub_fn_6},
                'estimator': {0: esub_fn_1, 1: esub_fn_2, 2: esub_fn_3, 3: esub_fn_4, 4: esub_fn_5, 5: esub_fn_6},
                'affine_study': {0: affine_case_study_1, 1: affine_case_study_2, 2: affine_case_study_3, 3: affine_case_study_4, 4: affine_case_study_5},
                'cstr': cstr_simulator,
                'waste_water': waste_water_simulator,
                'hydrogen_export': hydrogen_export_simulator,
                'biohydrogen': biohydrogen_simulator}