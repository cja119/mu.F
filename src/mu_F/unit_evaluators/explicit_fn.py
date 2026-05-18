
from functools import partial
import logging 
from jax import jit
from omegaconf import DictConfig
import jax.numpy as jnp
from jax import hessian
from jax.nn import sigmoid


# --- tablet press case study --- # 

@partial(jit, static_argnums=(0,))
def bulk_density_u1(cfg, design_args, input_args, *args):
    """bulk density function for conical mill
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - mass inflow of api, mass inflow of excipient, blade speed
    input args - None
    args - None

    Output:
        critical quality attributes (CQAs) - bulk density
        process constraints - None
        outputs - None

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
    """mean residence time function for conical mill
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - mass inflow of api, mass inflow of excipient, blade speed
    input args - None
    args - None

    Output:
        critical quality attributes (CQAs) - mean residence time
        process constraints - None
        outputs - None

    """
    cfg_args = cfg.model.unit_1_args.mean_residence_time
    cfg_d_args = cfg.case_study.KS_bounds.design_args[0][2]
    exp_term = -cfg_args[1] / (design_args[2] - cfg_d_args[0]) - cfg_args[2] / (
        design_args[0] + design_args[1]
    )
    return cfg_args[0] * (1 - jnp.exp(exp_term))


@partial(jit, static_argnums=(0,))
def unit_1_dynamics(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, *args: None
):
    """unit 1 function for conical mill
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - mass inflow of api, mass inflow of excipient, blade speed
    input args - None
    args - None

    Output:
        outputs - hold up mass, hold_up volume, mass outflow rate, mass fraction of api, mass fraction of excipient

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
    """steady state hold up mass function for convective blender
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - mass inflow of lubricant, blade speed
    input args - mass flow of api, mass flow of excipient
    args - None

    Output:
        outputs - steady state hold up mass

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
    """bulk density function for convective blender
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - mass inflow of lubricant, blade speed
    input args - mass flow of api, mass flow of excipient
    args - None

    Output:
        outputs - bulk density

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
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, *args: None
):
    """porosity estimate function for convective blender
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - mass inflow of lubricant, blade speed
    input args - mass flow of api, mass flow of excipient
    args - None

    Output:
        outputs - porosity estimate

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
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, *args: None
):
    """unit 2 function for convective blender
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - mass inflow of lubricant, blade speed
    input args - mass flow of api, mass flow of excipient
    args - None

    Output:
        outputs - hold up mass, hold_up volume, mass outflow rate, mass fraction of api, mass fraction of excipient, mass fraction of lubricant, porosity

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
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, *args: None
):
    """volume function for tablet press
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - Pre-compression pressure, main compression pressure
    input args - initial porosity
    args - pre-compression volume, pre-compression porosity

    Output:
        outputs - main-compression volume

    """
    V_pre, pre_comp_psty = args[0], args[1]
    main_comp_kawakita = cfg.model.unit_3_args.main_comp_kawakita
    numerator = V_pre * (1 - design_args[1] * main_comp_kawakita * (pre_comp_psty - 1))
    denominator = 1 + design_args[1] * main_comp_kawakita
    return numerator / denominator


@partial(jit, static_argnums=(0,))
def pre_comp_volume_unit_3(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, *args: None
):
    """pre comp volume function for tablet press
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - Pre-compression pressure, main compression pressure
    input args - initial porosity
    args - None

    Output:
        outputs - pre-compression volume

    """
    V_0 = cfg.model.unit_3_args.initial_volume_in_die
    pre_comp_kawakita = cfg.model.unit_3_args.pre_comp_kawakita
    numerator = V_0 * (1 - design_args[0] * pre_comp_kawakita * (input_args - 1))
    denominator = 1 + design_args[0] * pre_comp_kawakita
    return numerator / denominator


@partial(jit, static_argnums=(0,))
def porosity_update_u3(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, *args: None
):
    """porosity update function for tablet press
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - Pre-compression pressure, main compression pressure
    input args - initial porosity
    args - pre-compression volume

    Output:
        outputs - porosity

    """
    V_0 = cfg.model.unit_3_args.initial_volume_in_die
    V_pre = args[0]
    return 1 - (1 - input_args) * V_0 / V_pre


def hardness_estimate_u3(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, *args: None
):
    """hardness estimate function for tablet press
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - Pre-compression pressure, main compression pressure
    input args - initial porosity
    args - main-compression volume

    Output:
        outputs - hardness

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
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, *args: None
):
    """unit 3 function for tablet press
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - Pre-compression pressure, main compression pressure
    input args - porosity
    args - None

    Output:
        outputs - tablet hardness, pre-compression volume, main compression volume

    """
    input_args = input_args.squeeze() #+ design_args[-1] - design_args[-2]
    design_args = design_args.squeeze() 
    V_pre = pre_comp_volume_unit_3(cfg, design_args, input_args, *args)
    porosity = porosity_update_u3(cfg, design_args, input_args, *(V_pre,))
    V_main = main_comp_volume_unit_3(cfg, design_args, input_args, *(V_pre, porosity))
    H = hardness_estimate_u3(cfg, design_args, input_args, *(V_main,))

    return jnp.array([H, V_pre, V_main]).reshape(1, -1)



# --- convex estimator case study --- #

@partial(jit, static_argnums=(0,))
def sub_fn_2_eval(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
):
    """sub function 1 for convex estimator
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - None
    input args - None
    args - None

    Output:
        outputs - Block 1 output

    """

    log_terms = jnp.array([jnp.log(aux[i] + 1).squeeze() for i in range(aux.shape[0])])
    coefficients = jnp.array([design_args[i] for i in range(design_args.shape[0])])
    return -jnp.dot(coefficients, log_terms).squeeze()

jax_hessian_sub_fn_2 = hessian(sub_fn_2_eval, argnums=3, has_aux=False)

@partial(jit, static_argnums=(0,))
def sub_fn_2(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
    ):
    """
    sub fn 2 for convex estimator
    """
    eval = sub_fn_2_eval(
    cfg, design_args, input_args, aux.squeeze(), args)
    hess = jax_hessian_sub_fn_2(
        cfg, design_args, input_args, aux.squeeze(), args)
    cvx_prop = aux @ hess @ aux.T
    return jnp.hstack([eval.reshape(1,-1), cvx_prop.reshape(1,-1), aux.reshape(1,-1)])  


@partial(jit, static_argnums=(0,))
def sub_fn_3_eval(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
    ):
    """sub function 3 for convex estimator
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - None
    input args - None
    args - None

    Output:
        outputs - Block 2 output

    """
    log_terms = jnp.array([aux[i]*jnp.log(aux[i] + 1) for i in range(aux.shape[0])])
    coefficients = jnp.array([design_args[i] for i in range(design_args.shape[0])])
    return jnp.dot(coefficients, log_terms.T).squeeze()


jax_hessian_sub_fn_3 = hessian(sub_fn_3_eval, argnums=3, has_aux=False)


@partial(jit, static_argnums=(0,))
def sub_fn_3(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
):
    """sub function 3 for convex estimator
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - None
    input args - None
    args - None

    Output:
        outputs - Block 3 output

    """
    eval = sub_fn_3_eval(
    cfg, design_args, input_args, aux.squeeze(), args)
    hess = jax_hessian_sub_fn_3(
        cfg, design_args, input_args, aux.squeeze(), args)
    cvx_prop = aux @ hess @ aux.T
    return jnp.hstack([eval.reshape(1,-1), cvx_prop.reshape(1,-1), aux.reshape(1,-1)])


@partial(jit, static_argnums=(0,))
def sub_fn_1(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
):
    """sub function 3 for convex estimator
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - None
    input args - None
    args - None

    Output:
        outputs - Block 3 output

    """
    return jnp.hstack([design_args.reshape(1,1), aux])

@partial(jit, static_argnums=(0,))
def sub_fn_4(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
):
    """sub function 4 for convex estimator
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - None
    input args - None
    args - None

    Output:
        outputs - Block 4 output

    """
    return jnp.hstack([jnp.dot(design_args, aux.T).reshape(1,1), aux])

@partial(jit, static_argnums=(0,))
def sub_fn_5(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
):
    """sub function 5 for convex estimator
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - None
    input args - None
    args - None

    Output:
        outputs - Block 5 output

    """
    Q = jnp.diag(design_args[0,:-1])
    Q= Q.at[0,1].set(design_args[0,-1])
    Q = Q.at[1,0].set(design_args[0,-1])

    return jnp.hstack([jnp.matmul(jnp.matmul(aux, Q), aux.T).reshape(1,1), aux])
    

@partial(jit, static_argnums=(0,))
def sub_fn_6(cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None):
    """sub function 6 for convex estimator
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - None
    input args - None
    args - None

    Output:
        outputs - Block 6 output

    """
    return jnp.hstack([jnp.sum(input_args[:-2]).reshape(1,1), aux])

# ----------- estimator --------------  #

@partial(jit, static_argnums=(0,))
def esub_fn_2_eval(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
):
    """sub function 1 for convex estimator
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - None
    input args - None
    args - None

    Output:
        outputs - Block 1 output

    """

    log_terms = jnp.array([jnp.log(aux[i] + 1).squeeze() for i in range(aux.shape[0])])
    coefficients = jnp.array([design_args[i] for i in range(design_args.shape[0])])
    return -jnp.dot(coefficients, log_terms).squeeze()

jax_hessian_sub_fn_2 = hessian(esub_fn_2_eval, argnums=3, has_aux=False)

@partial(jit, static_argnums=(0,))
def esub_fn_2(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
    ):
    """
    sub fn 2 for convex estimator
    """
    z = aux[:,:-1]
    eval = esub_fn_2_eval(
    cfg, design_args, input_args, z.squeeze(), args)
    return jnp.hstack([eval.reshape(1,-1), aux.reshape(1,-1)])  


@partial(jit, static_argnums=(0,))
def esub_fn_3_eval(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
    ):
    """sub function 3 for convex estimator
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - None
    input args - None
    args - None

    Output:
        outputs - Block 2 output

    """
    log_terms = jnp.array([aux[i]*jnp.log(aux[i] + 1) for i in range(aux.shape[0])])
    coefficients = jnp.array([design_args[i] for i in range(design_args.shape[0])])
    return jnp.dot(coefficients, log_terms.T).squeeze()


jax_hessian_sub_fn_3 = hessian(esub_fn_3_eval, argnums=3, has_aux=False)


@partial(jit, static_argnums=(0,))
def esub_fn_3(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
):
    """sub function 3 for convex estimator
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - Nonee
    input args - None
    args - None

    Output:
        outputs - Block 3 output

    """
    z = aux[:,:-1]
    eval = esub_fn_3_eval(
    cfg, design_args, input_args, z.squeeze(), args)
    return jnp.hstack([eval.reshape(1,-1), aux.reshape(1,-1)])


@partial(jit, static_argnums=(0,))
def esub_fn_1(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
):
    """sub function 3 for convex estimator
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - None
    input args - None
    args - None

    Output:
        outputs - Block 3 output

    """
    return jnp.hstack([design_args.reshape(1,1), aux])

@partial(jit, static_argnums=(0,))
def esub_fn_4(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
):
    """sub function 4 for convex estimator
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - None
    input args - None
    args - None

    Output:
        outputs - Block 4 output

    """
    z = aux[:,:-1]
    return jnp.hstack([jnp.dot(design_args, z.T).reshape(1,1), aux])

@partial(jit, static_argnums=(0,))
def esub_fn_5(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
):
    """sub function 5 for convex estimator
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - None
    input args - None
    args - None

    Output:
        outputs - Block 5 output

    """
    z = aux[:,:-1]
    Q = jnp.diag(design_args[0,:-1])
    Q= Q.at[0,1].set(design_args[0,-1])
    Q = Q.at[1,0].set(design_args[0,-1])

    return jnp.hstack([jnp.matmul(jnp.matmul(z, Q), z.T).reshape(1,1), aux])
    

@partial(jit, static_argnums=(0,))
def esub_fn_6(cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None):
    """sub function 6 for convex estimator
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - None
    input args - None
    args - None

    Output:
        outputs - Block 6 output

    """
    return jnp.hstack([jnp.sum(input_args[:-2]).reshape(1,1), aux])

# ------------------------------------- #
# --------- affine case study --------- #
# ------------------------------------- #

@partial(jit, static_argnums=(0,))
def affine_case_study_1(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
):
    """affine case study for illustration
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - None
    input args - None
    args - None

    Output:
        outputs -  1-3

    """

    A = jnp.array(cfg.model.affine_case_study_args.A[0])
    B = jnp.array(cfg.model.affine_case_study_args.B[0])

    return (A @ design_args.T + B).squeeze() 

@partial(jit, static_argnums=(0,))
def affine_case_study_2(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
):
    """affine case study for illustration
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - None
    input args - None
    args - None

    Output:
        outputs - 2-3

    """

    A = jnp.array(cfg.model.affine_case_study_args.A[1])
    B = jnp.array(cfg.model.affine_case_study_args.B[1])

    return A @ design_args.T + B 



@partial(jit, static_argnums=(0,))
def affine_case_study_3(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
):
    """affine case study for illustration
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - None
    input args - None
    args - None

    Output:
        outputs - 3-[4,5]

    """

    A = jnp.array(cfg.model.affine_case_study_args.A[2])
    B = jnp.array(cfg.model.affine_case_study_args.B[2])

    return A @ design_args.T + B @ input_args.T


@partial(jit, static_argnums=(0,))
def affine_case_study_4(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
):
    """affine case study for illustration
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - None
    input args - None
    args - None

    Output:
        outputs

    """

    A = jnp.array(cfg.model.affine_case_study_args.A[3])
    B = jnp.array(cfg.model.affine_case_study_args.B[3])

    return A @ design_args.T + B @ input_args


@partial(jit, static_argnums=(0,))
def affine_case_study_5(
    cfg: DictConfig, design_args: jnp.ndarray, input_args: None, aux:None, *args: None
):
    """affine case study for illustration
    Args:
        cfg: hydra config
        design_args: design arguments
        input_args: input arguments
        *args: additional arguments

    design args - None
    input args - None
    args - None

    Output:
        outputs

    """

    A = jnp.array(cfg.model.affine_case_study_args.A[4])
    B = jnp.array(cfg.model.affine_case_study_args.B[4])

    return A @ design_args.T + B @ input_args

# -------------------------------------------------------------------------------- #
# ----------------------------- CSTR (pcgym, jax) -------------------------------- #
# -------------------------------------------------------------------------------- #

def _smooth_log(z, z0=0.02):
    f0 = jnp.log(z0)
    f1 = 1.0 / z0
    f2 = -1.0 / (z0 * z0)
    delta = z - z0
    fallback = f0 + f1 * delta + 0.5 * f2 * delta * delta
    return jnp.where(z >= z0, jnp.log(jnp.maximum(z, 1e-30)), fallback)


def _make_cstr_step(cfg: DictConfig):
    """Factory: build a JIT-compiled CSTR step function.

    All cfg-dependent values (constraint thresholds, setpoint trajectory,
    pcgym model class) are resolved eagerly into Python scalars / jnp arrays /
    Python objects at factory time, so the inner JIT'd body only ever touches
    JAX arrays. This avoids OmegaConf attribute-access overhead inside the
    per-step trace and makes the step XLA-friendly.
    """
    import importlib

    t_lower = float(cfg.model.t_lower)
    t_upper = float(cfg.model.t_upper)
    sp_ca = jnp.asarray(list(cfg.model.sp_ca))

    mod = importlib.import_module("pcgym.model_classes")
    model = getattr(mod, str(cfg.model.pcgym_model_class))(int_method="jax")

    @jit
    def _step(x: jnp.ndarray, u: jnp.ndarray, node):
        x = jnp.ravel(x)
        u = jnp.ravel(u)
        dxdt = model(x, u).squeeze()

        g_lower = -jnp.minimum(0.0, (x[1] - t_lower) / t_upper)
        g_upper = -jnp.minimum(0.0, (t_upper - x[1]) / t_upper)
        dgdt = jnp.concatenate([jnp.atleast_1d(g_lower), jnp.atleast_1d(g_upper)], axis=0)

        rwd = _smooth_log(jnp.abs(jnp.take(sp_ca, node) - x[0]))

        return jnp.concatenate([jnp.ravel(dxdt), jnp.ravel(dgdt), jnp.ravel(rwd)], axis=0)

    return _step


def cstr_simulator(cfg: DictConfig):
    """Factory for the CSTR steady-state-style simulator (used when
    unit_op == 'steady_state'). Returns a function with the standard
    case-study signature (cfg, design_args, input_args, aux, uncertainties, node).

    The inner step is JIT-compiled with cfg-resolved constants closed over.

    Bundled tensor returned by the step is [F | G | R] (no terminal cost):
        F = dxdt       (state derivatives, F_SIZE = X_SIZE = 2)
        G = path cons  (lower / upper temperature, G_SIZE = 2)
        R = stage cost (smooth log distance to setpoint, L_SIZE = 1)
    """
    step = _make_cstr_step(cfg)

    def cstr_simulator_fn(cfg_unused: DictConfig, design_args, input_args, aux, uncertainties, node):
        return step(input_args, design_args, node)

    return cstr_simulator_fn


# -------------------------------------------------------------------------------- #
# -------- Waste water (Bernard et al. 2001 AM2 anaerobic digestion) ------------- #
# -------------------------------------------------------------------------------- #

def _make_waste_water_step(cfg: DictConfig):
    """Factory: build a JIT-compiled AM2 step function.

    Returns a function with signature `_step(x, u, z, node)` that emits the
    bundled tensor [F | G | R] of shape (X_SIZE + G_SIZE + L_SIZE,) = 12:
        F = dxdt        (state derivatives, F_SIZE = X_SIZE = 6)
        G = path cons   (5 feasibility margins, positive = feasible)
        R = stage cost  (-q_m / Q_M_REF, lower is better; du² penalty dropped)

    All cfg-dependent constants are resolved eagerly at factory time so the
    inner JIT'd body only touches plain JAX arrays.
    """
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
    # Reward scaling
    Q_M_REF  = float(cfg.model.q_m_ref)
    # Auxiliary (biomass-in-liquid fraction)
    alpha    = float(cfg.model.alpha)

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
    def _step(x: jnp.ndarray, u: jnp.ndarray, z: jnp.ndarray, node):
        x = jnp.ravel(x)
        u = jnp.ravel(u)
        z = jnp.ravel(z)

        X1, X2, Z, S1, S2, C = x[0], x[1], x[2], x[3], x[4], x[5]
        D = u[0]
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
        p_c = (phi - jnp.sqrt(phi * phi - 4.0 * k_h * p_t * co2)) / (2.0 * k_h)
        q_c = kl_a * (co2 - k_h * p_c)
        pH = jnp.log10(Z - S2 + EPS_Z_S2) - jnp.log10(K_B * (C - Z + S2) + EPS_Z_S2)

        # Mass balances (Eqs. 20-25)
        dX1 = (mu_1 - alpha * D) * X1
        dX2 = (mu_2 - alpha * D) * X2
        dZ  = D * (Z_in - Z)
        dS1 = D * (S1_in - S1) - k_1 * mu_1 * X1
        dS2 = D * (S2_in - S2) + k_2 * mu_1 * X1 - k_3 * mu_2 * X2
        dC  = D * (C_in - C)   - q_c           + k_4 * mu_1 * X1 + k_5 * mu_2 * X2
        dxdt = jnp.array([dX1, dX2, dZ, dS1, dS2, dC])

        # Path constraints. 
        g_cod   = (COD_MAX - (S1 + GAMMA * S2)) / COD_MAX
        g_s2    = (S2_MAX - S2) / S2_MAX
        g_ph_hi = (PH_MAX - pH) / PH_MAX
        g_ph_lo = (pH - PH_MIN) / PH_MIN
        g_zs2   = (Z - S2 - EPS_Z_S2) / (jnp.abs(Z)+ jnp.abs(S2))
        dgdt = -jnp.maximum(jnp.array([g_cod, g_s2, g_ph_hi, g_ph_lo, g_zs2]), 0.0)

        # Stage cost
        q_m = k_6 * mu_2 * X2
        rwd = -q_m / Q_M_REF

        return jnp.concatenate([jnp.ravel(dxdt), jnp.ravel(dgdt), jnp.atleast_1d(rwd)], axis=0)

    return _step


def waste_water_simulator(cfg: DictConfig):
    """Factory for the waste-water steady-state-style simulator (used when
    unit_op == 'steady_state'). Returns a function with the standard
    case-study signature (cfg, design, input, aux, uncertainties, node)."""
    step = _make_waste_water_step(cfg)

    def waste_water_simulator_fn(cfg_unused: DictConfig, design_args, input_args, aux, uncertainties, node):
        return step(input_args, design_args, uncertainties, node)

    return waste_water_simulator_fn


# -------------------------------------------------------------------------------- #
# ----- Softplus-based smooth min/max helpers (used by smooth-dynamics models) --- #
# `_smooth_max(x, y, β) → max(x, y)` as `β → ∞`.  Inside the rounding zone
# `|x − y| ≲ 1/β` it's a smooth (C^∞) transition; outside it's effectively
# the hard max.  Composed for `_smooth_clip`.  Replaces kinks (`jnp.maximum(·, 0)`
# and `jnp.clip(·, lo, hi)`) that fundamentally break smooth-Newton SQP.
# -------------------------------------------------------------------------------- #

def _smooth_max(x, y, beta):
    """Softplus-smooth max(x, y).  beta -> inf recovers jnp.maximum."""
    return y + jnp.logaddexp(beta * (x - y), 0.0) / beta

def _smooth_min(x, y, beta):
    """Softplus-smooth min(x, y).  beta -> inf recovers jnp.minimum."""
    return -_smooth_max(-x, -y, beta)

def _softplus(x, beta):
    """Softplus function with smoothness parameter beta.  beta -> inf recovers relu."""
    return jnp.logaddexp(x * beta, 0.0) / beta


# -------------------------------------------------------------------------------- #
# ----------------- Hydrogen export (port of sample_envs/hydrogen3.py) ----------- #
# -------------------------------------------------------------------------------- #

def _make_hydrogen_export_step(cfg: DictConfig):
    """Factory: build a JIT-compiled hydrogen-export steady-state step.

    Port of sample_envs/hydrogen3.py: 3-train ammonia-vector hydrogen export.
    Renewable-energy disturbance is consumed *directly* (z = energy value),
    matching the legacy markov_process.yaml `parameters_*` semantics — the
    discrete distribution lives in `cfg.case_study.parameters_samples`, not
    in an inverse-CDF table.

    Returns a function `_step(x, u, z, node)` emitting the bundled tensor
    [F | G | R] of shape (X_SIZE + G_SIZE + L_SIZE,) = (2 + 3 + 1,) = 6:
        F = [hydrogen_storage, train_throughput]    (next state, F_SIZE=2)
        G = [lower_h2_storage, upper_h2_storage,
             energy_balance]                         (positive = feasible, G_SIZE=3)
        R = stage cost  (lower = better; 3-train negative throughput +
                         storage decay - lambda·||ramp||²)

    Note: hydrogen3.py treats the 3 conversion trains identically (same state
    component x[1], same control component u[0]), so the dynamics collapse to
    one shared throughput multiplied by 3.
    """
    # Capacities / counts
    n_turbines              = float(cfg.model.n_turbines)
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
    # Reward
    lambda_penalty          = float(cfg.model.lambda_penalty)

    train_throughput_max = vector_calorific_value * train_throughput_cap
    storage_lo = lower_storage_limit * upper_storage_limit
    storage_hi = upper_storage_limit

    SMOOTH_EPS = float(cfg.model.smooth_eps)
    beta_tt      = 1.0 / (SMOOTH_EPS * train_throughput_max)
    beta_storage = 1.0 / (SMOOTH_EPS * (storage_hi - storage_lo))
    # h2_throughput / efficiency lives on the train_throughput_max scale to
    # within a factor of `1/efficiency`; use that as the smoothing scale too.
    beta_h2      = 1.0 / (SMOOTH_EPS * train_throughput_max)

    @jit
    def _step(x: jnp.ndarray, u: jnp.ndarray, z: jnp.ndarray, node):
        x = jnp.ravel(x)
        u = jnp.ravel(u)
        z = jnp.ravel(z)

        _hydrogen_storage  = x[0]
        _train_throughput  = x[1]
        ramp_t             = u[0]
        hydrogen_throughput = u[1]

        # Disturbance: z is the renewable-energy value
        _renewable_energy = z[0]

        # All three trains share the same state and ramp.  Smooth saturation
        # at [0, train_throughput_max] so derivatives are well-defined at the
        # active set.
        tt_pre  = _train_throughput + ramp_t
        train_throughput = _smooth_min(
            _smooth_max(tt_pre, 0.0, beta_tt),
            train_throughput_max, beta_tt,
        )

        # Vector energy per train (3 identical trains).
        vector_energy_per_train = (
            train_throughput * (variable_energy_penalty / vector_calorific_value)
            * (1.0 - fixed_energy_penalty)
            + fixed_energy_penalty * variable_energy_penalty * train_throughput_cap
        )
        vector_energy_total = 3.0 * vector_energy_per_train

        # Electrolyser / fuel cell — smooth max(·, 0) so the kink at
        # hydrogen_throughput = 0 disappears.
        energy_electrolysis = _smooth_max(hydrogen_throughput / electrolyser_efficiency, 0.0, beta_h2)
        energy_fuelcell     = _smooth_max(-hydrogen_throughput / fuelcell_efficiency, 0.0, beta_h2)

        # Hydrogen consumption (3 trains).
        hydrogen_consumption = 3.0 * train_throughput / vector_molar_efficiency

        # Storage update (raw, before clipping for state propagation).
        hydrogen_storage = _hydrogen_storage + hydrogen_throughput - hydrogen_consumption

        # Constraints: positive = feasible margin (already in framework convention).
        lower_h2 = (hydrogen_storage - storage_lo) / upper_storage_limit
        upper_h2 = (upper_storage_limit - hydrogen_storage) / upper_storage_limit
        energy_balance = (
            n_turbines * _renewable_energy - energy_electrolysis - vector_energy_total + energy_fuelcell
        ) / (11.88 * n_turbines)

        # Smooth-clip storage for state passthrough.
        hydrogen_storage = _smooth_min(
            _smooth_max(hydrogen_storage, storage_lo, beta_storage),
            storage_hi, beta_storage,
        )

        # Reward: cost convention (lower = better). 3 identical trains, 3 identical ramps.
        penalty = 3.0 * jnp.square(ramp_t)
        reward = -(3.0 * train_throughput + 0.001 * _hydrogen_storage - lambda_penalty * penalty)

        outputs     = jnp.array([hydrogen_storage, train_throughput])      # F
        constraints = jnp.array([lower_h2, upper_h2, energy_balance])       # G
        cost        = jnp.atleast_1d(reward)                                # R

        return jnp.concatenate([outputs, constraints, cost], axis=0)

    return _step


def hydrogen_export_simulator(cfg: DictConfig):
    """Factory for hydrogen_export (steady-state). Returns a function with the
    standard case-study signature (cfg, design, input, aux, uncertainties, node)."""
    step = _make_hydrogen_export_step(cfg)

    def hydrogen_export_simulator_fn(cfg_unused: DictConfig, design_args, input_args, aux, uncertainties, node):
        return step(input_args, design_args, uncertainties, node)

    return hydrogen_export_simulator_fn


# -------------------------------------------------------------------------------- #
# ----- Biohydrogen (port of sample_envs/biohydrogen.py — fed-batch H2 culture) -- #
# -------------------------------------------------------------------------------- #

def _make_biohydrogen_step(cfg: DictConfig):
    """Factory: build a JIT-compiled biohydrogen step (fed-batch H2 culture).

    Port of sample_envs/biohydrogen.py.  States: X (biomass), C (carbon),
    N (culture nitrate), q (intracellular nitrogen quota), O (oxygen %),
    H (accumulated H2), F (accumulated feed volume).  Controls: u[0] =
    N_Fed (feed nitrate concentration mg/L), u[1] = log_F_in (natural log
    of feed flow rate L/h).  F_in = exp(u[1]) inside the step.  Sampling
    u[1] in log space biases Sobol/DEUS toward small F_in values, which
    is where the budget-feasible region lives.  No disturbance (Z_SIZE = 0).

    Global aux variable (sampled per trajectory): max_fr_per_node ∈ [0, 1].
    Sets the per-node F_in cap to max_fr_per_node · F_max / tf via the
    g_rate path constraint.  Lets Sobol / DEUS explore both how much to
    feed (u[1]) and how concentrated the feed strategy is (aux).

    Budget F ≤ F_max is also enforced as a path constraint (g_F).  Both
    PCs use the framework's negative-violated / zero-feasible convention.

    Returns `_step(x, u, z, aux, node)` emitting [F | G | R | Φ] of shape
    (7 + 4 + 1 + 1,) = 13.  Integration is in hours.
    """
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
    TF       = float(cfg.model.integration.tf)

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
        t3 = sigmoid(O * 2.0)                  # ~0 at O≈0, ~1 at O>0
        # Numerically-stable (1 - σ(2·O)): σ(-2·O) avoids the catastrophic
        # cancellation that makes (1 - σ(2·20)) round to 0 in float64.
        gate = sigmoid(-O * 2.0)
        f_N = sigmoid((N_SWITCH - N) * 1.0)    # ~1 at N<switch, ~0 otherwise

        # State derivatives.  F_in is the decision directly (L/h); cumulative
        # budget is enforced via g_F below, per-node cap via g_rate.
        dX = X * t1 - mu_d * X ** 2
        dC = -Y_CX * X * t1 + F_in * C_Fed
        dN = -Y_NX * X * t2 * mu_max + F_in * N_Fed
        dq = Y_qX * t2 * mu_max - t1 * q
        dO = Y_OX * X * t2 - Y_d * X ** 2 * t3 + O_Fed * F_in
        dH = Y_HX * X * gate * f_N
        dF = F_in
        dxdt = jnp.array([dX, dC, dN, dq, dO, dH, dF])

        # Path constraints — framework convention (negative = violated, 0 feasible).
        g_N = -jnp.maximum(N - N_MAX, 0.0) / N_MAX
        g_O = -jnp.maximum(O - O_MAX, 0.0) / O_MAX
        g_F = -jnp.maximum(F - F_max, 0.0) / F_max
        # Per-node feed cap from the global aux: F_in ≤ max_fr·F_max/tf
        f_in_cap = max_fr_per_node * F_max / TF
        g_rate = -jnp.maximum(F_in - f_in_cap, 0.0) / (F_max / TF)
        dgdt = jnp.array([g_N, g_O, g_F, g_rate])

        # Cost
        rwd = -Y_HX * X * gate * f_N
        phi = -H / TF

        return jnp.concatenate([
            jnp.ravel(dxdt), jnp.ravel(dgdt),
            jnp.atleast_1d(rwd), jnp.atleast_1d(phi),
        ], axis=0)

    return _step


def biohydrogen_simulator(cfg: DictConfig):
    """Factory for biohydrogen used when `unit_op == 'steady_state'`.

    Real workflow uses `unit_op: 'dynamic'` and routes through
    `unit_evaluators.ode.biohydrogen_ode`; this entry exists for symmetry
    with the other dynamic case studies.
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