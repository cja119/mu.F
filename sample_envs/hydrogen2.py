"""
Environment function definition for the hydrogen export case study.
"""

from functools import partial

from flax.core import FrozenDict
import jax.numpy as jnp
import jax


# -------------------------------------------------------------------------------- #
# ----------------------------- Combined Simulator ------------------------------- #
# -------------------------------------------------------------------------------- #

ENV_PARAMS = FrozenDict({
    "hydrogen_storage_capacity": 295435,
    "vector_storage_capacity": 0.0,
    "n_turbines": 1513,
    "n_trains_conversion": 4,
    "hfc_capacity": 832,
    "renewable_energy_value": 5.9,
    "train_throughput_capacity": 114,
    "vector_molar_efficiency": 0.888,
    "electrolyser_efficiency": 0.85,
    "fuelcell_efficiency": 0.6,
    "fixed_energy_penalty": 0.4,
    "variable_energy_penalty": 1.826,
    "vector_calorific_value": 18.8,
    "lower_ramp_limit": 0.25,
    "upper_ramp_limit": 0.1,
    "lower_storage_limit": 0.2,
    "upper_storage_limit": 295435,
    "feas_thresh": 0.1,
})

SHAPE_DICT = {
    "X_SIZE": 2,
    "U_SIZE": 2,
    "F_SIZE": 2,
    "G_SIZE": 5,
    "Z_SIZE": 1,
    "L_SIZE": 1,
    "PHI_SIZE": 0
}    

@partial(jax.jit, static_argnums=(0,))
def simulator(
    param_dict, node,  x: jnp.ndarray, u: jnp.ndarray, z: jnp.ndarray = None
) -> jnp.ndarray:
    """
    Combined simulator for the H2 Export model.
    """
    # Implement the simulation logic
    _hydrogen_storage = x[..., 0]
    _vector_throughput = x[..., 1]
    _renewable_energy = z[..., 0] if z is not None else param_dict["renewable_energy_value"]
    train_1_throughput = u[..., 0]
    train_2_throughput = u[..., 1]
    train_3_throughput = u[..., 2]
    hydrogen_throughput = u[..., 3]

    # Simulate the model dynamics here
    _active_trains = number_active_trains_eq(
        _vector_throughput,
        param_dict["train_throughput_capacity"],
        param_dict["vector_calorific_value"],
    )
    active_trains = number_active_trains_eq(
        vector_throughput,
        param_dict["train_throughput_capacity"],
        param_dict["vector_calorific_value"],
    )
    vector_energy = vector_energy_eq(
        vector_throughput,
        active_trains,
        param_dict["variable_energy_penalty"],
        param_dict["vector_calorific_value"],
        param_dict["fixed_energy_penalty"],
        param_dict["train_throughput_capacity"],
    )
    energy_electrolysis = energy_electrolysis_eq(
        hydrogen_throughput, param_dict["electrolyser_efficiency"]
    )

    energy_fuelcell = energy_fuelcell_eq(
        hydrogen_throughput, param_dict["fuelcell_efficiency"]
    )

    hydrogen_storage = hydrogen_storage_eq(_hydrogen_storage, hydrogen_throughput)

    # Calculate constraints
    lower_ramp_cons = vector_ramping_lower_cons(
        vector_throughput,
        _vector_throughput,
        _active_trains,
        param_dict["vector_calorific_value"],
        param_dict["lower_ramp_limit"],
        param_dict["train_throughput_capacity"],
    )
    upper_ramp_cons = vector_ramping_upper_cons(
        vector_throughput,
        _vector_throughput,
        _active_trains,
        param_dict["vector_calorific_value"],
        param_dict["upper_ramp_limit"],
        param_dict["train_throughput_capacity"],
        param_dict["n_trains_conversion"],
    )
    lower_h2_storage_cons = hydrogen_storage_lower_cons(
        hydrogen_storage,
        param_dict["lower_storage_limit"],
        param_dict["upper_storage_limit"],
    )
    upper_h2_storage_cons = hydrogen_storage_upper_cons(
        hydrogen_storage, param_dict["upper_storage_limit"]
    )

    energy_balance_cons = energy_balance_upper_cons(
        vector_energy,
        energy_electrolysis,
        energy_fuelcell,
        _renewable_energy,
        param_dict["n_turbines"],
    )

    # Calculate reward
    reward = jnp.broadcast_to(-vector_throughput, hydrogen_storage.shape)

    # Stack outputs and constraints - saturate outputs to reduce domain in longer graphs
    throughput_sat_l = vector_throughput_sat_l(
        _vector_throughput,
        _active_trains,
        param_dict["vector_calorific_value"],
        param_dict["lower_ramp_limit"],
        param_dict["train_throughput_capacity"],
    )
    throughput_sat_u = vector_throughput_sat_u(
        _vector_throughput,
        _active_trains,
        param_dict["vector_calorific_value"],
        param_dict["upper_ramp_limit"],
        param_dict["train_throughput_capacity"],
        param_dict["n_trains_conversion"],
    )

    hydrogen_storage = jnp.clip(
        hydrogen_storage,
        a_min=param_dict["lower_storage_limit"] * param_dict["upper_storage_limit"],
        a_max=param_dict["upper_storage_limit"],
    )
    vector_throughput = jnp.clip(
        vector_throughput, a_min=throughput_sat_l, a_max=throughput_sat_u
    )

    outputs = jnp.stack([hydrogen_storage, vector_throughput], axis=-1)
    constraints = jnp.stack(
        [
            lower_ramp_cons,
            upper_ramp_cons,
            lower_h2_storage_cons,
            upper_h2_storage_cons,
            energy_balance_cons,
        ],
        axis=-1,
    )
    reward = jnp.expand_dims(reward, axis=-1)

    return jnp.concatenate([outputs, constraints, reward], axis=-1)


# -------------------------------------------------------------------------------- #
# ------------------------------ Equations --------------------------------------- #
# -------------------------------------------------------------------------------- #
@partial(jax.jit, static_argnums=(1, 2))
def number_active_trains_eq(
    vector_throughput, train_throughput_capacity, vector_calorific_value
):
    """Calculate the number of active trains based on vector throughput"""
    # (GJ/ h) / (t(NH3) / train * GJ/t(NH3)) = trains
    return jnp.ceil(
        vector_throughput / (train_throughput_capacity * vector_calorific_value)
    )


@partial(jax.jit, static_argnums=(1,))
def energy_electrolysis_eq(hydrogen_throughput, electrolyser_efficiency):
    """Calculate the energy used for electrolysis"""
    # Number * GJ / h - GJ / h = GJ / h
    return jnp.maximum(hydrogen_throughput / electrolyser_efficiency, 0)


@partial(jax.jit, static_argnums=(1,))
def energy_fuelcell_eq(hydrogen_throughput, fuelcell_efficiency):
    """Calculate the energy used by the fuelcell"""
    # GJ / h - GJ / h = GJ / h
    return jnp.maximum(-hydrogen_throughput / fuelcell_efficiency, 0)


@jax.jit
def hydrogen_storage_eq(hydrogen_storage_prev, hydrogen_delta):
    """Update hydrogen storage based on stored hydrogen and throughput"""
    # GJ + GJ = GJ
    return hydrogen_storage_prev + hydrogen_delta


@partial(jax.jit, static_argnums=(3, 4, 5))
def hydrogen_delta_eq(
    vector_throughput,
    energy_electrolysis,
    energy_fuelcell,
    vector_molar_efficiency,
    electrolyser_efficiency,
    fuelcell_efficiency,
):
    """Calculate hydrogen removed based on vector throughput"""
    # (GJ / h) / (-) - - (GJ / h) / (-) - (GJ / h) / (-) = GJ / h
    return (
        vector_throughput / vector_molar_efficiency
        - energy_electrolysis / electrolyser_efficiency
        - energy_fuelcell / fuelcell_efficiency
    )


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def vector_energy_eq(
    vector_throughput,
    number_active_trains,
    variable_energy_penalty,
    vector_calorific_value,
    fixed_energy_penalty,
    train_throughput_capacity,
):
    """Update vector energy based on previous energy and throughput"""
    # GJ / h * ( GJ / tonne (NH3) * (tonne(NH3) / GJ)) * (1 - (-)) + Number * (-) * GJ / tonne (NH3) * (tonne(NH3) / h) = GJ / h
    return (
        vector_throughput
        * (variable_energy_penalty / vector_calorific_value)
        * (1 - fixed_energy_penalty)
        + number_active_trains
        * fixed_energy_penalty
        * variable_energy_penalty
        * train_throughput_capacity
    )


# -------------------------------------------------------------------------------- #
# ------------------------------ Constraints ------------------------------------- #
# -------------------------------------------------------------------------------- #
@partial(jax.jit, static_argnums=(3, 4, 5))
def vector_ramping_lower_cons(
    vector_throughput,
    _vector_throughput,
    _active_trains,
    vector_calorific_value,
    lower_ramp_limit,
    train_throughput_capacity,
):
    """Constraint for lower ramping limit of vector energy"""
    # GJ(NH3) / h - GJ(NH3) / h - (-) * Number * GJ(NH3) / h = GJ(NH3) / h
    return -(
        (_vector_throughput - vector_throughput) / vector_calorific_value
        - lower_ramp_limit * (_active_trains) * train_throughput_capacity
    ) / (lower_ramp_limit * (_active_trains) * train_throughput_capacity)


@partial(jax.jit, static_argnums=(2, 3, 4))
def vector_throughput_sat_l(
    _vector_throughput,
    _active_trains,
    vector_calorific_value,
    lower_ramp_limit,
    train_throughput_capacity,
):
    return (
        _vector_throughput
        - lower_ramp_limit
        * vector_calorific_value
        * _active_trains
        * train_throughput_capacity
    )


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def vector_ramping_upper_cons(
    vector_throughput,
    _vector_throughput,
    _active_trains,
    vector_calorific_value,
    upper_ramp_limit,
    train_throughput_capacity,
    total_trains,
):
    """Constraint for upper ramping limit of vector energy"""
    # GJ(NH3) / h - GJ(NH3) / h - (-) * Number * GJ(NH3) / h = GJ(NH3) / h
    return -(
        (vector_throughput - _vector_throughput) / vector_calorific_value
        - upper_ramp_limit
        * (total_trains - _active_trains + 1)
        * train_throughput_capacity
    ) / (
        upper_ramp_limit
        * (total_trains - _active_trains + 1)
        * train_throughput_capacity
    )


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def vector_throughput_sat_u(
    _vector_throughput,
    _active_trains,
    vector_calorific_value,
    upper_ramp_limit,
    train_throughput_capacity,
    total_trains,
):
    return (
        _vector_throughput
        + vector_calorific_value
        * upper_ramp_limit
        * (total_trains - _active_trains + 1)
        * train_throughput_capacity
    )


@partial(jax.jit, static_argnums=(1, 2))
def hydrogen_storage_lower_cons(
    hydrogen_storage, lower_storage_limit, upper_storage_limit
):
    """Constraint for lower hydrogen storage limit"""
    # GJ - (-) * GJ = GJ
    return (
        hydrogen_storage - lower_storage_limit * upper_storage_limit
    ) / upper_storage_limit


@partial(jax.jit, static_argnums=(1,))
def hydrogen_storage_upper_cons(hydrogen_storage, upper_storage_limit):
    """Constraint for upper hydrogen storage limit"""
    # GJ - GJ = GJ
    return (upper_storage_limit - hydrogen_storage) / upper_storage_limit


@partial(jax.jit, static_argnums=(4,))
def energy_balance_upper_cons(
    vector_energy, energy_electrolysis, energy_fuelcell, renewable_energy, n_turbines
):
    """Constraint for energy balance"""
    # GJ / h - GJ / h - GJ / h = GJ / h
    return (
        n_turbines * renewable_energy
        - energy_electrolysis
        - vector_energy
        + energy_fuelcell
    ) / (11.88 * n_turbines)
