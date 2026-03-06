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
    "lambda": 10e-5
})

SHAPE_DICT = {
    "X_SIZE": 4,
    "U_SIZE": 4,
    "F_SIZE": 4,
    "G_SIZE": 3,
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
    weather_map = jnp.array(
        [
            11.88, 11.88, 11.88, 11.88, 11.88, 11.88, 11.88, 11.88, 11.88, 11.88,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]
    )
    weather_map = 11.88 * jnp.linspace(0.75, 0.25, 20)
    
    # Implement the simulation logic
    _hydrogen_storage = x[..., 0]
    _train_1_throughput = x[..., 1]
    _train_2_throughput = x[..., 2]
    _train_3_throughput = x[..., 3]
    _renewable_energy = jnp.take(weather_map, node)  # z[..., 0] if z is not None else param_dict["renewable_energy_value"]
    ramp_t1 = u[..., 0]
    ramp_t2 = u[..., 1]
    ramp_t3 = u[..., 2]
    hydrogen_throughput = u[..., 3]

    # Find the new throughput, saturating at the maximum throughput capacity
    train_1_throughput = jnp.clip(_train_1_throughput + ramp_t1, a_min=0.0, a_max=18.8*param_dict["train_throughput_capacity"])
    train_2_throughput = jnp.clip(_train_2_throughput + ramp_t2, a_min=0.0, a_max=18.8*param_dict["train_throughput_capacity"])
    train_3_throughput = jnp.clip(_train_3_throughput + ramp_t3, a_min=0.0, a_max=18.8*param_dict["train_throughput_capacity"])

    # Simulate the model dynamics 
    vector_energy_1,  = vector_energy_eq(
        train_1_throughput,
        param_dict["variable_energy_penalty"],
        param_dict["vector_calorific_value"],
        param_dict["fixed_energy_penalty"],
        param_dict["train_throughput_capacity"]
    )
    vector_energy_2,  = vector_energy_eq(
        train_2_throughput,
        param_dict["variable_energy_penalty"],
        param_dict["vector_calorific_value"],
        param_dict["fixed_energy_penalty"],
        param_dict["train_throughput_capacity"]
    )
    vector_energy_3,  = vector_energy_eq(
        train_3_throughput,
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
    lower_h2_storage_cons = hydrogen_storage_lower_cons(
        hydrogen_storage,
        param_dict["lower_storage_limit"],
        param_dict["upper_storage_limit"],
    )
    upper_h2_storage_cons = hydrogen_storage_upper_cons(
        hydrogen_storage, param_dict["upper_storage_limit"]
    )

    energy_balance_cons = energy_balance_upper_cons(
        vector_energy_1 + vector_energy_2 + vector_energy_3,
        energy_electrolysis,
        energy_fuelcell,
        _renewable_energy,
        param_dict["n_turbines"],
    )

    hydrogen_storage = jnp.clip(
        hydrogen_storage,
        a_min=param_dict["lower_storage_limit"] * param_dict["upper_storage_limit"],
        a_max=param_dict["upper_storage_limit"],
    )


    # penalty can take a mazimum value of 861,180.6252
    # maximum reward is 6429.6
    # lambda between 10e-5 and 10e-6 should be appropriate

    _lambda = param_dict["lambda"]
    penalty = jnp.square(ramp_t1) + jnp.square(ramp_t2) + jnp.square(ramp_t3)

    reward = jnp.broadcast_to(-sum([train_1_throughput, train_2_throughput, train_3_throughput, - _lambda * penalty]), hydrogen_storage.shape)
    outputs = jnp.stack([hydrogen_storage, train_1_throughput, train_2_throughput, train_3_throughput], axis=-1)

    constraints = jnp.stack(
        [
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


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def vector_energy_eq(
    vector_throughput,
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
        + fixed_energy_penalty
        * variable_energy_penalty
        * train_throughput_capacity
    )


# -------------------------------------------------------------------------------- #
# ------------------------------ Constraints ------------------------------------- #
# -------------------------------------------------------------------------------- #
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
