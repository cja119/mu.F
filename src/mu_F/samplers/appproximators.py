"""Box outer approximation of sampled data for the samplers."""
import jax.numpy as jnp


def calculate_box_outer_approximation(data, config, ndim=3):
    """
    Box outer approximation of the given data, padded by
    config.samplers.vol_scale relative to the data range.
    """

    if ndim == 3:
        # reshape data to account for the uncertainty parameters
        if data.ndim < 3:
            data = jnp.expand_dims(data, axis=-1)
        data = jnp.vstack([data[:,i,:].reshape(data.shape[0], data.shape[2]) for i in range(data.shape[1])])

    data_range = jnp.max(data, axis=0) - jnp.min(data, axis=0)

    # increment/decrement applied to each face of the box
    delta = config.samplers.vol_scale / 2 * data_range

    min_value = jnp.min(data - delta, axis=0)
    max_value = jnp.max(data + delta, axis=0)

    return [min_value.reshape(1,-1), max_value.reshape(1,-1)]
