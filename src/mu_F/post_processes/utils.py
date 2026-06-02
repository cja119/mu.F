"""Sobol sampling and box-scaling helpers for post-processing."""
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

def generate_sobol(n_dim: int, n_points: int) -> np.ndarray:
    """
    Generate Sobol samples in the unit hypercube [0, 1]^n_dim.
    """
    sobol_samples = qmc.Sobol(d=n_dim, scramble=True).random(n_points)
    return sobol_samples

def scale_sobol(
        sobol: np.ndarray, lower_bounds: jnp.ndarray, upper_bounds: jnp.ndarray
        ) -> jnp.ndarray:
    """
    Scale unit-hypercube Sobol samples into the given lower/upper bounds.
    """
    return jnp.array(lower_bounds) + (jnp.array(upper_bounds) - jnp.array(lower_bounds)) * jnp.array(sobol)

def scaling_fn_box(x: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
      """
      Scale points from [0, 1]^n into the box [d_lower, d_upper]^n.
      """
      d = d.reshape(2,-1)
      d_lower = d[0, :].reshape(1,-1)
      d_upper = d[1, :].reshape(1,-1)
      return d_lower + (d_upper - d_lower) * x.reshape(1,-1)