"""Sobol sampler for covering the design space during reconstruction."""
import numpy as np
from scipy.stats.qmc import Sobol

class SobolSampler:
    """Quasi-random Sobol sampler over the design space.

    Wraps the scipy Sobol sequence and rescales samples to the supplied
    bounds, providing the design points consumed by the reconstruction
    sampling loop.

    """

    # ---- External Methods ----

    def __init__(self):
        pass

    def sample_design_space(self, n_design_args, bounds, n):
        """
        Draw n design points across the design space.
        """
        return sobol_sample_design_space_nd(n_design_args, bounds, n)


def sobol_sample_design_space_nd(n_design_args, bounds, n):
    """
    Draw a Sobol sample over the full design space and rescale it to the
    given lower/upper bounds.
    """
    sobol_values = Sobol(n_design_args).random(n)

    lower_bound = np.array(bounds[0])
    upper_bound = np.array(bounds[1])
    design_args = lower_bound + (upper_bound - lower_bound) * sobol_values[:, :]

    return design_args
