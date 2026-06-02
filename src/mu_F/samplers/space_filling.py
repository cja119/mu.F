"""Sobol space-filling sampler for the design space."""
import numpy as np
from scipy.stats.qmc import Sobol


class SobolSampler:
    """Thin wrapper over the Sobol design-space sampler.

    Exposes a single sample_design_space entry point that the sampler
    constructors call, delegating to the module-level Sobol routine.

    """

    # ---- External Methods ----

    def __init__(self):
        pass

    def sample_design_space(self, n_design_args, bounds, n):
        """
        Draw n Sobol points over the design space.
        """
        return sobol_sample_design_space_nd(n_design_args, bounds, n)


def sobol_sample_design_space_nd(n_design_args, bounds, n):
    """
    Draw n Sobol points over the full-dimensional design space and
    scale them into the supplied bounds.
    """

    # sample over the whole dimensionality of the design space for coverage
    sobol_values = Sobol(n_design_args).random(n)

    lower_bound = np.array(bounds[0])
    upper_bound = np.array(bounds[1])
    design_args = lower_bound + (upper_bound - lower_bound) * sobol_values[:, :]

    return design_args
