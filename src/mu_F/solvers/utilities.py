"""Solver-layer utilities: batch sizing, Sobol initial guesses, surrogate rebuild."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy.stats import qmc

from mu_F.surrogate.surrogate import SurrogateReconstruction


__all__ = [
    "determine_batches",
    "create_batches",
    "generate_initial_guess",
    "construct_model",
]


def determine_batches(n_starts, num_workers):
    """
    Chunk n_starts items into batches sized num_workers, with any remainder
    in a final short batch.
    """
    num_batches = n_starts // num_workers
    batch_size = [num_workers for _ in range(num_batches)]
    if n_starts % num_workers != 0:
        batch_size.append(n_starts % num_workers)
    return batch_size, np.cumsum(batch_size)


def create_batches(batch_size, object_iterable):
    """
    Split object_iterable along its leading axis according to batch_size.
    """
    cumsum = np.cumsum(batch_size) - batch_size[0]
    return [
        object_iterable[cumsum[i] : cumsum[i] + batch_size[i]]
        for i in range(len(batch_size))
    ]


def generate_initial_guess(n_starts, n_d, bounds, seed=None):
    """
    Draw n_starts scrambled Sobol points inside the given box bounds.
    n_d is inferred from bounds[0]; the positional arg is kept for back-compat.
    """
    n_d = len(bounds[0])
    lower_bound = bounds[0]
    upper_bound = bounds[1]
    sobol_samples = qmc.Sobol(d=n_d, scramble=True, seed=seed).random(n_starts)
    return jnp.array(lower_bound) + (jnp.array(upper_bound) - jnp.array(lower_bound)) * sobol_samples


def construct_model(problem_data, cfg, supervised_learner: str,
                    model_type: str, model_surrogate: str):
    """
    Rebuild a live Surrogate callable from the serialised weights dict.
    Only reached on checkpoint-resume paths where a pickle has dropped the
    live callable; normal runs read it off the graph directly.
    """
    return SurrogateReconstruction(
        cfg,
        (supervised_learner, model_type, model_surrogate),
        problem_data,
    ).rebuild_model()
