"""Data structures threaded through one per-node DEUS evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import jax.numpy as jnp


@dataclass
class EvalContext:
    """Mutable bag for one batch of designs

    Fields are grouped by the evaluation step that fills them; each per-node
    DEUS evaluation threads one instance through the formulation stages.

    """
    d: jnp.ndarray
    p: jnp.ndarray
    collecting: bool = False                   # True only in the EFP phase
    # node_evaluator (the real model)
    outputs:   Optional[jnp.ndarray] = None    # (n_d, n_theta, n_out)
    design:    Optional[jnp.ndarray] = None
    inputs:    Optional[jnp.ndarray] = None
    aux:       Optional[jnp.ndarray] = None
    # current (REAL, consumes outputs)
    process:   Optional[jnp.ndarray] = None     # raw per-scenario g  (n_d, n_theta, n_g)
    node_cost: Optional[jnp.ndarray] = None     # (n_d, n_theta, 1)
    # upstream / downstream stage outputs — meaning is formulation-specific
    # (probabilistic: scalar feasibility P_up / P_down; deterministic: raw g)
    upstream:   Optional[jnp.ndarray] = None
    downstream: Optional[jnp.ndarray] = None
    keep:       Optional[jnp.ndarray] = None     # feasible design indices (the CTG gate)
    # assemble
    p_feas:    Optional[jnp.ndarray] = None     # (n_d,)
    ctg:       Optional[jnp.ndarray] = None     # (n_keep, 1)
    result:    Optional[jnp.ndarray] = None     # what s() returns to DEUS


class TrainingStore:
    """Surrogate training data for one node

    Accumulates per-target design/output/value chunks during evaluation and
    serves them back as stacked matrices for surrogate fitting.

    """

    # ---- External Methods ----

    def __init__(self):
        self._data: dict = {}

    def add(self, key: str, designs, outputs, values) -> None:
        """Append a chunk of training data under the given target key."""
        chunk = self._data.setdefault(key, {'designs': [], 'outputs': [], 'values': []})
        chunk['designs'].append(np.asarray(designs))
        chunk['outputs'].append(np.asarray(outputs))
        chunk['values'].append(np.asarray(values))

    def has(self, key: str) -> bool:
        """Whether any data has been stored for the given target key."""
        return key in self._data

    def matrix(self, key: str):
        """
        Return the stacked (X, y) matrices for training surrogate `key`.
        """
        chunk = self._data[key]
        return np.vstack(chunk['designs']), np.vstack(chunk['values'])
