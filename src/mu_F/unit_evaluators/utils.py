"""Helper functions and data containers for the ODE term definitions."""
from abc import ABC
from jax import jit
import jax.numpy as jnp
import numpy as np

@jit
def arrhenius_kinetics_fn(decision_params, uncertainty_params, Ea, A, R):
    temperature = decision_params[0]  # temperature is always the first decision parameter
    return A * jnp.exp(-Ea / (R * temperature))

@jit
def arrhenius_kinetics_fn_2(decision_params, uncertainty_params, Ea, R):
    temperature = decision_params[0]  # temperature is always the first decision parameter
    A = uncertainty_params
    return A * jnp.exp(-Ea / (R * temperature))

class RegressorData(ABC):
    """Accumulates input/output samples for training a regressor.

    Collects appended samples and, on request, packages them into a
    dataset stored on the graph for downstream post-processing regressors.

    """

    # ---- External Methods ----

    def __init__(self, cfg):
        self.cfg = cfg
        self.inputs, self.outputs = [], []

    def append_to_live_set(self, x, y):
        """
        Append a single input/output sample to the live set.
        """
        self.inputs.append(x)
        self.outputs.append(y.reshape(-1,1))
        return

    def load_regression_data_to_graph(self, graph=None, str_='post_process_lower'):
        """
        Stack the accumulated samples into a dataset and attach
        it to the graph under the given identifier for regressor training.
        """
        if graph is None:
            raise ValueError("Graph must be provided to load regression data.")

        inputs, outputs = np.vstack(self.inputs), np.vstack(self.outputs)
        graph.graph[str_+ 'regressor_training'] = RegressorDataset(inputs, outputs)
        return graph

class RegressorDataset(ABC):
    """Minimal input/output container that ensures at least 2-D arrays.

    Wraps the supplied X and y, expanding their last axis where needed so
    downstream regressors always receive rank-2 design and output matrices.

    """

    # ---- External Methods ----

    def __init__(self, X, y):
        self.input_rank = len(X.shape)
        self.output_rank = len(y.shape)
        self.X = X if self.input_rank >= 2 else np.expand_dims(X,axis=-1)
        self.y = y if self.output_rank >=2 else np.expand_dims(y, axis=-1)
