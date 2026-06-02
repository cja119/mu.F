"""Surrogate model wrappers used to parameterise feasibility, probability and unit dynamics."""
from abc import ABC
from typing import Tuple

from omegaconf import DictConfig
import jax.numpy as jnp

from mu_F.surrogate.predictor import Predictor
from mu_F.surrogate.trainer import Trainer, Rebuilder


class SurrogateBase(ABC):
    """Abstract surrogate interface.

    Validates the requested model type and declares the fit / predict /
    get_model callbacks that concrete surrogates implement.

    """

    # ---- External Methods ----

    def __init__(self, graph, unit_index: int, cfg: DictConfig, model_type: str, iterate: int) -> None:

        self.cfg = cfg
        self.graph = graph
        self.unit_index = unit_index
        assert type(model_type) == tuple, "model_type must be a tuple of strings"
        self.model_type = model_type
        self.model_class = model_type[0]
        self.model_subclass = model_type[1]
        self.model_surrogate = model_type[2]
        self.iterate = iterate

        assert self.model_class in ["regression", "classification"], "model_class must be either 'regression' or 'classification'"
        if self.model_class == "regression":
            assert self.model_subclass in ["ANN", "GP"], "regression model_subclass must be either 'ANN' or 'GP'"
        elif self.model_class == "classification":
            assert self.model_subclass in ["ANN", "SVM"], "classifier model_subclass must be either 'ANN', or 'SVM'"

        assert self.model_surrogate in ["live_set_surrogate", "probability_map_surrogate", "forward_evaluation_surrogate", "post_process_forward",  'ctg_surrogate'], "model_surrogate must be one of ['live_set_surrogate', 'probability_map_surrogate', 'forward_evaluation_surrogate', post_process_forward] indicating a parameterisation of the feasible region, probability map or unit dynamics respectively."

    def fit(self) -> None:
        """Fit hook implemented by concrete surrogates."""
        pass

    def predict(self, string: str, X: jnp.ndarray) -> jnp.ndarray:
        """Predict hook implemented by concrete surrogates."""
        pass

    def get_model(self, string:str) -> callable:
        """Return the named prediction function; implemented by subclasses."""
        pass



class Surrogate(SurrogateBase):
    """Concrete surrogate backed by a Trainer and Predictor.

    Trains the requested model and exposes prediction functions used by the
    forward pass and the feasibility / probability maps.

    """

    # ---- External Methods ----

    def __init__(self, graph, unit_index, cfg: DictConfig, model_type: tuple[str], iterate:int, data_str: str='classifier_training') -> None:
        super().__init__(graph, unit_index, cfg, model_type, iterate)
        self.model = None
        self.trainer = Trainer(graph, unit_index, cfg, model_type, iterate, data_str)
        self.predictor = Predictor(cfg, model_type)

    def fit(self, node=None) -> None:
        """Train the model and load it into the predictor."""
        self.trainer.train(node)
        self.predictor.load_trained_model(self.trainer)

    def predict(self, string:str, X: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the named prediction function on X."""
        return self.predictor.predict(self.get_model(string), X)

    def get_model(self, string: str) -> callable:
        """Return the named prediction function from the predictor."""
        return self.predictor.return_prediction_function(string)

    def get_serailised_model_data(self) -> Tuple:
        """Expose the serialised model data for reconstruction."""
        return self.predictor.get_serialised_model_data()

    def from_method(cls, graph, data_str) -> None:
        """Build a Surrogate from an existing instance's model attributes."""
        return Surrogate(graph, cls.unit_index, cls.cfg, (cls.model_class, cls.model_subclass, cls.model_surrogate), cls.iterate, data_str)



class SurrogateReconstruction(ABC):
    """Rebuild a serialised surrogate from stored problem data.

    Validates the model type and delegates to a Rebuilder to reconstruct the
    callable prediction function outside the training context.

    """

    # ---- External Methods ----

    def __init__(self, cfg: DictConfig, model_type: str, problem_data: dict) -> None:
        self.cfg = cfg
        self.model_type = model_type
        self.problem_data = problem_data

        self.model_class = model_type[0]
        self.model_subclass = model_type[1]
        self.model_surrogate = model_type[2]

        assert self.model_class in ["regression", "classification"], "model_class must be either 'regression' or 'classification'"
        if self.model_class == "regression":
            assert self.model_subclass in ["ANN", "GP"], "regression model_subclass must be either 'ANN' or 'GP'"
        elif self.model_class == "classification":
            assert self.model_subclass in ["ANN", "SVM"], "classifier model_subclass must be either 'ANN', or 'SVM'"

        assert self.model_surrogate in ["live_set_surrogate", "probability_map_surrogate", "forward_evaluation_surrogate", "post_process_forward", "ctg_surrogate"], "model_surrogate must be one of ['ctg_surrogate', 'live_set_surrogate', 'probability_map_surrogate', 'forward_evaluation_surrogate', 'post_process_forward'] indicating a parameterisation of the feasible region, probability map or unit dynamics respectively."


    def rebuild_model(self):
        """Reconstruct the surrogate callable via a Rebuilder."""
        return Rebuilder(self.cfg, self.model_type, self.problem_data).rebuild()
