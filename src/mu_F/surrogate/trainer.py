"""Trainers and rebuilders that fit and reconstruct surrogate models per model type."""
from abc import ABC
from typing import List, Tuple
from functools import partial

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig


from mu_F.surrogate.data_utils import binary_classifier_data_preparation, ctg_data_preparation, regression_node_data_preparation, forward_evaluation_data_preparation
from mu_F.surrogate.gp_utils import train as train_gp
from mu_F.surrogate.gp_utils import build_gp
from mu_F.surrogate.nn_utils import hyperparameter_selection as train_ann
from mu_F.surrogate.nn_utils import Dataset, build_ann
from mu_F.surrogate.svm_utils import train as train_svm, build_svm


class TrainerBase(ABC):
    """Abstract trainer interface.

    Stores the graph context and model type and declares the data / training
    callbacks that the concrete Trainer implements.

    """

    # ---- External Methods ----

    def __init__(self, graph, unit_index, cfg, model_type, iterate, data_str: str = 'classifier_training'):
        self.cfg = cfg
        self.model_type = model_type
        self.model_class = model_type[0]
        self.model_subclass = model_type[1]
        self.model_surrogate = model_type[2]
        self.graph = graph
        self.unit_index = unit_index
        self.iterate = iterate
        self.data_str = data_str

    def get_model(self, path: str, model_object) -> None:
        """Model accessor hook implemented by subclasses."""
        pass

    def load_trainer_methods(self, prediction_string:str) -> None:
        """Trainer-selection hook implemented by subclasses."""
        pass

    def train(self) -> jnp.ndarray:
        """Training hook implemented by subclasses."""
        pass

class Trainer(TrainerBase):
    """Fit the surrogate model selected by model type.

    Prepares the appropriate dataset, dispatches to the ANN / GP / SVM
    trainer, and stores the standardised / unstandardised models and metrics.

    """

    # ---- External Methods ----

    def __init__(self, graph, unit_index, cfg, model_type, iterate, data_str: str = 'classifier_training'):
        super().__init__(graph, unit_index, cfg, model_type, iterate, data_str)
        self.x_scalar_override = None

    def get_model_object(self, string: str) -> None:
        """Return the named trained model or standardisation metrics."""
        if string == 'standardised_model':
            return self.standardised_model
        elif string == 'unstandardised_model':
            return self.unstandardised_model
        elif string == 'standardisation_metrics_input':
            return self.standardisation_metrics_input
        elif string == 'standardisation_metrics_output':
            return self.standardisation_metrics_output

    def load_trainer_methods(self) -> None:
        """Select the trainer callable for the configured model subclass."""
        if self.model_subclass == 'ANN':
            if self.model_class == 'regression':
                self.trainer = partial(train_ann, model_type='regressor', model_surrogate=self.model_surrogate, x_scalar_override=self.x_scalar_override)
            elif self.model_class == 'classification':
                self.trainer = partial(train_ann, model_type='classifier', model_surrogate=self.model_surrogate)
        elif self.model_subclass == 'GP':
            self.trainer = train_gp
        elif self.model_subclass == 'SVM':
            self.trainer = partial(train_svm, unit_index=self.unit_index, iterate=self.iterate)
        return 

    def get_data(self, successor_node: int = None) -> None:
        """Prepare the training dataset for the configured model and surrogate type."""
        if (self.model_class == 'regression') and (self.model_surrogate != 'forward_evaluation_surrogate')  and (self.model_surrogate != 'ctg_surrogate'):  # TODO act on the right graph component (edge or node)
            dataset = regression_node_data_preparation(self.graph, self.unit_index, self.cfg, data_str=self.data_str)
        elif (self.model_class == 'regression') and (self.model_surrogate == 'forward_evaluation_surrogate'):
            dataset = forward_evaluation_data_preparation(self.graph, self.unit_index, self.cfg, successor_node)
        elif (self.model_class == 'regression') and (self.model_surrogate == 'ctg_surrogate'):
            dataset = ctg_data_preparation(self.graph, self.unit_index, self.cfg)
        elif self.model_class == 'classification':  # node-level feasibility approximation
            data_points, labels = binary_classifier_data_preparation(self.graph, self.unit_index, self.cfg, data_str=self.data_str)
            if self.model_subclass == 'SVM' : dataset = (data_points, labels)
            elif self.model_subclass == 'ANN' : dataset = Dataset(X=data_points, y=labels)
        return dataset


    def train(self, node=None) -> jnp.ndarray:
        """Fit the model, store the resulting components and return the model."""
        if node is None:
            dataset = self.get_data() # TODO
        else:
            dataset = self.get_data(successor_node=node)

        self.load_trainer_methods()
        model, args, serialised_data = self.trainer(self.cfg, dataset, self.cfg.surrogate.num_folds)

        if self.model_class == 'regression':
            assert len(args) == 4, "Regression model training should return 4 arguments; standardised model (i.e. model mapping from and into a standardised space), unstandardised model (i.e. model mapping from and into original data space), standardisation metrics for input and output"
            self.standardised_model, self.unstandardised_model, self.standardisation_metrics_input, self.standardisation_metrics_output = args
        elif self.model_class == 'classification':
            assert len(args) == 3, "Classification model training should return 3 arguments; standardised model, unstandardised model, standardisation metrics for input and output"
            self.standardised_model, self.unstandardised_model, self.standardisation_metrics_input = args

        self.serialised_data = serialised_data

        del dataset

        return model

    def get_serialised_model_data(self) -> dict:
        """Expose the serialised model data for reconstruction."""
        return self.serialised_data



class Rebuilder(ABC):
    """Reconstruct a surrogate callable from serialised problem data.

    Selects the builder matching the model type and returns the prediction
    function used outside the training context.

    """

    # ---- External Methods ----

    def __init__(self, cfg: DictConfig, model_type: str, problem_data: dict) -> None:
        self.cfg = cfg
        self.model_type = model_type
        self.problem_data = problem_data
        self.model_class = model_type[0]
        self.model_subclass = model_type[1]
        self.model_surrogate = model_type[2]
        self.load_rebuilder_methods()

    def load_rebuilder_methods(self) -> None:
        """Select the builder callable for the configured model subclass."""
        if self.model_subclass == 'ANN':
            if self.model_class == 'regression':
                self.builder = partial(build_ann, model_class='regressor')
            elif self.model_class == 'classification':
                self.builder = partial(build_ann, model_class='classifier')
        elif self.model_subclass == 'GP':
            self.builder = build_gp
        elif self.model_subclass == 'SVM':
            self.builder = build_svm
        return 

    def rebuild(self) -> jnp.ndarray:
        """Reconstruct and return the surrogate callable."""
        return self.builder(self.cfg, self.problem_data)