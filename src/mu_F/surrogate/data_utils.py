"""Data preparation helpers for training surrogate models on graph node/edge data."""
import jax.numpy as jnp
import jax
import pandas as pd
from omegaconf import DictConfig
from dataclasses import dataclass


@dataclass
class StandardisationMetrics:
    mean: jnp.ndarray
    std: jnp.ndarray


def evaluate_classifier(classifier, data_points, cfg, index):
    """Score points with a classifier and return a labelled DataFrame."""
    predictions = classifier.predict(data_points)
    mapping = {'Prediction': predictions} | {name: data_points[:,i]*cfg.scale[index][i] for i,name in enumerate(cfg.process_space_names[index])}
    df = pd.DataFrame(mapping)

    return df

def forward_evaluation_data_preparation(graph: dict, unit_index, cfg: DictConfig = None, successor_node: int = None):
    """Fetch the forward-evaluation training data stored on a graph edge."""
    input_data = graph.edges[unit_index, successor_node]["surrogate_training"]

    return input_data

def regression_node_data_preparation(graph: dict, unit_index: int, cfg: DictConfig = None, data_str: str = 'probability_map_training'):
    """Fetch regression training data from a node, or the graph for the centralised DS."""
    if unit_index is not None:
        data = graph.nodes[unit_index][data_str]
    else:
        # unit_index None -> regression model parameterising the centralised DS
        data = graph.graph[data_str]

    return data

def ctg_data_preparation(graph: dict, unit_index: int, cfg: DictConfig = None):
    """Fetch the cost-to-go training data stored on a graph node."""
    data = graph.nodes[unit_index]["ctg_func_training"]
    return data

def binary_classifier_data_preparation(
    graph: dict,
    unit_index: int,
    cfg: DictConfig = None,
    data_str: str = 'classifier_training'
):
    """
    Build balanced binary-classifier training data from node/graph support
    and labels, relabelling to {-1, +1} per the feasibility convention.
    """
    if type(unit_index) is int:
        assert unit_index in graph.nodes, f"Unit index {unit_index} not found in graph nodes."
        data = graph.nodes[unit_index][data_str]
    else:
        # non-int unit_index -> classifier parameterising the centralised DS
        data = graph.graph[data_str]

    support = data.X
    labels = data.y
    support = support.reshape(support.shape[0], -1)  # (N, d), keeps d=1 case

    if cfg.formulation == 'deterministic':
        if cfg.samplers.notion_of_feasibility == 'positive':
            select_cond = jnp.min(labels, axis=1)  >= 0
        else:
            select_cond = jnp.max(labels, axis=1)  <= 0
    elif cfg.formulation == 'probabilistic':
        select_cond = labels >= cfg.samplers.unit_wise_target_reliability[unit_index]
    else:
        raise ValueError(f"Formulation {cfg.formulation} not recognised. Please use 'probabilistic' or 'deterministic'.")

    # feasible label is negative: problem coupling always minimises
    labels = jnp.where(select_cond, -1, 1)

    # augment data to equalise the negative and positive class counts
    num_pos = jnp.sum(labels == 1)
    num_neg = jnp.sum(labels == -1)
    Key = jax.random.PRNGKey(0)

    if num_pos > num_neg:
        # resample negatives to match the positive count
        neg_indices = jnp.where(labels == -1)[0]
        selected_indices = jax.random.choice(Key, neg_indices, shape=(num_pos - num_neg,))
        support = jnp.concatenate([support, support[selected_indices]], axis=0)
        labels = jnp.concatenate([labels, labels[selected_indices]], axis=0)
    elif (num_neg > num_pos) and (min(num_pos, num_neg) > 0):
        # resample positives to match the negative count
        pos_indices = jnp.where(labels == 1)[0]
        selected_indices = jax.random.choice(Key, pos_indices, shape=(num_neg - num_pos,))
        support = jnp.concatenate([support, support[selected_indices]], axis=0)
        labels = jnp.concatenate([labels, labels[selected_indices]], axis=0)

    return support, labels



def return_subsample_of_data(data, labels, subsample_size):
    """Subsample data to a fixed size, keeping all negatives and topping up with positives."""
    if data.shape[0] > subsample_size:
        select_cond = labels == 1
        data_pos = data[select_cond.squeeze(),:]
        select_cond = labels == -1
        data_neg = data[select_cond.squeeze(),:]
        assert subsample_size > data_neg.shape[0], f"Negative data size {data_neg.shape[0]} is larger than subsample size {subsample_size}"

        data_new = jnp.vstack([data_neg, data_pos[:subsample_size-data_neg.shape[0],:]])
        labels_new = jnp.vstack([-jnp.ones((data_neg.shape[0],1)), jnp.ones((subsample_size-data_neg.shape[0],1))*1 ])
        return data_new, labels_new
    else:
        return data, labels
