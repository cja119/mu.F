"""General-purpose utilities: figure saving, graph serialisation and dataset containers."""
import cloudpickle as pickle
from pathlib import Path

import inspect
from functools import wraps
from abc import ABC
import jax.numpy as jnp
from typing import List


SVG_MAX_BYTES = 500 * 1024 * 1024


from pathlib import Path


def enforce_svg_size_limit(path, max_bytes=SVG_MAX_BYTES, delete_oversize=False):
    """Return file size if an SVG exceeds the configured limit (no exception)."""
    p = Path(path)
    if not p.exists() or p.suffix.lower() != ".svg":
        return None

    size = p.stat().st_size
    if size <= max_bytes:
        return size

    if delete_oversize:
        p.unlink(missing_ok=True)

    return size


def savefig_with_svg_limit(save_callable, path, *args, max_svg_bytes=SVG_MAX_BYTES, **kwargs):
    """Save a figure and enforce the global SVG size limit, falling back to PNG if exceeded."""
    p = Path(path)

    save_callable(p, *args, **kwargs)

    if p.suffix.lower() != ".svg":
        return None

    size = enforce_svg_size_limit(p, max_bytes=max_svg_bytes)
    if size is None or size <= max_svg_bytes:
        return None

    png_path = p.with_suffix(".png")

    p.unlink(missing_ok=True)

    png_kwargs = dict(kwargs)
    png_kwargs.pop("format", None)
    png_kwargs.pop("dpi", None)

    png_kwargs.setdefault("bbox_inches", "tight")
    png_kwargs.setdefault("pad_inches", 0.05)

    dpi = 300
    if size > 5 * max_svg_bytes:
        dpi = 150
    elif size > 2 * max_svg_bytes:
        dpi = 200

    save_callable(
        png_path,
        *args,
        format="png",
        dpi=dpi,
        **png_kwargs,
    )

    return None

def save_graph(G, mode):
    """
    Serialise the graph to a pickle, first dropping all
    non-serializable evaluators, surrogates and classifiers.
    """
    # drop everything not needed for serialisation
    for node in G.nodes:
        G.nodes[node]["forward_evaluator"] = None  # drop forward evaluators
        for predec in G.predecessors(node):
            G.edges[predec, node]["edge_fn"] = None  # drop edge functions
            G.edges[predec, node]["forward_surrogate"] = None  # drop forward surrogates
            G.edges[predec, node]["aux_filter"] = None  # drop backward surrogates
        G.nodes[node]["constraints"] = None  # drop constraint functions (non-serializable)
        G.nodes[node]["constraints_vmap"] = None  # drop backward evaluators
        G.nodes[node]['classifier'] = None
        G.nodes[node]['unit_params_fn'] = None
        G.nodes[node]['post_process_constraints'] = None

    G.graph['post_process_classifier'] = None  # drop post process classifier
    G.graph['post_process_lower_classifier'] = None  # drop post process lower classifier
    G.graph['post_process_upper_classifier'] = None  # drop post process upper classifier
    G.graph['post_process_lower_regressor'] = None  # drop post process lower regressor
    G.graph['post_process_upper_regressor'] = None  # drop post process upper regressor
    G.graph['post_process_training_methods'] = None  # drop post process training methods
    G.graph['post_process_solver_methods'] = None  # drop post process solver methods
    G.graph['post_process_solution_evaluator'] = None  # drop post process solution evaluator
    G.graph['post_process_solution_visualiser'] = None  # drop post process solution visualiser

    filename = f"graph_{mode}.pickle"
    with open(filename, 'wb') as f:
        protocol = getattr(pickle, "HIGHEST_PROTOCOL", getattr(pickle, "DEFAULT_PROTOCOL", 4))
        pickle.dump(G, f, protocol=protocol)
    return


class DatasetObject(ABC):
    """Growable design/parameter/output dataset of rank-3 batches.

    Stores each added (d, p, y) triple as a list of expanded arrays so the
    accumulated samples can be consumed batch-wise by the data processors.

    """

    # ---- External Methods ----

    def __init__(self, d, p, y):
        self.input_rank = len(d.shape)
        self.p_rank = len(p.shape)
        self.output_rank = len(y.shape)

        d = d if self.input_rank > 1 else jnp.expand_dims(d,axis=-1)
        p = p if self.p_rank > 1 else jnp.expand_dims(p,axis=-1)
        y = y if self.output_rank >1 else jnp.expand_dims(y,axis=-1)

        self.input_rank = len(d.shape)
        self.p_rank = len(p.shape)
        self.output_rank = len(y.shape)

        self.d = [d if self.input_rank > 2 else jnp.expand_dims(d, axis=-1)]
        self.p = [p if self.p_rank > 2 else jnp.expand_dims(p,axis=-1)]
        self.y = [y if self.output_rank >2 else jnp.expand_dims(y,axis=-1)]


    def add(self, d_in, p_in, y_in):
        """
        Append a new design/parameter/output batch, expanding
        each to the stored rank-3 shape.
        """
        input_rank = len(d_in.shape)
        p_rank = len(p_in.shape)
        output_rank = len(y_in.shape)

        d_in = d_in if input_rank > 1 else jnp.expand_dims(d,axis=-1)
        p_in = p_in if p_rank > 1 else jnp.expand_dims(p,axis=-1)
        y_in = y_in if output_rank >1 else jnp.expand_dims(y,axis=-1)

        input_rank = len(d_in.shape)
        p_rank = len(p_in.shape)
        output_rank = len(y_in.shape)


        self.d.append(d_in if input_rank > 2 else jnp.expand_dims(d_in,axis=-1))
        self.p.append(p_in if input_rank > 2 else jnp.expand_dims(p_in,axis=-1))
        self.y.append(y_in if output_rank >2 else jnp.expand_dims(y_in,axis=-1))
        return


class TrainingDataset(ABC):
    """Minimal input/output container that ensures at least 2-D arrays.

    Wraps the supplied X and y, expanding their last axis where needed so
    downstream consumers always receive rank-2 matrices.

    """

    # ---- External Methods ----

    def __init__(self, X, y):
        self.input_rank = len(X.shape)
        self.output_rank = len(y.shape)
        self.X = X if self.input_rank >= 2 else jnp.expand_dims(X,axis=-1)
        self.y = y if self.output_rank >=2 else jnp.expand_dims(y, axis=-1)


class DataProcessing(ABC):
    """Iterates and reshapes a DatasetObject into model-ready matrices.

    Optionally slices the stored batches from an index and exposes them either
    as a generator or flattened into stacked input-output matrices for training.

    """

    # ---- External Methods ----

    def __init__(self, dataset_object, index_on=None):
        self.dataset_object = dataset_object
        self.index_on = index_on
        if index_on is None:
            self.d = dataset_object.d
            self.p = dataset_object.p
            self.y = dataset_object.y
        else:
            self.d = dataset_object.d[index_on:]
            self.p = dataset_object.p[index_on:]
            self.y = dataset_object.y[index_on:]

        self.check_data()

    def check_data(self):
        """
        Assert the design, parameter and output batch lists are equal length.
        """
        n_d = len(self.d)
        assert n_d == len(self.p) == len(self.y), "The number of design, parameters and outputs objects within the data object should be the same, currently {} n_d, {} n_p and {} n_o".format(n_d, len(self.p), len(self.y))

    def get_data(self):
        """
        Yield each stored (d, p, y) batch in turn.
        """
        for i in range(len(self.d)):
            yield self.d[i], self.p[i], self.y[i]

    def transform_data_to_matrix(self, edge_fn, feasible_indices=None, index_on=None):
        """
        Flatten the stored batches into stacked input-output matrices,
        useful for training neural networks or other input-output models.
        """
        data = self.get_data()
        data_store_X, data_store_Y, data_store_Z = [], [], []
        if feasible_indices is not None: data = zip(data, feasible_indices)
        else: data = zip(data)
        for index, data_zip in enumerate(data):
            # skip the data collected in sampling iterations until index_on
            if index_on is not None:
                if index<index_on:
                    pass
            # unpack feasible indices if provided
            if feasible_indices is not None:
                (d, p, y), fe_ind = data_zip
            else:
                d, p, y = data_zip[0]
            X, Y, Z = [], [], []
            if p.ndim<2: p=p.reshape(1,-1)
            if d.ndim<2: d=d.reshape(1,-1)
            if p.ndim < 3: p = jnp.expand_dims(p, axis=1)
            if d.ndim < 3: d = jnp.expand_dims(d, axis=1)

            y_edge = edge_fn(y)
            if y_edge.ndim < 3: y_edge = jnp.expand_dims(y_edge, axis=-1)

            # build input-output data per uncertain parameter (vectorised model evaluations)
            for i in range(p.shape[0]):
                X.append(jnp.hstack([d.reshape((d.shape[0], d.shape[1])), jnp.repeat(p[i].reshape(1,-1),d.shape[0], axis=0)]))
                Y.append(y_edge[:,i,:].reshape(d.shape[0],-1))
                if feasible_indices is not None:
                    # restrict to the feasible region for KS updates / feasible-only fits
                    Z.append(y_edge[:,i,:].reshape(d.shape[0],-1)[fe_ind.squeeze()])

            X = jnp.vstack(X)
            Y = jnp.vstack(Y)
            Z = jnp.vstack(Z) if feasible_indices is not None else None
            data_store_X.append(X)
            data_store_Y.append(Y)
            data_store_Z.append(Z)

        if feasible_indices is not None:
            return jnp.vstack(data_store_X), jnp.vstack(data_store_Y), jnp.vstack(data_store_Z)
        else:
            return jnp.vstack(data_store_X), jnp.vstack(data_store_Y), None


class FeasibilityBase(ABC):
    """Base class binding a feasibility rule to a dataset.

    Selects either the probabilistic or deterministic feasibility method based
    on the requested formulation; subclasses implement the concrete evaluation.

    """

    # ---- External Methods ----

    def __init__(self, dataset_X, dataset_Y, cfg, node, feasibility):
        self.dataset_X = dataset_X
        self.dataset_Y = dataset_Y
        self.node = node
        self.cfg = cfg
        if feasibility == 'probabilistic':
            self.feasible_function = self.probabilistic_feasibility
        elif feasibility == 'deterministic':
            self.feasible_function = self.deterministic_feasibility
        else:
            raise ValueError(f"Formulation {self.cfg.formulation} not recognised. Please use 'probabilistic' or 'deterministic'.")

    def probabilistic_feasibility(self, X, Y):
        """
        Stub for probabilistic feasibility, overridden by subclasses.
        """
        pass

    def deterministic_feasibility(self, X, Y):
        """
        Stub for deterministic feasibility, overridden by subclasses.
        """
        pass

class FeasibilityFunction:
    """Per-design weighted joint-feasibility probability over a discrete uncertainty set.

    For each design row it sums the scenario weights over the scenarios where
    every constraint is satisfied. Deliberately not inheriting FeasibilityBase,
    whose __init__ would shadow the method defined here.

    """

    # ---- External Methods ----

    def __init__(self, dataset_X, dataset_Y, cfg, node, feasibility=None):
        # dataset_Y carries the constraint tensor (N_batch, n_theta, n_g);
        # dataset_X and feasibility kept for signature parity but unused here.
        self.dataset_X = dataset_X
        self.dataset_Y = dataset_Y
        self.cfg = cfg
        self.node = node

    def feasible_function(self, probabilities):
        """
        Weighted joint-feasibility probability per design row.
        """
        Y = self.dataset_Y  # (N_batch, n_theta, n_g)

        if self.cfg.samplers.notion_of_feasibility == 'positive':
            feasibility = jnp.where(Y >= 0, 1.0, 0.0)
        else:
            feasibility = jnp.where(Y <= 0, 1.0, 0.0)

        all_feasible = feasibility.prod(axis=-1)  # AND across constraints -> (N_batch, n_theta)

        probabilities = jnp.asarray(probabilities).reshape(-1)
        assert all_feasible.shape[1] == probabilities.size, (
            f"Number of weights ({probabilities.size}) must match the scenario axis "
            f"({all_feasible.shape[1]})."
        )

        return (all_feasible * probabilities[jnp.newaxis, :]).sum(axis=-1)  # (N_batch,)


class ApplyFeasibility(FeasibilityBase):
    """Concrete feasibility evaluator over stored constraint batches.

    Implements the probabilistic and deterministic feasibility methods bound by
    FeasibilityBase, returning stacked inputs, labels and per-batch indicators.

    """

    # ---- External Methods ----

    def __init__(self, dataset_X, dataset_Y, cfg, node, feasibility):
        super().__init__(dataset_X, dataset_Y, cfg, node, feasibility)

    def get_feasible(self, return_indices=True):
        """
        Dispatch to the bound feasibility method, which already reads X, Y off self.
        """
        return self.feasible_function(return_indices=return_indices)

    def probabilistic_feasibility(self, return_indices=True):
        """
        Probabilistic feasibility: fraction of scenarios feasible per design.
        X is N x (n_d + n_u); Y is N x n_p x n_g.
        """
        X, Y = self.dataset_X, self.dataset_Y

        Y_s = []
        for x, y in zip(X, Y):

            if self.cfg.samplers.notion_of_feasibility == 'positive':
                y = jnp.min(y, axis=-1).reshape(y.shape[0],y.shape[1])
                indicator = jnp.where(y>=0, 1, 0)
            else:
                y = jnp.max(y, axis=-1).reshape(y.shape[0],y.shape[1])
                indicator = jnp.where(y<=0, 1, 0)

            n_s = y.shape[1]
            prob_feasible = jnp.sum(indicator, axis=1)/n_s
            Y_s.append(prob_feasible.reshape(y.shape[0],1))

        if not return_indices:
            return jnp.vstack(X), jnp.vstack(Y_s)
        else:
            return jnp.vstack(X), jnp.vstack(Y_s), [ys >= self.cfg.samplers.unit_wise_target_reliability[self.node] for ys in Y_s]

    def deterministic_feasibility(self, return_indices=True):
        """
        Deterministic feasibility: worst-case constraint label per design.
        """
        X, Y = self.dataset_X, self.dataset_Y
        cond = []
        labels = []
        for x, y in zip(X, Y):
            if self.cfg.samplers.notion_of_feasibility == 'positive':
                lab = jnp.min(y, axis=-1)
                select_cond = lab  >= 0
            else:
                lab = jnp.max(y, axis=-1)
                select_cond = lab  <= 0

            cond.append(select_cond)
            labels.append(lab)


        if not return_indices:
            return jnp.vstack(X), jnp.vstack(labels)
        else:
            return  jnp.vstack(X), jnp.vstack(labels), cond


def check_requires(fn, param_name: str) -> bool:
    """True if `fn` can accept `param_name` (explicitly or via **kwargs)."""
    sig = inspect.signature(fn)
    params = sig.parameters

    if param_name in params:
        return True

    # accepts the param if it has **kwargs
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
