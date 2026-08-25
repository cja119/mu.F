"""Training-data shaping and node-bound updates for the decomposition."""
import numpy as np
import jax.numpy as jnp

from mu_F.samplers.appproximators import calculate_box_outer_approximation
from mu_F.utils import TrainingDataset
from mu_F.utils import DataProcessing as data_processor
from mu_F.utils import ApplyFeasibility


def del_data(graph, node):
    """
    Clear a node's classifier and successor-edge surrogate training data.
    """
    # classifier_training is absent under direct-probability (no classifier).
    graph.nodes[node]["classifier_training"] = None
    graph.nodes[node]["margin_training"] = None

    for successor in graph.successors(node):
        if 'surrogate_training' in graph.edges[node, successor]:
            del graph.edges[node, successor]["surrogate_training"]
            graph.edges[node, successor]["surrogate_training"] = None
    return graph


def update_node_bounds_iplus1(graph, node, cfg):
    """
    Set the node's iterate i+1 bounds from its iterate i live set.
    """
    new_bounds = calculate_box_outer_approximation(graph.nodes[node]["live_set_inner"], cfg, ndim=2)
    graph.nodes[node]['extendedDS_bounds'] = new_bounds

    return


def _feasibility_training_data(cfg, graph, node, model):
    """
    Write the node's feasibility and ctg surrogate targets onto the graph,
    returning feasible-design indices in the deterministic case else None.
    """
    store = model.evaluator.store
    on = cfg.surrogate.index_on
    feasible_indices = None

    if store.has('pmap'):                              # probabilistic: P_feas target
        X, y = store.matrix('pmap')
        graph.nodes[node]["probability_map_training"] = TrainingDataset(
            X=jnp.asarray(X[on:]), y=jnp.asarray(y[on:]))
    elif store.has('classifier'):                      # deterministic: per-scenario g
        x_d, y_d = store.matrix('classifier')
        # ApplyFeasibility iterates over a list of batches -> wrap the sliced arrays
        xc, yc, feasible_indices = ApplyFeasibility(
            [x_d[on:]], [y_d[on:]], cfg, node, cfg.formulation).get_feasible(return_indices=True)
        graph.nodes[node]["classifier_training"] = TrainingDataset(X=xc, y=yc)

    if store.has('classifier'):                        # same values, unfiltered, for the regressor
        x_m, y_m = store.matrix('classifier')
        y_m = np.asarray(y_m)
        if y_m.ndim == 3:                              # (N, n_theta, G) -> worst case over scenarios
            y_m = y_m.min(axis=1)
        x_m, y_m = np.asarray(x_m)[on:], y_m[on:]
        finite = np.isfinite(y_m).all(axis=-1)         # drop non-viable (failed-integration) rows
        graph.nodes[node]["margin_training"] = TrainingDataset(
            X=jnp.asarray(x_m[finite]), y=jnp.asarray(y_m[finite]))

    if cfg.case_study.eval_cost and store.has('ctg'):  # E_p[ctg]; drop NaN (non-viable) rows
        Xc, yc = store.matrix('ctg')
        Xc, yc = Xc[on:], yc[on:]
        finite = np.isfinite(yc).all(axis=-1)
        graph.nodes[node]["ctg_func_training"] = TrainingDataset(X=jnp.asarray(Xc[finite]), y=jnp.asarray(yc[finite]))

    return feasible_indices


def _forward_surrogate_training_data(cfg, graph, node, model, feasible_indices):
    """
    Build, per successor edge, the input->output surrogate training set and
    the box bounds on the coupling inputs.
    """
    if cfg.samplers.ku_approximation != 'box':
        raise NotImplementedError(f"{cfg.samplers.ku_approximation} approximation not implemented.")
    for successor in graph.successors(node):
        io_fn = graph.edges[node, successor]["edge_fn"]
        x_io, y_io, selected_y_io = data_processor(
            model.evaluator.io_data, index_on=cfg.surrogate.index_on
        ).transform_data_to_matrix(io_fn, feasible_indices)

        if cfg.formulation == 'deterministic':         # drop the scenario columns, keep theta
            n = graph.nodes[node]
            keep = (n['n_design_args'] + n['n_input_args'] + n['n_auxiliary_args']
                    + int(graph.graph.get('n_theta_input', 0)))
            x_io = x_io[:, :keep]
        if y_io.ndim > 2:          y_io = y_io.squeeze()
        if selected_y_io.ndim < 2: selected_y_io = selected_y_io.reshape(-1, 1)

        graph.edges[node, successor]["input_data_bounds"] = \
            calculate_box_outer_approximation(selected_y_io, cfg, ndim=2)
        graph.edges[node, successor]["surrogate_training"] = TrainingDataset(X=x_io, y=y_io)
        del x_io, y_io, selected_y_io


def process_data_forward(cfg, graph, node, model, live_set, mode):
    """
    Collect this node's surrogate training data and refresh its bounds;
    the training step is skipped under the 'backward-forward' tuner pass.
    """
    if mode != 'backward-forward':
        feasible_indices = _feasibility_training_data(cfg, graph, node, model)
        if cfg.surrogate.forward_evaluation_surrogate:
            _forward_surrogate_training_data(cfg, graph, node, model, feasible_indices)

    graph.nodes[node]["live_set_inner"] = live_set
    update_node_bounds_iplus1(graph, node, cfg)
