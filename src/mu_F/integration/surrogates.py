"""Surrogate construction for the per-node decomposition pipeline."""

import logging

import numpy as np

from mu_F.surrogate.surrogate import Surrogate


def _query_model(trained, cfg, selection, head):
    """The graph-stored callable + its serialised form."""
    serialised = trained.get_serailised_model_data()
    if selection == 'ANN':
        from mu_F.surrogate.nn_utils import build_ann
        return build_ann(cfg, serialised, head), serialised
    return trained.get_model('unstandardised_model'), serialised


def surrogate_training_forward(cfg, graph, node, iterate: int = 0):
    """Train the forward-evaluation Surrogate (input -> output) on each successor
    edge."""
    if graph.out_degree()[node] == 0:
        return

    for successor in graph.successors(node):
        trained = Surrogate(graph, node, cfg,
                            ('regression', cfg.surrogate.regressor_selection, 'forward_evaluation_surrogate'),
                            iterate, data_str='None')   # forward data handling is data_str-independent
        trained.fit(node=successor)
        query_model = trained.get_model('unstandardised_model')

        graph.edges[node, successor]["forward_surrogate"] = query_model
        graph.nodes[node]['x_scalar'] = trained.trainer.get_model_object('standardisation_metrics_input')
        graph.edges[node, successor]['y_scalar'] = trained.trainer.get_model_object('standardisation_metrics_output')
        graph.edges[node, successor]["forward_surrogate_serialised"] = trained.get_serailised_model_data()

    return query_model


def probability_map_construction(cfg, graph, node, iterate):
    """Train the node's probability_map (the P_feas regressor) and store it.
    Always a sigmoid-headed ANN, kept as the unstandardised model so the head
    survives a graph reload."""
    trained = Surrogate(graph, node, cfg, ('regression', 'ANN', 'probability_map_surrogate'),
                        iterate, data_str='probability_map_training')
    trained.fit(node=None)
    query_model = trained.get_model('unstandardised_model')

    graph.nodes[node]["probability_map"] = query_model
    graph.nodes[node]['probability_map_x_scalar'] = trained.trainer.get_model_object('standardisation_metrics_input')
    graph.nodes[node]['probability_map_y_scalar'] = trained.trainer.get_model_object('standardisation_metrics_output')


def classifier_construction(cfg, graph, node, iterate):
    """Train the node's feasibility classifier and store it.
    Convention: g <= 0 ⇔ inside the feasible region."""
    trained = Surrogate(graph, node, cfg,
                        ('classification', cfg.surrogate.classifier_selection, 'live_set_surrogate'),
                        iterate, data_str='classifier_training')
    trained.fit(node=node)
    query_model, serialised = _query_model(trained, cfg, cfg.surrogate.classifier_selection, 'classifier')

    graph.nodes[node]["classifier"] = query_model
    graph.nodes[node]['classifier_x_scalar'] = trained.trainer.get_model_object('standardisation_metrics_input')
    graph.nodes[node]['classifier_serialised'] = serialised

    del trained


def ctg_surrogate_construction(cfg, graph, node, iterate):
    """Train the cost-to-go regressor and store it on the graph."""
    trained = Surrogate(graph, node, cfg, ('regression', cfg.surrogate.regressor_selection, 'ctg_surrogate'), iterate)
    trained.fit(node=None)
    query_model, serialised = _query_model(trained, cfg, cfg.surrogate.regressor_selection, 'regressor')

    graph.nodes[node]["ctg_surrogate"] = query_model
    # x_scalar / serialised kept for diagnostics; no evaluator reads them under the new contract.
    graph.nodes[node]['ctg_surrogate_x_scalar'] = trained.trainer.get_model_object('standardisation_metrics_input')
    graph.nodes[node]['ctg_surrogate_serialised'] = serialised

    del trained


def cluster_classifier_construction(cfg, graph, node, iterate):
    """Build a K-head ANN classifier from this node's training data."""

    from mu_F.surrogate.nn_utils import train_multihead_regressor

    pre_aug_K = graph.nodes[node].get('pre_augmentation_n_clusters', None)
    if pre_aug_K is not None and pre_aug_K <= 1:
        logging.info(
            f"Cluster classifier: pre-augmentation x-means K={pre_aug_K} at "
            f"node {node} — feasibility is connected; using single-head classifier."
        )
        classifier_construction(cfg, graph, node, iterate)
        return

    ds = graph.nodes[node].get('classifier_training', None)
    if ds is None:
        logging.warning(f"Cluster classifier: no classifier_training data on node {node}")
        return

    X = np.asarray(ds.X)
    if X.ndim > 2:
        X = X.squeeze()
    y = np.asarray(ds.y).reshape(-1)

    feas_mask = (y >= 0)
    X_feas = X[feas_mask]

    if X_feas.shape[0] < 2:
        logging.warning(f"Cluster classifier: too few feasible points at node {node}")
        return

    from mu_F.surrogate.augmentation import xmeans_cluster_indices
    cluster_indices = xmeans_cluster_indices(X_feas)
    n_clusters = len(cluster_indices)

    if n_clusters <= 1:
        logging.info(
            f"Cluster classifier: x-means K={n_clusters} on classifier_training "
            f"at node {node} — using single-head classifier."
        )
        classifier_construction(cfg, graph, node, iterate)
        return
    cluster_labels = np.empty(X_feas.shape[0], dtype=int)

    for k, ids in enumerate(cluster_indices):
        cluster_labels[ids] = k

    N = X.shape[0]
    Y = np.ones((N, n_clusters), dtype=np.float32)
    feas_idx = np.where(feas_mask)[0]
    for j, k in zip(feas_idx, cluster_labels):
        Y[j, k] = -1.0

    head_callable = train_multihead_regressor(
        cfg, X, Y, num_folds=int(cfg.surrogate.num_folds),
    )
    graph.nodes[node]['cluster_classifier_head']    = head_callable
    graph.nodes[node]['cluster_classifier_n_heads'] = n_clusters
    logging.info(f"Cluster classifier: trained {n_clusters}-head ANN at node {node}")
