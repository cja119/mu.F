"""Surrogate construction for the per-node decomposition pipeline."""

import logging

import numpy as np
import jax
import jax.numpy as jnp

from mu_F.surrogate.surrogate import Surrogate
from mu_F.utils import TrainingDataset


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
    held_out = _hold_out_batch(cfg, graph, node)        # None unless the backoff is on
    _train_node_classifier(cfg, graph, node, iterate)
    if held_out is not None:
        graph.nodes[node]['constraint_backoff'] = _calibrate_backoff(cfg, graph, node, held_out)
        graph.nodes[node]['classifier_training'] = held_out['full']


def _train_node_classifier(cfg, graph, node, iterate):
    """Fit the feasibility classifier from the node's current training data and
    store its callable + serialised form. Read by classifier_construction."""
    trained = Surrogate(graph, node, cfg,
                        ('classification', cfg.surrogate.classifier_selection, 'live_set_surrogate'),
                        iterate, data_str='classifier_training')
    trained.fit(node=node)
    query_model, serialised = _query_model(trained, cfg, cfg.surrogate.classifier_selection, _classifier_head(cfg))

    graph.nodes[node]["classifier"] = query_model
    graph.nodes[node]['classifier_x_scalar'] = trained.trainer.get_model_object('standardisation_metrics_input')
    graph.nodes[node]['classifier_serialised'] = serialised
    del trained


def _classifier_head(cfg):
    """Logit-margin head when the backoff is on (unsaturated, so the backoff can
    separate near-boundary false positives the softmax score collapses); the
    softmax score otherwise. ANN only. Read by _train_node_classifier."""
    bk = cfg.surrogate.get('backoff', None)
    on = bool(bk.get('enabled', False)) if bk else False
    return 'classifier_logit' if (on and cfg.surrogate.classifier_selection == 'ANN') else 'classifier'


def _hold_out_batch(cfg, graph, node):
    """Hold a batch of the original sampling data out of training, for the
    a-posteriori backoff calibration; None unless the backoff is enabled."""
    bk = cfg.surrogate.get('backoff', None)
    if bk is None or not bk.get('enabled', False):
        return None

    full = graph.nodes[node]['classifier_training']
    n = int(full.X.shape[0])
    n_hold = max(1, int(float(bk.get('held_out_frac', 0.2)) * n))
    perm = np.random.default_rng(0).permutation(n)
    hold_idx, train_idx = perm[:n_hold], perm[n_hold:]
    graph.nodes[node]['classifier_training'] = TrainingDataset(full.X[train_idx], full.y[train_idx])
    return {'full': full, 'X': full.X[hold_idx], 'y': full.y[hold_idx]}


def _calibrate_backoff(cfg, graph, node, held_out):
    """Minimal a-posteriori backoff: the smallest margin that excludes all but a
    fraction epsilon of the held-out false positives (infeasible points the
    classifier still admits). Read by classifier_construction."""
    eps = float(cfg.surrogate.backoff.get('epsilon', 0.05))
    classifier = graph.nodes[node]['classifier']

    if cfg.samplers.notion_of_feasibility == 'positive':
        infeasible = np.asarray(jnp.min(held_out['y'], axis=1) < 0).reshape(-1)
    else:
        infeasible = np.asarray(jnp.max(held_out['y'], axis=1) > 0).reshape(-1)

    margins = np.asarray(jax.vmap(classifier)(jnp.asarray(held_out['X']))).reshape(-1)
    false_positive = margins[infeasible & (margins <= 0)]            # admitted infeasibles
    if false_positive.shape[0] == 0:
        return jnp.asarray(0.0)

    backoff = float(max(0.0, -np.quantile(false_positive, eps)))
    kept = float(np.mean(margins[~infeasible] <= -backoff)) if (~infeasible).any() else 1.0
    logging.info(f"Node {node}: held-out backoff {backoff:.3f} "
                 f"({false_positive.shape[0]} false positives, eps={eps}, "
                 f"feasible region kept {kept:.1%})")
    return jnp.asarray(backoff)


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

    held_out = _hold_out_batch(cfg, graph, node)        # None unless the backoff is on
    ds = graph.nodes[node]['classifier_training']

    X = np.asarray(ds.X)
    if X.ndim > 2:
        X = X.squeeze()
    y = np.asarray(ds.y).reshape(-1)

    feas_mask = (y >= 0)
    X_feas = X[feas_mask]

    if X_feas.shape[0] < 2:
        logging.warning(f"Cluster classifier: too few feasible points at node {node}")
        _restore_full_training(graph, node, held_out)
        return

    from mu_F.surrogate.augmentation import xmeans_cluster_indices
    cluster_indices = xmeans_cluster_indices(X_feas)
    n_clusters = len(cluster_indices)

    if n_clusters <= 1:
        logging.info(
            f"Cluster classifier: x-means K={n_clusters} on classifier_training "
            f"at node {node} — using single-head classifier."
        )
        _restore_full_training(graph, node, held_out)   # single-head path re-splits itself
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

    if held_out is not None:
        graph.nodes[node]['constraint_backoff'] = _calibrate_backoff_multihead(
            cfg, graph, node, held_out)
        graph.nodes[node]['classifier_training'] = held_out['full']


def _restore_full_training(graph, node, held_out):
    """Undo the hold-out split on early exits so no samples are lost.
    Read by cluster_classifier_construction."""
    if held_out is not None:
        graph.nodes[node]['classifier_training'] = held_out['full']


def _calibrate_backoff_multihead(cfg, graph, node, held_out):
    """Per-head a-posteriori backoff: for each head, the smallest margin that
    excludes all but a fraction epsilon of the held-out infeasible points that
    head still admits. Read by cluster_classifier_construction."""
    eps = float(cfg.surrogate.backoff.get('epsilon', 0.05))
    head = graph.nodes[node]['cluster_classifier_head']

    if cfg.samplers.notion_of_feasibility == 'positive':
        infeasible = np.asarray(jnp.min(held_out['y'], axis=1) < 0).reshape(-1)
    else:
        infeasible = np.asarray(jnp.max(held_out['y'], axis=1) > 0).reshape(-1)

    X = jnp.asarray(held_out['X'])
    if X.ndim > 2:
        X = X.squeeze()
    margins = np.asarray(jax.vmap(head)(X))                          # (n_hold, K)

    backoffs = []
    for k in range(margins.shape[1]):
        fp = margins[infeasible & (margins[:, k] <= 0), k]           # head-k admitted infeasibles
        backoffs.append(float(max(0.0, -np.quantile(fp, eps))) if fp.size else 0.0)

    # feasible points still admitted by some head under the per-head backoffs
    tightened = margins + np.asarray(backoffs)[None, :]
    kept = (float(np.mean(np.min(tightened[~infeasible], axis=1) <= 0.0))
            if (~infeasible).any() else 1.0)
    logging.info(f"Node {node}: held-out per-head backoff "
                 f"{np.round(backoffs, 3).tolist()} (eps={eps}, "
                 f"feasible region kept {kept:.1%})")
    return jnp.asarray(backoffs)
