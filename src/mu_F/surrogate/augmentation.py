"""Local density augmentation of training data around the feasible region."""
from __future__ import annotations

import logging
from typing import Protocol, Tuple

import jax.numpy as jnp
import numpy as np
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Clustering primitive
# ---------------------------------------------------------------------------

def xmeans_cluster_indices(X_feas):
    """
    Cluster X_feas (N, D) with DEUS's BIC-driven x-means.
    Returns a list of K index arrays, one per cluster; K is chosen by x-means
    via the bic_ellipsoid criterion. Shared with classifier construction.
    """
    from deus.activities.solvers.algorithms.primitive import XMeans
    X = np.asarray(X_feas, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    lo = X.min(axis=0) - 1.0e-6
    hi = X.max(axis=0) + 1.0e-6
    bounds = np.stack([lo, hi], axis=1)
    xm = XMeans({"metric": "euclidean", "score_criterion": "bic_ellipsoid"})
    xm.set_bounds(bounds)
    xm.set_points(X)
    xm.normalize_points()
    xm.cluster_points()
    return [np.asarray(c, dtype=int) for c in xm.get_clusters_as_indices()]


# ---------------------------------------------------------------------------
# Strategy protocol and concrete implementations
# ---------------------------------------------------------------------------

class AugmentationStrategy(Protocol):
    """Generate extra training samples around a feasible region.

    Implementations return (aug_samples, log_descriptor): the raw samples in
    design+input+aux space and a short phrase for the dispatcher's log line.

    """
    def augment(self, X_feas, cfg, graph, node,
                *, n_samples: int, radius: float) -> Tuple[np.ndarray, str]: ...


class ClusterGaussianStrategy:
    """Gaussian draws around each x-means cluster centre.

    For each cluster, draw n_samples Gaussian points (scaled by radius) in
    standardised coordinates and invert the standardisation; stack across
    clusters. Used when there are no integer design dims to stratify by.

    """

    # ---- External Methods ----

    def augment(self, X_feas, cfg, graph, node,
                *, n_samples: int, radius: float):
        """
        Draw Gaussian augmentation samples around each x-means cluster centre.
        Called by the dispatcher when no integer design dims are present.
        """
        cluster_indices = xmeans_cluster_indices(X_feas)
        n_clusters = len(cluster_indices)

        scaler = StandardScaler().fit(X_feas)
        X_std = scaler.transform(X_feas)

        rng = np.random.default_rng(0)
        aug_std_blocks = []
        for cluster_ids in cluster_indices:
            members = X_std[cluster_ids]
            if members.shape[0] < 2:
                continue
            centre = members.mean(axis=0)
            cov = (np.cov(members, rowvar=False)
                   + 1.0e-6 * np.eye(members.shape[1]))
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                L = np.eye(members.shape[1]) * np.sqrt(np.diag(cov).mean())
            z = rng.standard_normal((n_samples, members.shape[1]))
            aug_std_blocks.append(centre + radius * (z @ L.T))

        if not aug_std_blocks:
            return np.empty((0, X_feas.shape[1]), dtype=np.float32), ""

        aug_std = np.concatenate(aug_std_blocks, axis=0)
        aug = scaler.inverse_transform(aug_std).astype(np.float32)
        log_desc = f"{n_samples}/cluster × {n_clusters} clusters [x-means K={n_clusters}]"
        return aug, log_desc


class IntegerCornerGaussianStrategy:
    """Per-integer-combo Gaussian around the nearest feasible cluster.

    For each integer combo, draw n_samples Gaussian samples around the
    continuous distribution of the nearest cluster, pinning the integer dims
    to the combo's values; densifies the corners of the integer grid.

    """

    # ---- External Methods ----

    def augment(self, X_feas, cfg, graph, node,
                *, n_samples: int, radius: float):
        """
        Draw augmentation samples per integer combo around its nearest cluster.
        Called by the dispatcher when integer design dims are present.
        """
        from mu_F.solvers.integer_nlp import IntegerProblem

        problem = IntegerProblem.from_cfg(
            design_domain=cfg.case_study.get('design_domain', []),
            n_heads=0,                          # head dim doesn't enter design vector
        )
        int_indices = np.array([d.slot for d in problem.design_dims], dtype=int)
        if int_indices.size == 0:
            # No integer dims: defer to the cluster-Gaussian strategy.
            return ClusterGaussianStrategy().augment(
                X_feas, cfg, graph, node, n_samples=n_samples, radius=radius,
            )

        int_table = np.asarray(problem.feasible_assignments())   # (n_combo, n_int)
        cluster_indices = xmeans_cluster_indices(X_feas)
        n_clusters = len(cluster_indices)

        n_full = X_feas.shape[1]
        cont_indices = np.setdiff1d(np.arange(n_full), int_indices).astype(int)

        # Integer-dim mean per cluster, used to pick the nearest cluster.
        cluster_int_centres = np.stack([
            X_feas[cluster_ids][:, int_indices].mean(axis=0)
            for cluster_ids in cluster_indices
        ], axis=0)                              # (n_clusters, n_int)

        rng = np.random.default_rng(0)
        aug_blocks = []
        for c in range(int_table.shape[0]):
            int_vals_c = int_table[c]
            dist = np.linalg.norm(
                cluster_int_centres - int_vals_c[None, :], axis=1,
            )
            k_best = int(np.argmin(dist))
            members = X_feas[cluster_indices[k_best]]
            if members.shape[0] < 2:
                continue
            cont = members[:, cont_indices]
            cont_mean = cont.mean(axis=0)
            cont_cov = (np.cov(cont, rowvar=False)
                        + 1.0e-6 * np.eye(cont_indices.size))
            try:
                L = np.linalg.cholesky(cont_cov)
            except np.linalg.LinAlgError:
                L = np.eye(cont_indices.size) * np.sqrt(np.diag(cont_cov).mean())
            z = rng.standard_normal((n_samples, cont_indices.size))
            cont_samples = cont_mean + radius * (z @ L.T)

            block = np.zeros((n_samples, n_full), dtype=np.float32)
            block[:, cont_indices] = cont_samples
            block[:, int_indices] = int_vals_c[None, :]
            aug_blocks.append(block)

        if not aug_blocks:
            return np.empty((0, n_full), dtype=np.float32), ""

        aug = np.concatenate(aug_blocks, axis=0).astype(np.float32)
        log_desc = (f"{n_samples}/combo × {int_table.shape[0]} integer combos "
                    f"[x-means K={n_clusters}]")
        return aug, log_desc


_STRATEGY_REGISTRY = {
    'cluster_gaussian':         ClusterGaussianStrategy(),
    'integer_corner_gaussian':  IntegerCornerGaussianStrategy(),
}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _resolve_augmentation_cfg(cfg) -> dict:
    """
    Resolve augmentation parameters from cfg.surrogate.augmentation, falling
    back to the legacy cfg.samplers.ns knobs for back-compat.
    """
    aug_cfg = cfg.surrogate.get('augmentation', None) if hasattr(cfg, 'surrogate') else None
    if aug_cfg is not None:
        return {
            'enabled':   bool(aug_cfg.get('enabled', True)),
            'strategy':  str(aug_cfg.get('strategy', 'integer_corner_gaussian')),
            'n_samples': int(aug_cfg.get('n_samples', 500)),
            'radius':    float(aug_cfg.get('radius', 3.0)),
        }
    n_samples = int(cfg.samplers.ns.get('n_anneal', 0))
    return {
        'enabled':   n_samples > 0,
        'strategy':  'integer_corner_gaussian',
        'n_samples': n_samples,
        'radius':    float(cfg.samplers.ns.get('anneal_radius', 2.0)),
    }


def _extract_feasibles(model) -> np.ndarray:
    """
    Pull DEUS-evaluated samples and feasibility mask from model.constraint_data.
    A point is feasible when its min over (theta, g) margin is non-negative;
    returns the feasible rows and the full sample set.
    """
    X = jnp.vstack(model.constraint_data.d)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    Y = jnp.vstack(model.constraint_data.y)
    if Y.ndim == 3:
        y_min = jnp.min(Y, axis=(1, 2))
    else:
        y_min = jnp.min(Y.reshape(Y.shape[0], -1), axis=-1)
    feas_mask = np.asarray(y_min >= 0)
    return np.asarray(X)[feas_mask], np.asarray(X)


def augment_training_data(cfg, graph, node, model) -> None:
    """
    Grow model.constraint_data and model.ctg_values via the configured
    strategy. Called after DEUS finishes a node and before
    process_data_forward, so the samples land in both training pools.
    """
    aug_cfg = _resolve_augmentation_cfg(cfg)
    if not aug_cfg['enabled']:
        return
    if model.constraint_data is None:
        logging.warning(f"Augmentation node {node}: no constraint_data on model — skipping")
        return

    X_feas, X_all = _extract_feasibles(model)
    if X_feas.shape[0] < 4:
        logging.warning(
            f"Augmentation node {node}: only {X_feas.shape[0]} feasibles — skipping"
        )
        return

    # Cache the pre-augmentation x-means K so cluster_classifier_construction
    # judges multi-head on natural geometry, not the post-augmentation K.
    graph.nodes[node]['pre_augmentation_n_clusters'] = len(
        xmeans_cluster_indices(X_feas)
    )

    strategy = _STRATEGY_REGISTRY.get(aug_cfg['strategy'])
    if strategy is None:
        raise ValueError(
            f"Unknown augmentation strategy: {aug_cfg['strategy']!r}. "
            f"Available: {sorted(_STRATEGY_REGISTRY)}"
        )

    aug, log_desc = strategy.augment(
        X_feas, cfg, graph, node,
        n_samples=aug_cfg['n_samples'], radius=aug_cfg['radius'],
    )
    if aug.shape[0] == 0:
        return

    # Clip to the box spanned by every DEUS-evaluated sample so far, honouring
    # the sampling region the surrogates were trained on.
    lo = X_all.min(axis=0)
    hi = X_all.max(axis=0)
    aug = np.clip(aug, lo, hi).astype(np.float32)

    # Build the uncertain-parameter vector the same way DEUS does, so model.s()
    # evaluates the augmented samples under the same parametric distribution.
    if cfg.formulation == 'deterministic':
        p = jnp.array(graph.nodes[node]['parameters_best_estimate']).reshape(1, -1)
    else:
        p_list = graph.nodes[node]['parameters_samples']
        p = jnp.array([s['c'] for s in p_list])

    # model.s() appends to model.constraint_data and (if eval_cost)
    # model.ctg_values; that side effect is the whole point of this call.
    _ = model.s(jnp.asarray(aug, dtype=jnp.float32), p)
    logging.info(
        f"Augmentation node {node}: added {aug.shape[0]} samples "
        f"({log_desc}, radius={aug_cfg['radius']}, strategy={aug_cfg['strategy']})"
    )
