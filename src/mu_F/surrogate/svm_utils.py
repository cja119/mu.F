"""SVM classifier training and JAX conversion utilities for the surrogate feasibility map."""
import pandas as pd

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator

from omegaconf import DictConfig, OmegaConf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, GroupKFold
from sklearn.cluster import KMeans

import jax.numpy as jnp
from jax import jit
from functools import partial

from mu_F.surrogate.data_utils import binary_classifier_data_preparation, StandardisationMetrics

import logging

@jit
def all_feasible(x):
    """Fallback decision function when every sample is feasible."""
    return jnp.linalg.norm(x*jnp.array([1e-4]), ord='fro') + jnp.array([-5])

@jit
def no_feasible(x):
    """Fallback decision function when no sample is feasible."""
    return jnp.linalg.norm(x*jnp.array([1e-4]), ord='fro') + jnp.array([5])

def get_serialised_model_data(model):
    """Delegate to the model's serialised-data accessor."""
    return model.get_serialised_model_data()

def train(cfg, dataset, num_folds, unit_index, iterate):
    """
    Construct the classifier for the forward pass, falling back to a
    constant decision function when the dataset is single-class.
    """
    data_points, labels = dataset
    # construct a classifier from the available data
    s_vectors, classifier, p_dict, labels = compute_best_svm_classifier(
        data_points, labels, unit_index=unit_index, iterate=iterate, cfg=cfg, num_folds=num_folds
    )

    if classifier is not None:
        _, args, model_data = convert_svm_to_jax(classifier.best_estimator_)
        return classifier, args, model_data
    else:
        if jnp.all(labels == -1):
            classifier = all_feasible
            model_data = get_model_data(
            jnp.zeros_like(data_points[0,:], dtype=jnp.float32), 
            jnp.ones_like(data_points[0,:], dtype=jnp.float32), 
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32), 
            jnp.array([-5.], dtype=jnp.float32), 
            1.0
        )
        elif jnp.all(labels == 1):
            classifier = no_feasible
            model_data = get_model_data(
            jnp.zeros_like(data_points[0,:], dtype=jnp.float32), 
            jnp.ones_like(data_points[0,:], dtype=jnp.float32), 
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32), 
            jnp.array([5.], dtype=jnp.float32), 
            1.0
        )
        args = (classifier, classifier, None)
        

        return classifier, args, model_data


def _spatial_groups(data_points, num_folds):
    """Spatial cross-validation groups for the SVM grid search.
    Clusters the support so GroupKFold holds out whole feature-space regions,
    defeating the boundary-density optimism that lets high-gamma fits pass.
    """
    n_clusters = max(5 * num_folds, 20)
    scaled = StandardScaler().fit_transform(data_points)
    return KMeans(n_clusters=n_clusters, n_init=3, random_state=0).fit_predict(scaled)


def compute_best_svm_classifier(
    data_points, labels, unit_index, cfg, iterate, num_folds
):
    """
    Grid-search an SVM classifier over the configured kernel parameters and
    report cross-validation performance.
    """
    # build classification model.  NuSVC parameterises the SV budget directly
    # (nu lower-bounds the SV fraction), so it caps the decision function the
    # backward sweep vmaps — plain SVC only controls it indirectly via C/gamma.
    x_r = sum(labels) / labels.shape[0]
    svm_cfg = cfg.surrogate.classifier_args.svm
    use_nu = str(svm_cfg.get('type', 'SVC')) == 'NuSVC'

    # Bias the boundary toward feasibility: weight the infeasible class (+1) so a
    # misclassified infeasible point (a phantom-feasible admission) costs more,
    # trading rejected-feasibles for fewer false positives.
    w = float(svm_cfg.get('infeasible_weight', 1.0))
    y_vals = jnp.asarray(labels).squeeze()
    class_weight = ({int(c): (w if c > 0 else 1.0) for c in jnp.unique(y_vals).tolist()}
                    if w != 1.0 else None)

    estimator = (svm.NuSVC(cache_size=2000, class_weight=class_weight) if use_nu
                 else svm.SVC(cache_size=2000, class_weight=class_weight))
    clf = Pipeline([("scaler", StandardScaler()), ("svc", estimator)])
    grid = {
        "svc__kernel": svm_cfg.kernel,
        "svc__gamma": svm_cfg.gamma,
        "svc__probability": svm_cfg.probability,
    }
    grid["svc__nu" if use_nu else "svc__C"] = svm_cfg.nu if use_nu else svm_cfg.C
    parameters = [grid]
    n_jobs = int(getattr(cfg, 'max_devices', 1))
    if data_points.ndim > 2:
        data_points = data_points.squeeze()

    # Hold out whole feature-space clusters; random k-fold on the boundary-dense live set can't penalise over-flexible fits.
    groups = _spatial_groups(data_points, num_folds)
    model = GridSearchCV(clf, parameters, scoring="balanced_accuracy", cv=GroupKFold(num_folds), n_jobs=n_jobs)
    try:
        model.fit(data_points, labels.squeeze(), groups=groups)
    except:
        # single-class dataset (labels -1 feasible, +1 infeasible): caller falls back to a constant map
        ls = jnp.asarray(labels).squeeze()
        if jnp.all(ls == -1):
            logging.info("classification model skipped — all samples are feasible")
        elif jnp.all(ls == 1):
            logging.info("classification model skipped — all samples are infeasible")
        else:
            logging.info("error in fitting classification model")

        return None, None, None, labels
    support_vectors = model.best_estimator_["svc"].support_vectors_
    accuracy = model.score(data_points, labels)
    tn, fp, fn, tp = confusion_matrix(model.predict(data_points), labels).ravel()
    # n_sv is what the backward sweep pays for; fp is the phantom-feasibility count
    training_performance = {"acc": accuracy, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                            "n_sv": int(support_vectors.shape[0]),
                            "params": model.best_params_}

    logging.info(f"training_performance: {training_performance}")


    df = pd.DataFrame(model.cv_results_)
    df = df.sort_values(by=["rank_test_score"])
    df.to_excel(f"svm_cv_results_{unit_index}_{iterate}.xlsx")

    return support_vectors, model, training_performance, labels


@jit
def rbf_kernel(x, y, epsilon=1e-3):
    """RBF kernel for the multivariate case, avoiding large dense matrices."""
    diff = x - y
    squared_diff = jnp.linalg.norm(diff, axis=1)**2
    return jnp.exp(-epsilon * squared_diff)

def get_x_scalar_from_pipeline(pipeline):
    """Return the input scaler stage of the fitted pipeline."""
    return pipeline[0]

def convert_svm_to_jax(pipeline):
    """
    Extract the fitted SVM parameters and return JAX decision functions in
    both standardised and unstandardised input spaces.
    """
    svm_model = pipeline[1]
    support_vectors = svm_model.support_vectors_
    y_data = pipeline.predict(support_vectors).reshape(-1,1)
    coefficients = svm_model.dual_coef_[0].reshape(1,-1)  # binary classifier: one row per support vector
    intercept = svm_model.intercept_
    kernel_param = svm_model.gamma

    x_scalar = pipeline[0]
    x_mean = jnp.array(x_scalar.mean_)
    x_std = jnp.array(x_scalar.scale_)

    @jit
    def svm_unstandardised(x):
        """Decision function taking inputs in the original data space."""
        x_ = (x.reshape(-1,) - x_mean.reshape(-1,)) / x_std.reshape(-1,)
        x_ = x_.reshape(1, -1)
        decision = jnp.dot(coefficients, rbf_kernel(support_vectors, x_,epsilon=kernel_param)) + intercept
        return decision.squeeze()

    @jit
    def svm_standardised(x):
        """Decision function taking already-standardised inputs (no batching)."""
        x = x.reshape(1, -1)
        decision = jnp.dot(coefficients, rbf_kernel(support_vectors, x,epsilon=kernel_param)) + intercept
        return decision.squeeze()

    model_data = get_model_data(x_mean, x_std, support_vectors, coefficients, intercept, kernel_param)

    return svm_model, (svm_standardised, svm_unstandardised, StandardisationMetrics(mean=x_mean, std=x_std)), model_data

def get_model_data(x_mean, x_std, support_vectors, coefficients, intercept, kernel_param):
    """Pack standardisation metrics and SVM parameters for serialisation."""
    model_data = {'standardisation_metrics_input': StandardisationMetrics(mean=x_mean, std=x_std),
                  'serialized_params': {'support_vectors': support_vectors,
                                        'coefficients': coefficients,
                                        'intercept': intercept,
                                        'kernel_param': kernel_param}}
    return model_data

@jit
def _svm_fn_standardised(x, sv, coef, intercept, kp):
    """Standalone standardised decision function (no batching support)."""
    x = x.reshape(1, -1)
    return (jnp.dot(coef, rbf_kernel(sv, x, epsilon=kp)) + intercept).reshape(-1, 1)

@jit
def _svm_fn_unstandardised(x, sv, coef, intercept, kp, x_mean, x_std):
    """Standalone decision function taking inputs in the original data space."""
    x_ = ((x.reshape(-1,) - x_mean.reshape(-1,)) / x_std.reshape(-1,)).reshape(1, -1)
    return (jnp.dot(coef, rbf_kernel(sv, x_, epsilon=kp)) + intercept).reshape(-1, 1)


def build_svm(cfg, model_data):
    """Reconstruct the unstandardised SVM decision function from serialised data."""
    x_standardisation = model_data['standardisation_metrics_input']
    x_mean = x_standardisation.mean
    x_std = x_standardisation.std

    support_vectors = model_data['serialized_params']['support_vectors']
    coefficients = model_data['serialized_params']['coefficients']
    intercept = model_data['serialized_params']['intercept']
    kernel_param = model_data['serialized_params']['kernel_param']

    return partial(
        _svm_fn_unstandardised,
        sv=support_vectors, coef=coefficients, intercept=intercept,
        kp=kernel_param, x_mean=x_mean, x_std=x_std,
    )
