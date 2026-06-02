"""Neural-network surrogate training, serialisation and inference utilities."""
from typing import Dict
from itertools import product
from abc import ABC
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax import jit, lax


from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from flax import linen as nn
from flax import struct
from flax.training import train_state
from jax.nn import softmax
from flax.serialization import to_bytes, from_bytes
from flax import jax_utils
import optax
from sklearn.metrics import confusion_matrix

import logging
from functools import partial
from omegaconf import DictConfig
from functools import partial

from mu_F.surrogate.data_utils import StandardisationMetrics


class Dataset(ABC):
    """Lightweight (X, y) container that enforces at least 2D arrays.

    Wraps training inputs and targets, expanding rank-1 arrays so downstream
    NN code can assume a trailing feature/output dimension.

    """

    # ---- External Methods ----

    def __init__(self, X, y):
        self.input_rank = len(X.shape)
        self.output_rank = len(y.shape)
        self.X = X if self.input_rank >= 2 else jnp.expand_dims(X, axis=-1)
        self.y = y if self.output_rank >=2 else jnp.expand_dims(y,axis=-1)


def resample_inverse_density(D, n_bins: int = 20, seed: int = 0):
    """
    Resample designs (with replacement) so the target is flat across n_bins:
    each point is drawn with probability inverse to its target-value density,
    restoring the rare high-P_feas region a plain-MSE fit washes out.
    """
    y = np.asarray(D.y).reshape(-1)
    edges = np.linspace(y.min(), y.max() + 1e-9, n_bins + 1)
    b = np.clip(np.digitize(y, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(b, minlength=n_bins).astype(float)
    w = 1.0 / counts[b]
    w /= w.sum()
    sel = np.random.default_rng(seed).choice(y.shape[0], size=y.shape[0], replace=True, p=w)
    return Dataset(jnp.asarray(D.X)[sel], jnp.asarray(D.y)[sel])


# ---------------------------------------------------------------------------
# Neural network regressor
# ---------------------------------------------------------------------------

def identify_neural_network(hidden_units, output_units, activation_functions, output_activation='identity') -> nn.Module:
    """Build a NeuralNetworkEstimator from architecture arguments."""
    return NeuralNetworkEstimator(hidden_units=hidden_units, output_units=output_units,
                                  activation_functions=activation_functions,
                                  output_activation=output_activation)

def serialise_model(params, model, x_scalar, y_scalar, model_type, model_data):
    """
    Serialise model params, architecture and standardisation metrics into a
    dict, verifying the params round-trip through (de)serialisation.
    """
    model_data['hidden_units'] = model.hidden_units
    model_data['output_units'] = model.output_units
    model_data['activation_function'] = model.activation_functions
    model_data['output_activation'] = getattr(model, 'output_activation', 'identity')
    model_data['serialized_params'] = to_bytes(params)
    model_data['standardisation_metrics_input'] = StandardisationMetrics(x_scalar.mean_, x_scalar.scale_)
    if model_type == 'regressor':
        model_data['standardisation_metrics_output'] = StandardisationMetrics(y_scalar.mean_, y_scalar.scale_)

    # recursively compare original and deserialised parameters
    def compare_params(original, deserialized):
        for k, v in original.items():
            if isinstance(v, dict):
                compare_params(v, deserialized[k])
            else:
                assert jnp.allclose(v, deserialized[k]), f"Mismatch found in parameter {k}"

    try:
        compare_params(params, from_bytes(params, model_data['serialized_params']))
        print("Serialization and deserialization successful. Parameters match!")
    except AssertionError as e:
        print(f"Error: {e}")
    return model_data


def train_multihead_regressor(cfg, X, Y_kvec, num_folds: int = 2,
                              rng_key=jax.random.PRNGKey(0)):
    """
    Train a shared-trunk ANN with K linear heads on +/-1 cluster targets,
    using per-head class-balanced MSE. Returns a jitted f(x) -> (K,) whose
    k-th entry <= 0 means "predicted in cluster k".
    """
    X_arr = jnp.asarray(X, dtype=jnp.float32)
    if X_arr.ndim > 2:
        X_arr = X_arr.squeeze()
    Y_arr = jnp.asarray(Y_kvec, dtype=jnp.float32)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)

    n_heads = int(Y_arr.shape[1])
    n_total = int(Y_arr.shape[0])

    # standardise X; +/-1 targets need no scaling
    x_scalar = StandardScaler().fit(np.asarray(X_arr))
    X_std = jnp.asarray(x_scalar.transform(np.asarray(X_arr)), dtype=jnp.float32)

    # per-head class-balanced weights: each (i, k) weight = 0.5 / class count in head k
    n_minus = jnp.sum(Y_arr == -1, axis=0)          # (K,)
    n_plus  = jnp.sum(Y_arr ==  1, axis=0)          # (K,)
    w_minus = 0.5 / jnp.maximum(n_minus, 1).astype(jnp.float32)
    w_plus  = 0.5 / jnp.maximum(n_plus,  1).astype(jnp.float32)
    W = jnp.where(Y_arr == -1, w_minus[None, :], w_plus[None, :])  # (N, K)

    def balanced_loss(params, model, x, y, w):
        y_pred = model.apply(params, x)
        return jnp.sum(w * jnp.square(y - y_pred))

    ann_cfg = cfg.surrogate.classifier_args.ann
    hidden_sizes = list(ann_cfg.hidden_size_options)
    afs = list(ann_cfg.activation_functions)
    num_epochs = int(ann_cfg.num_epochs)
    lr = float(ann_cfg.learning_rate)
    weight_decay = float(getattr(ann_cfg, 'weight_decay', 0.0))

    def _fit_once(model, train_idx, val_idx):
        X_tr, Y_tr, W_tr = X_std[train_idx], Y_arr[train_idx], W[train_idx]
        X_va, Y_va, W_va = X_std[val_idx],   Y_arr[val_idx],   W[val_idx]
        params = model.init(rng_key, X_tr[:1])
        tx = optax.adamw(lr, weight_decay=weight_decay)
        opt_state = tx.init(params)
        grad_fn = jax.value_and_grad(balanced_loss)

        @jit
        def step(params, opt_state):
            loss, grads = grad_fn(params, model, X_tr, Y_tr, W_tr)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for _ in range(num_epochs):
            params, opt_state, _ = step(params, opt_state)
        val_loss = float(balanced_loss(params, model, X_va, Y_va, W_va))
        return params, val_loss

    # CV hyperparameter search over (hidden_size, activation)
    kf = KFold(n_splits=max(num_folds, 2), shuffle=True, random_state=0)
    splits = list(kf.split(np.arange(n_total)))

    best_avg_loss = float('inf')
    best_hidden, best_af = hidden_sizes[0], afs[0]
    best_model, best_params = None, None
    for hidden_size, af in product(hidden_sizes, afs):
        model = identify_neural_network(hidden_size, n_heads, af)
        fold_results = [_fit_once(model, tr, va) for tr, va in splits]
        avg = sum(loss for _, loss in fold_results) / len(fold_results)
        if avg < best_avg_loss:
            best_avg_loss = avg
            best_hidden, best_af = hidden_size, af
            # keep params from the lowest-val_loss fold for this config
            best_params, _ = min(fold_results, key=lambda r: r[1])
            best_model = model

    x_mean = jnp.array(x_scalar.mean_)
    x_std  = jnp.array(x_scalar.scale_)
    apply_fn = best_model.apply

    @jit
    def query_unstandardised(x):
        if x.ndim < 2:
            x = x.reshape(1, -1)
        x_s = (x - x_mean) / x_std
        out = apply_fn(best_params, x_s)
        return out.reshape(-1)   # (K,)

    logging.info(
        f"multihead classifier: K={n_heads} hidden={best_hidden} "
        f"af={best_af} best_val_loss={float(best_avg_loss):.4g} "
        f"per-head counts: n_minus={list(map(int, n_minus))} "
        f"n_plus={list(map(int, n_plus))}"
    )
    return query_unstandardised


def check_dims(D):
    """Coerce a Dataset's X and y to 2D arrays in place."""
    x, y = D.X, D.y

    if x.ndim < 2:
        x = x.reshape(1,-1)
    if y.ndim < 2:
        y = y.reshape(1,-1)

    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if y.ndim > 2:
        y = y.reshape(y.shape[0], -1)

    D.X, D.y = x, y

    return D


class _ScalerShim:
    """Minimal StandardScaler stand-in exposing mean_/scale_ attributes.

    Used to inject an externally supplied standardisation into
    hyperparameter_selection without refitting a StandardScaler.

    """

    # ---- External Methods ----

    def __init__(self, mean, scale):
        import numpy as np
        self.mean_ = np.array(mean)
        self.scale_ = np.array(scale)

def hyperparameter_selection(cfg: DictConfig, D, num_folds: int, model_type, model_surrogate=None, rng_key: random.PRNGKey=jax.random.PRNGKey(0), x_scalar_override=None):
    """
    Cross-validate over (hidden_size, activation), retrain the best on all
    data, and return the model plus standardised/unstandardised query
    functions and its serialised payload.
    """
    if model_type == 'regressor':
        if model_surrogate == 'ctg_surrogate':
            surrogate_cfg = cfg.surrogate.surrogate_ctg.ann
        else:
            surrogate_cfg = cfg.surrogate.surrogate_forward.ann
    elif model_type == 'classifier':
        surrogate_cfg = cfg.surrogate.classifier_args.ann
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
    hidden_sizes = surrogate_cfg.hidden_size_options
    afs = surrogate_cfg.activation_functions

    best_hyperparams = {}
    best_avg_loss = float('inf')

    D = check_dims(D)

    if x_scalar_override is not None:
        import numpy as np
        x_mean_np = np.array(x_scalar_override.mean)
        x_std_np = np.array(x_scalar_override.std)
        x_scalar = _ScalerShim(x_mean_np, x_std_np)
        standard_X = (np.array(D.X) - x_mean_np) / x_std_np
    else:
        x_scalar = StandardScaler().fit(D.X)
        standard_X = x_scalar.transform(D.X)

    # probability_map uses a sigmoid head on [0, 1]; bypass y-standardisation
    out_af = 'sigmoid' if model_surrogate == 'probability_map_surrogate' else 'identity'

    if model_type == 'regressor':
        y_scalar = StandardScaler().fit(D.y)
        if out_af == 'sigmoid':
            y_scalar.mean_[:] = 0.0
            y_scalar.scale_[:] = 1.0
        standard_D = Dataset(standard_X, y_scalar.transform(D.y))
        if out_af == 'sigmoid':                    # flatten the P_feas target density
            standard_D = resample_inverse_density(standard_D)
    elif model_type == 'classifier':
        y_scalar = jnp.astype((D.y + 1)/2, jnp.int32)
        standard_D = Dataset(standard_X, y_scalar)

    # cross-validated hyperparameter selection
    for hidden_size, af in product(hidden_sizes, afs):
        if model_type == 'regressor':
            model = identify_neural_network(hidden_size, standard_D.y.shape[1], af, output_activation=out_af)
        elif model_type == 'classifier':
            model = identify_neural_network(hidden_size, 2, af)
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")
        avg_loss = train_nn_surrogate_model(surrogate_cfg, standard_D, model, num_folds, rng_key, model_type=model_type)

        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_hyperparams = {
                'hidden_size': hidden_size,
                'activation_function': af
            }

    # retrain the best hyperparameters on all the data
    if model_type == 'regressor':
        best_model = identify_neural_network(best_hyperparams['hidden_size'], standard_D.y.shape[1], best_hyperparams['activation_function'], output_activation=out_af)
        best_hyperparams['output_units'] = standard_D.y.shape[1]
    elif model_type == 'classifier':
        best_model = identify_neural_network(best_hyperparams['hidden_size'], 2, best_hyperparams['activation_function'])
        best_hyperparams['output_units'] = 2
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    # hold out one fold for early-stopping validation
    kf_final = KFold(n_splits=num_folds, shuffle=True, random_state=0)
    train_idx, val_idx = next(kf_final.split(standard_D.X))
    train_D = Dataset(standard_D.X[train_idx], standard_D.y[train_idx])
    val_D = Dataset(standard_D.X[val_idx], standard_D.y[val_idx])
    best_params, _, _ = train(surrogate_cfg, best_model, train_D, val_D, model_type=model_type)

    opt_model = partial(best_model.apply, best_params)
    x_mean = jnp.array(x_scalar.mean_)
    x_std = jnp.array(x_scalar.scale_)

    best_training_performance = getattr(train, 'last_training_performance', {})
    logging.info(
        f"training_performance ({model_type}): {best_training_performance} "
        f"| best_hyperparams={best_hyperparams} "
        f"| best_avg_val_loss={float(best_avg_loss):.4g}"
    )

    serialised_model = serialise_model(best_params, best_model, x_scalar, y_scalar, model_type, {})
    serialised_model['output_activation'] = out_af

    del standard_D
 

    @jit
    def standardise(x):
        return (x - x_mean) / x_std

    if model_type == 'regressor':

        y_mean = jnp.array(y_scalar.mean_)
        y_std = jnp.array(y_scalar.scale_)

        @jit
        def project(y):
            return y * y_std + y_mean

        @jit
        def query_unstandardised_model(x):
            if x.ndim <2 : x = x.reshape(1,-1)
            return project(opt_model(standardise(x)))

        @jit
        def query_standardised_model(x):
            if x.ndim <2: x = x.reshape(1,-1)
            return opt_model(x)

        return opt_model, (query_standardised_model, query_unstandardised_model, StandardisationMetrics(x_mean, x_std), StandardisationMetrics(y_mean, y_std)), serialised_model
    
    elif model_type == 'classifier':

        def mapp_(y):
            if y.ndim <2 : y = y.reshape(1,-1)
            # quantity <= 0 means the sample predicts feasibility
            return jnp.array([0.5]) - softmax(y, axis=-1)[0,0]

        @jit
        def query_unstandardised_classifier(x):
            if x.ndim <2 : x = x.reshape(1,-1)
            return mapp_(opt_model(standardise(x)))

        @jit
        def query_standardised_classifier(x):
            if x.ndim <2: x = x.reshape(1,-1)
            if x.shape[0]>= x.shape[1]: x= x.T
            return mapp_(opt_model(x))

        return opt_model, (query_standardised_classifier, query_unstandardised_classifier, StandardisationMetrics(x_mean, x_std)), serialised_model



def train_nn_surrogate_model(cfg: DictConfig, D, model: nn.Module, num_folds: int, rng_key: random.PRNGKey=jax.random.PRNGKey(0), model_type='regressor') -> float:
    """
    K-fold train and validate a model, returning the average validation
    loss used to score a candidate architecture.
    """
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)
    fold_indices = kf.split(D.X)

    fold_losses = []
    for train_index, val_index in fold_indices:
        X_train, X_val = D.X[train_index], D.X[val_index]
        y_train, y_val = D.y[train_index], D.y[val_index]

        trained_params, _, _ = train(cfg, model, Dataset(X_train, y_train), Dataset(X_val, y_val), model_type=model_type)

        y_pred = model.apply(trained_params, X_val)
        if model_type == 'classifier':
            fold_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(jnp.expand_dims(y_pred, axis=1), y_val))
        elif getattr(model, 'output_activation', 'identity') == 'sigmoid':
            fold_loss = _bce(y_val, y_pred)
        elif model_type == 'regressor':
            fold_loss = jnp.mean(jnp.square(y_val - y_pred))
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")

        fold_losses.append(fold_loss)

        del X_train, X_val, y_train, y_val

    avg_loss = jnp.mean(jnp.array(fold_losses))

    return avg_loss

       
_OUTPUT_AFS = {'identity': (lambda x: x), 'sigmoid': nn.sigmoid,
               'relu': nn.relu, 'tanh': nn.tanh}


class NeuralNetworkEstimator(nn.Module):
    """Flax MLP with configurable hidden layers and a typed output activation.

    Builds a stack of Dense layers with per-hidden-layer activations and a
    final output activation, used as the surrogate model throughout this module.

    """

    hidden_units: list
    output_units: int
    activation_functions: list
    output_activation: str = 'identity'

    # ---- External Methods ----

    def setup(self):
        """Construct the Dense layers and activation stack (flax hook)."""
        self.layers = [nn.Dense(hidden_unit) for hidden_unit in self.hidden_units] + [nn.Dense(self.output_units)]

        self.afs = []
        for i, af in enumerate(self.activation_functions):
            if af == 'relu':
                self.afs += (nn.relu,)
            elif af == 'sigmoid':
                self.afs += (nn.sigmoid,)
            elif af == 'tanh':
                self.afs += (nn.tanh,)

        self.afs += (_OUTPUT_AFS[self.output_activation],)

        # one activation per layer (output layer gets an implicit identity)
        assert len(self.afs) == len(self.layers), (
            f"NeuralNetworkEstimator layer/activation mismatch: "
            f"len(hidden_units)={len(self.hidden_units)} + 1 output layer "
            f"= {len(self.layers)} Dense layers, but "
            f"len(activation_functions)={len(self.activation_functions)} "
            f"+ 1 output identity = {len(self.afs)} afs.  "
            f"`activation_functions` must have exactly one entry per "
            f"hidden layer (it gets implicitly padded with an identity "
            f"for the output layer).  Fix the yaml — e.g. for "
            f"hidden_units={self.hidden_units!r} you need "
            f"activation_functions of length {len(self.hidden_units)}."
        )


    def __call__(self, x):
        """Forward pass applying each layer then its activation (flax hook)."""
        for i, (layer, af) in enumerate(zip(self.layers, self.afs)):
            x = layer(x)
            x = af(x)

        return x


def _bce(y, p, eps=1e-6):
    """
    Binary cross-entropy for a soft probability target. Matched to a sigmoid
    head so the logit gradient is (p - y), driving the saturated 0/1 corners
    that MSE leaves stranded.
    """
    p = jnp.clip(p, eps, 1.0 - eps)
    return -jnp.mean(y * jnp.log(p) + (1.0 - y) * jnp.log1p(-p))


@partial(jit, static_argnames=('model',))
def _loss_fn_regressor(params, model, batch):
    """Regression loss: BCE for a sigmoid head, otherwise MSE."""
    y_pred = model.apply(params, batch['X'])
    if getattr(model, 'output_activation', 'identity') == 'sigmoid':
        return _bce(batch['y'], y_pred)          # probability target -> BCE
    return jnp.mean(jnp.square(batch['y'] - y_pred))


@partial(jit, static_argnames=('model',))
def _loss_fn_classifier(params, model, batch):
    """Classification softmax cross-entropy loss over integer labels."""
    y_pred = model.apply(params, batch['X'])
    # labels map to {0, 1} with 0 the negative class and 1 the positive class
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(jnp.expand_dims(y_pred, axis=1), batch['y']))


_grad_fn_regressor  = jax.value_and_grad(_loss_fn_regressor)
_grad_fn_classifier = jax.value_and_grad(_loss_fn_classifier)


def train_one_step_regressor(state, model, batch):
    """Single value-and-grad step for the regression loss."""
    loss, grad = _grad_fn_regressor(state.params, model, batch)
    return loss, grad


def train_one_step_classifier(state, model, batch):
    """Single value-and-grad step for the classification loss."""
    loss, grad = _grad_fn_classifier(state.params, model, batch)
    return loss, grad


def get_initial_params(key: jax.Array, data:jnp.array, model: nn.Module) -> Dict:
  """Initialise model params from a Dataset's input shape."""
  input_dims = tuple(data.X.shape[1:])
  init_shape = jnp.ones(input_dims, jnp.float32)
  initial_params = model.init(key, init_shape)
  return initial_params

def get_initial_params_serial(key: jax.Array, data:jnp.array, model: nn.Module) -> Dict:
  """Initialise model params from a raw array's input shape."""
  input_dims = tuple(data.shape[1:])
  init_shape = jnp.ones(input_dims, jnp.float32)
  initial_params = model.init(key, init_shape)
  return initial_params


@struct.dataclass
class _ScanEarlyStopping:
    """JAX-traceable mirror of flax.training.early_stopping.EarlyStopping.

    Holds early-stopping state that threads through lax.scan: min_delta and
    patience are static fields, the metric/counter/flag fields are dynamic.

    """

    min_delta: float = struct.field(pytree_node=False)
    patience: int = struct.field(pytree_node=False)
    best_metric: jnp.ndarray = struct.field(pytree_node=True)
    patience_count: jnp.ndarray = struct.field(pytree_node=True)
    should_stop: jnp.ndarray = struct.field(pytree_node=True)

    # ---- External Methods ----

    @classmethod
    def create(cls, min_delta: float, patience: int, *, dtype=jnp.float32):
        """
        Build the initial state. dtype must match the validation-loss dtype
        so update's jnp.where keeps both lax.cond branches type-consistent.
        """
        return cls(
            min_delta=float(min_delta),
            patience=int(patience),
            best_metric=jnp.asarray(jnp.inf, dtype=dtype),
            patience_count=jnp.asarray(0, dtype=jnp.int32),
            should_stop=jnp.asarray(False),
        )

    def update(self, metric):
        """
        Advance the early-stopping state given a new validation metric,
        flipping should_stop once patience is exhausted.
        """
        # isinf handles the first call where best_metric starts at inf
        improved = (jnp.isinf(self.best_metric)
                    | (self.best_metric - metric > self.min_delta))
        new_best = jnp.where(improved, metric, self.best_metric)
        new_patience = jnp.where(
            improved, jnp.asarray(0, dtype=jnp.int32), self.patience_count + 1,
        )
        new_should_stop = jnp.where(
            improved,
            self.should_stop,
            (self.patience_count >= self.patience) | self.should_stop,
        )
        return self.replace(
            best_metric=new_best,
            patience_count=new_patience,
            should_stop=new_should_stop,
        )


def _architecture_key(model):
    """Hashable signature for compile-cache keying."""
    return (
        tuple(getattr(model, 'hidden_units', ()) or ()),
        getattr(model, 'output_units', None),
        tuple(getattr(model, 'activation_functions', ()) or ()),
    )


_TRAIN_PMAP_CACHE: Dict = {}


def _build_fused_train_pmap(model, model_type, num_devices, num_epochs, dispatch):
    """
    Build a pmap/vmap/serial wrapper around the epoch scan for one
    (architecture, model_type, num_devices, num_epochs, dispatch) signature,
    cached in _TRAIN_PMAP_CACHE.
    """
    if model_type == 'regressor':
        grad_fn = _grad_fn_regressor
    elif model_type == 'classifier':
        grad_fn = _grad_fn_classifier
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    loss_dtype = jnp.asarray(0.0).dtype

    def _val_loss(params, valid_X, valid_y):
        y_pred = model.apply(params, valid_X)
        if model_type == 'classifier':
            out = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(
                    jnp.expand_dims(y_pred, axis=1), valid_y,
                )
            )
        elif getattr(model, 'output_activation', 'identity') == 'sigmoid':
            out = _bce(valid_y, y_pred)
        else:
            out = jnp.mean(jnp.square(valid_y - y_pred))
        return out.astype(loss_dtype)

    devices = [d for i, d in enumerate(jax.devices('cpu')) if i < num_devices]

    def fused_train(state, minibatch, valid_X, valid_y, init_es):
        """
        Run the full epoch loop in one dispatch, returning the final state,
        per-epoch losses, early-stopping state and the epoch where it
        converged (else num_epochs).
        """

        def epoch_body(carry, epoch_idx):
            state, es, done, converged_at = carry

            def do_epoch():
                loss, grads = grad_fn(state.params, model, minibatch)
                grads = lax.pmean(grads, "device")
                new_state = state.apply_gradients(grads=grads)
                loss_avg = lax.pmean(loss, "device").astype(loss_dtype)
                val_loss = _val_loss(new_state.params, valid_X, valid_y)
                new_es = es.update(val_loss)
                new_done = done | new_es.should_stop
                # freeze converged_at at the first step we flip to done
                just_converged = new_done & (~done)
                new_converged_at = jnp.where(just_converged, epoch_idx, converged_at)
                return (
                    (new_state, new_es, new_done, new_converged_at),
                    (loss_avg, val_loss),
                )

            def skip_epoch():
                z = jnp.zeros((), dtype=loss_dtype)
                return (
                    (state, es, done, converged_at),
                    (z, z),
                )

            return lax.cond(done, skip_epoch, do_epoch)

        init_carry = (
            state,
            init_es,
            jnp.asarray(False),
            jnp.asarray(num_epochs, dtype=jnp.int32),
        )
        (state_f, es_f, _done_f, converged_at), (losses, val_losses) = lax.scan(
            epoch_body, init_carry, jnp.arange(num_epochs, dtype=jnp.int32),
        )
        return state_f, losses, val_losses, es_f, converged_at

    in_axes  = (None, 0, None, None, None)
    out_axes = 0

    if dispatch == "pmap":
        return jax.pmap(
            fused_train, axis_name="device",
            in_axes=in_axes, out_axes=out_axes, devices=devices,
        )
    if dispatch == "vmap":
        return jax.jit(jax.vmap(
            fused_train, axis_name="device",
            in_axes=in_axes, out_axes=out_axes,
        ))
    if dispatch == "serial":
        # length-1 vmap per iter keeps lax.pmean(..., "device") valid; loop runs host-side
        inner = jax.jit(jax.vmap(
            fused_train, axis_name="device",
            in_axes=in_axes, out_axes=out_axes,
        ))
        def _serial(state, minibatch, valid_X, valid_y, init_es):
            W = jax.tree_util.tree_leaves(minibatch)[0].shape[0]
            results = [
                inner(state,
                      jax.tree_util.tree_map(
                          lambda x: jnp.expand_dims(x[i], 0), minibatch),
                      valid_X, valid_y, init_es)
                for i in range(W)
            ]
            return jax.tree_util.tree_map(
                lambda *xs: jnp.concatenate(xs, axis=0), *results
            )
        return _serial
    raise ValueError(
        f"Unknown dispatch={dispatch!r}; expected 'pmap', 'vmap', or 'serial'"
    )


def _get_or_build_train_pmap(model, model_type, num_devices, num_epochs, dispatch):
    """Memoised factory — one compile per arch × model_type × devices × epochs × dispatch."""
    key = (
        _architecture_key(model),
        str(model_type),
        int(num_devices),
        int(num_epochs),
        str(dispatch),
    )
    fn = _TRAIN_PMAP_CACHE.get(key)
    if fn is None:
        fn = _build_fused_train_pmap(model, model_type, num_devices, num_epochs, dispatch)
        _TRAIN_PMAP_CACHE[key] = fn
    return fn


def train(cfg, model, data, valid_data, model_type):
    """
    Train one NN with device-parallel minibatch SGD and early stopping via
    a single cached pmap(scan) dispatch, returning the trained params,
    model and loss history.
    """
    # define optimiser
    if cfg.decaying_lr_and_clip_param:
        lr = optax.linear_schedule(
            init_value=cfg.learning_rate,
            end_value=cfg.terminal_lr,
            transition_steps=cfg.num_epochs,
        )
    else:
        lr = cfg.learning_rate

    tx = optax.adamw(lr, weight_decay=getattr(cfg, 'weight_decay', 0.0))

    # initialise parameters + TrainState
    params = get_initial_params(jax.random.PRNGKey(0), data, model)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    # create minibatches host-side, once per training run
    num_devices = jax.local_device_count('cpu')
    minibatches = create_minibatches(data, cfg.batch_size, num_devices)
    minibatches = minibatch_reshape(minibatches)
    actual_num_devices = int(minibatches['X'].shape[0])

    init_es = _ScanEarlyStopping.create(
        min_delta=cfg.min_delta,
        patience=cfg.patience,
        dtype=jnp.asarray(0.0).dtype,
    )

    # cached dispatch; defaults to 'pmap', overridable via cfg.dispatch
    fused_train = _get_or_build_train_pmap(
        model, model_type, actual_num_devices, int(cfg.num_epochs),
        str(getattr(cfg, "dispatch", "pmap")),
    )

    # one dispatch for the whole training run
    state_rep, losses, val_losses, es_rep, converged_at_rep = fused_train(
        state,
        minibatches,
        valid_data.X,
        valid_data.y,
        init_es,
    )

    # unreplicate once; pmean'd grads leave all pmap replicas identical
    state = jax_utils.unreplicate(state_rep)
    converged_at = int(jax_utils.unreplicate(converged_at_rep))
    num_epochs = int(cfg.num_epochs)
    losses_1d     = jnp.asarray(losses)[0, :converged_at]
    val_losses_1d = jnp.asarray(val_losses)[0, :converged_at]
    loss_history  = list(losses_1d)

    if converged_at < num_epochs:
        last_loss = float(losses_1d[-1]) if losses_1d.size else 0.0
        last_val  = float(val_losses_1d[-1]) if val_losses_1d.size else 0.0
        logging.debug(
            'Converged. Training stopped at iteration %d, '
            'loss value %.4f, val. loss value %.4f'
            % (converged_at, last_loss, last_val)
        )

    if model_type == 'classifier':
        def model_predict(params, data_points):
            return jnp.argmax(model.apply(params, data_points), axis=-1)

        def score(params, data_points, labels):
            return jnp.mean(jnp.equal(model_predict(params, data_points), labels.reshape(-1,)))

        accuracy = score(state.params, data.X, data.y)
        # confusion-matrix entries for false-positive diagnostics
        try:
            tn, fp, fn, tp = confusion_matrix(y_pred=jnp.astype(model_predict(state.params, data.X), jnp.int32), y_true=jnp.astype(data.y, jnp.int32).squeeze()).ravel()
        except:
            if data.y.all() == 0:
                tn, fp, fn, tp = len(data.y), 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, len(data.y)
        training_performance = {"acc": accuracy, "tn": tn, "fp": fp, "fn": fn, "tp": tp}
        logging.debug(f"--- {model_type} ---")
        logging.debug(f"training_performance: {training_performance}")

    if model_type == 'regressor':
        mse = jnp.mean(jnp.square(data.y - model.apply(state.params, data.X)))
        training_performance = {
            "mse": mse,
            "standardised_mape": jnp.mean(
                jnp.abs((data.y - model.apply(state.params, data.X)) / data.y)
            ),
        }
        logging.debug(f"--- {model_type} ---")
        logging.debug(f"training_performance: {training_performance}")

    train.last_training_performance = training_performance

    return state.params, model, loss_history


def get_serial_state_params(params):
    """Strip the leading pmap device axis from each replicated parameter."""
    return {'params': {layer: {mod: val[0] for mod,val in layer_v.items() } for layer, layer_v in params['params'].items()}}


def minibatch_reshape(batches):
    """Stack equal-sized minibatches into a leading device axis for pmap."""
    return {'X': jnp.vstack([batch['X'].reshape(1,-1,batch['X'].shape[-1]) for batch in batches if batch['X'].shape[0]==batches[0]['X'].shape[0]]), 'y': jnp.vstack([batch['y'].reshape(1,-1,batch['y'].shape[-1]) for batch in batches if batch['X'].shape[0]==batches[0]['X'].shape[0]])}

def create_minibatches(dataset, batch_size, num_devices=1):
    """Split a Dataset into at most num_devices minibatches for device sharding."""
    num_examples = dataset.X.shape[0]
    num_batches = num_examples // batch_size

    if num_batches > num_devices:
        num_batches = num_devices
        if num_devices == 1:
            batch_size = num_examples
        else:
            batch_size = num_examples // (num_devices-1)

    minibatches = []
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        if end > num_examples-1:
            end = num_examples-1
        minibatch = {'X': dataset.X[start:end], 'y': dataset.y[start:end]}
        minibatches.append(minibatch)


    # leftover examples form an extra minibatch
    if num_examples % batch_size != 0:
        if num_devices == 1:
            start = num_batches * batch_size
            minibatch = {'X': dataset.X[start:], 'y': dataset.y[start:]}
            minibatches.append(minibatch)
        else:
            # upsample
            start = num_batches * batch_size
            num_remaining = start - (batch_size - (num_examples % batch_size))
            minibatch = {'X': dataset.X[num_remaining:], 'y': dataset.y[num_remaining:]}
            minibatches.append(minibatch)

    return [minibatches[batch] for batch in range(min(num_devices, len(minibatches)))]


def _ann_mapp(y):
    """Map classifier logits to 0.5 - softmax(y)[0]; <= 0 means feasible."""
    if y.ndim < 2: y = y.reshape(1, -1)
    return jnp.array([0.5]) - softmax(y, axis=-1)[0, 0]

def _ann_forward_std_regressor(params, x, model):
    """Forward a regressor on already-standardised inputs."""
    if x.ndim < 2: x = x.reshape(1, -1)
    return model.apply(params, x)

def _ann_forward_unstd_regressor(params, x, x_mean, x_std, y_mean, y_std, model):
    """Forward a regressor on raw inputs, standardising in and de-standardising out."""
    if x.ndim < 2: x = x.reshape(1, -1)
    return model.apply(params, (x - x_mean) / x_std) * y_std + y_mean

def _ann_forward_std_classifier(params, x, model):
    """Forward a classifier on already-standardised inputs."""
    if x.ndim < 2: x = x.reshape(1, -1)
    if x.shape[0] >= x.shape[1]: x = x.T
    return _ann_mapp(model.apply(params, x))

def _ann_forward_unstd_classifier(params, x, x_mean, x_std, model):
    """Forward a classifier on raw inputs, standardising before the model."""
    if x.ndim < 2: x = x.reshape(1, -1)
    return _ann_mapp(model.apply(params, (x - x_mean) / x_std))

_ann_forward_std_regressor_jit    = jit(_ann_forward_std_regressor,    static_argnames=('model',))
_ann_forward_unstd_regressor_jit  = jit(_ann_forward_unstd_regressor,  static_argnames=('model',))
_ann_forward_std_classifier_jit   = jit(_ann_forward_std_classifier,   static_argnames=('model',))
_ann_forward_unstd_classifier_jit = jit(_ann_forward_unstd_classifier, static_argnames=('model',))


def build_ann(cfg, model_data, model_class):
    """
    Rebuild a serialised ANN and return a jitted predictor that maps raw
    inputs to de-standardised regressor outputs or classifier scores.
    """
    x_standardisation = model_data['standardisation_metrics_input']
    x_mean = x_standardisation.mean
    x_std = x_standardisation.std

    model = NeuralNetworkEstimator(hidden_units=model_data['hidden_units'], output_units=model_data['output_units'], activation_functions=model_data['activation_function'], output_activation=model_data.get('output_activation', 'identity'))
    params = get_initial_params_serial(jax.random.PRNGKey(0), x_mean.reshape(1,-1), model)
    params = from_bytes(params, model_data['serialized_params'])

    if model_class == 'regressor':
        y_standardisation = model_data['standardisation_metrics_output']
        y_mean = jnp.array(y_standardisation.mean)
        y_std = jnp.array(y_standardisation.std)

        def _predict_regressor(x):
            return _ann_forward_unstd_regressor_jit(
                params, x, x_mean, x_std, y_mean, y_std, model=model,
            )
        return _predict_regressor

    elif model_class == 'classifier':
        def _predict_classifier(x):
            return _ann_forward_unstd_classifier_jit(
                params, x, x_mean, x_std, model=model,
            )
        return _predict_classifier

    else:
        raise NotImplementedError(f"Model class {model_class} not implemented")

    