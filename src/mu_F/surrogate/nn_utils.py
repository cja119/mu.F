from typing import Dict
from itertools import product
from abc import ABC
import jax
import jax.numpy as jnp
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

from mu_F.surrogate.data_utils import standardisation_metrics



class Dataset(ABC):
    def __init__(self, X, y):
        self.input_rank = len(X.shape)
        self.output_rank = len(y.shape)
        self.X = X if self.input_rank >= 2 else jnp.expand_dims(X, axis=-1)
        self.y = y if self.output_rank >=2 else jnp.expand_dims(y,axis=-1)


# --- neural network regressor --- #

def identify_neural_network(hidden_units, output_units, activation_functions) -> nn.Module:
    return NeuralNetworkEstimator(hidden_units=hidden_units, output_units=output_units, activation_functions=activation_functions)

def serialise_model(params, model, x_scalar, y_scalar, model_type, model_data):
    model_data['hidden_units'] = model.hidden_units
    model_data['output_units'] = model.output_units
    model_data['activation_function'] = model.activation_functions
    model_data['serialized_params'] = to_bytes(params)
    model_data['standardisation_metrics_input'] = standardisation_metrics(x_scalar.mean_, x_scalar.scale_)
    if model_type == 'regressor':
        model_data['standardisation_metrics_output'] = standardisation_metrics(y_scalar.mean_, y_scalar.scale_)

    #assess model serialisation 
    # Function to compare original and deserialized parameters
    def compare_params(original, deserialized):
        for k, v in original.items():
            if isinstance(v, dict):
                compare_params(v, deserialized[k])
            else:
                assert jnp.allclose(v, deserialized[k]), f"Mismatch found in parameter {k}"

    # Compare original and deserialized parameters
    try:
        compare_params(params, from_bytes(params, model_data['serialized_params']))
        print("Serialization and deserialization successful. Parameters match!")
    except AssertionError as e:
        print(f"Error: {e}")
    return model_data

def check_dims(D):

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
    def __init__(self, mean, scale):
        import numpy as np
        self.mean_ = np.array(mean)
        self.scale_ = np.array(scale)

def hyperparameter_selection(cfg: DictConfig, D, num_folds: int, model_type, rng_key: random.PRNGKey=jax.random.PRNGKey(0), x_scalar_override=None): 
    # Define the hyperparameters to search over
    if model_type == 'regressor':
        surrogate_cfg = cfg.surrogate.surrogate_forward.ann
    elif model_type == 'classifier':
        surrogate_cfg = cfg.surrogate.classifier_args.ann
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
    hidden_sizes = surrogate_cfg.hidden_size_options
    afs = surrogate_cfg.activation_functions

    # Initialize the best hyperparameters and the best average validation loss
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

    if model_type == 'regressor':
        y_scalar = StandardScaler().fit(D.y)
        standard_D = Dataset(standard_X, y_scalar.transform(D.y))
    elif model_type == 'classifier':
        y_scalar = jnp.astype((D.y + 1)/2, jnp.int32)
        standard_D = Dataset(standard_X, y_scalar)

    # Perform hyperparameter selection using cross-validation
    for hidden_size, af in product(hidden_sizes, afs):
        # Set the current hyperparameters
        # Train the model using the current hyperparameters
        if model_type == 'regressor':
            model = identify_neural_network(hidden_size, standard_D.y.shape[1], af)
        elif model_type == 'classifier':
            model = identify_neural_network(hidden_size, 2, af)
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")
        avg_loss = train_nn_surrogate_model(surrogate_cfg, standard_D, model, num_folds, rng_key, model_type=model_type)

        # Check if the current hyperparameters are the best so far
        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_hyperparams = {
                'hidden_size': hidden_size,
                'activation_function': af
            }

    # Train the model with the best hyperparameters using all the data
    if model_type == 'regressor':
        best_model = identify_neural_network(best_hyperparams['hidden_size'], standard_D.y.shape[1], best_hyperparams['activation_function'])
        best_hyperparams['output_units'] = standard_D.y.shape[1]
    elif model_type == 'classifier':
        best_model = identify_neural_network(best_hyperparams['hidden_size'], 2, best_hyperparams['activation_function'])
        best_hyperparams['output_units'] = 2
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    # Hold out one fold for early stopping validation (consistent with CV above)
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
            return project(opt_model(standardise(x))) # pipeline model

        @jit
        def query_standardised_model(x):
            if x.ndim <2: x = x.reshape(1,-1)
            return opt_model(x)

        return opt_model, (query_standardised_model, query_unstandardised_model, standardisation_metrics(x_mean, x_std), standardisation_metrics(y_mean, y_std)), serialised_model
    
    elif model_type == 'classifier':

        def mapp_(y):
            if y.ndim <2 : y = y.reshape(1,-1)
            return jnp.array([0.5]) - softmax(y, axis=-1)[0,0]    # if this quantity is less than or equal to zero, then the sample predicts feasibility.

        @jit
        def query_unstandardised_classifier(x):
            if x.ndim <2 : x = x.reshape(1,-1)
            return mapp_(opt_model(standardise(x))) # pipeline model
        
        @jit
        def query_standardised_classifier(x):
            if x.ndim <2: x = x.reshape(1,-1)
            if x.shape[0]>= x.shape[1]: x= x.T
            return mapp_(opt_model(x))
        
        return opt_model, (query_standardised_classifier, query_unstandardised_classifier, standardisation_metrics(x_mean, x_std)), serialised_model #, standardisation_metrics(jnp.zeros((1,2)), jnp.ones((1,2))))



def train_nn_surrogate_model(cfg: DictConfig, D, model: nn.Module, num_folds: int, rng_key: random.PRNGKey=jax.random.PRNGKey(0), model_type='regressor') -> float:
    # Split data into folds
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)
    fold_indices = kf.split(D.X)


    # Train and validate the model for each fold
    fold_losses = []
    for train_index, val_index in fold_indices:
        # Split data into train and validation sets
        X_train, X_val = D.X[train_index], D.X[val_index]
        y_train, y_val = D.y[train_index], D.y[val_index]

        trained_params, _, _ = train(cfg, model, Dataset(X_train, y_train), Dataset(X_val, y_val), model_type=model_type)

        # Evaluate the model on the validation set
        y_pred = model.apply(trained_params, X_val)
        if model_type == 'classifier':
            fold_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(jnp.expand_dims(y_pred, axis=1), y_val))
        elif model_type == 'regressor':
            fold_loss = jnp.mean(jnp.square(y_val - y_pred))
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")
        
        fold_losses.append(fold_loss)

        del X_train, X_val, y_train, y_val

    # Compute average validation loss across folds
    avg_loss = jnp.mean(jnp.array(fold_losses))


    return avg_loss

       
class NeuralNetworkEstimator(nn.Module):
    hidden_units: list
    output_units: int
    activation_functions: list

    def setup(self):
        self.layers = [nn.Dense(hidden_unit) for hidden_unit in self.hidden_units] + [nn.Dense(self.output_units)]

        self.afs = []
        for i, af in enumerate(self.activation_functions):
            if af == 'relu':
                self.afs += (nn.relu,)
            elif af == 'sigmoid':
                self.afs += (nn.sigmoid,)
            elif af == 'tanh':
                self.afs += (nn.tanh,)

        self.afs += (lambda x: x,) # currently have no output activation function (feel free to change it)

        # -- Hard-fail structural check ---------------------------------
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
        for i, (layer, af) in enumerate(zip(self.layers, self.afs)):
            x = layer(x)
            x = af(x)

        return x


@partial(jit, static_argnames=('model',))
def _loss_fn_regressor(params, model, batch):
    y_pred = model.apply(params, batch['X'])
    return jnp.mean(jnp.square(batch['y'] - y_pred))


@partial(jit, static_argnames=('model',))
def _loss_fn_classifier(params, model, batch):
    #labels must be a one hot encoded array
    y_pred = model.apply(params, batch['X'])
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(jnp.expand_dims(y_pred, axis=1), batch['y']))  #  turns labels to 0 and 1, rather than -1 and 1, with zero indicating the negative class and 1 the positive class


_grad_fn_regressor  = jax.value_and_grad(_loss_fn_regressor)
_grad_fn_classifier = jax.value_and_grad(_loss_fn_classifier)


def train_one_step_regressor(state, model, batch):
    loss, grad = _grad_fn_regressor(state.params, model, batch)
    return loss, grad


def train_one_step_classifier(state, model, batch):
    loss, grad = _grad_fn_classifier(state.params, model, batch)
    return loss, grad


def get_initial_params(key: jax.Array, data:jnp.array, model: nn.Module) -> Dict:
  input_dims = tuple(data.X.shape[1:])  # (minibatch, height, width, stacked frames))
  init_shape = jnp.ones(input_dims, jnp.float32)
  initial_params = model.init(key, init_shape)#['params']
  return initial_params

def get_initial_params_serial(key: jax.Array, data:jnp.array, model: nn.Module) -> Dict:
  input_dims = tuple(data.shape[1:])  # (minibatch, height, width, stacked frames))
  init_shape = jnp.ones(input_dims, jnp.float32)
  initial_params = model.init(key, init_shape)#['params']
  return initial_params


@struct.dataclass
class _ScanEarlyStopping:
    """
    JAX-traceable mirror of `flax.training.early_stopping.EarlyStopping`.

    `min_delta` and `patience` are static pytree fields — they bake into
    the compiled kernel as constants rather than flowing as tracers.
    The mutable fields (`best_metric`, `patience_count`, `should_stop`)
    thread through `lax.scan` cleanly as dynamic state.
    """
    min_delta: float = struct.field(pytree_node=False)
    patience: int = struct.field(pytree_node=False)
    best_metric: jnp.ndarray = struct.field(pytree_node=True)
    patience_count: jnp.ndarray = struct.field(pytree_node=True)
    should_stop: jnp.ndarray = struct.field(pytree_node=True)

    @classmethod
    def create(cls, min_delta: float, patience: int, *, dtype=jnp.float32):
        """
        `dtype` must match the dtype that `metric` will be passed in as
        (i.e. the computed validation loss dtype).  If they don't match,
        `update`'s `jnp.where` will promote `best_metric` to the wider
        type in one branch of `lax.cond` but not in the other — which
        fails `cond`'s equal-output-types invariant.

        The caller in `train()` pulls this from `valid_data.y.dtype`
        since the validation-loss compute preserves that dtype.
        """
        return cls(
            min_delta=float(min_delta),
            patience=int(patience),
            best_metric=jnp.asarray(jnp.inf, dtype=dtype),
            patience_count=jnp.asarray(0, dtype=jnp.int32),
            should_stop=jnp.asarray(False),
        )

    def update(self, metric):
        # `jnp.isinf(best)` handles the first-call case (best starts at inf)
        # where `best - metric > min_delta` would be inf - finite = inf > x,
        # which is true anyway — but we keep the explicit isinf for parity
        # with flax's reference implementation.
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


def _build_fused_train_pmap(model, model_type, num_devices, num_epochs):
    """
    Construct a `jax.pmap(scan_over_epochs)` for one `(architecture, model_type,
    num_devices, num_epochs)` signature.  Cached in `_TRAIN_PMAP_CACHE`.
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
        else:
            out = jnp.mean(jnp.square(valid_y - y_pred))
        return out.astype(loss_dtype)

    devices = [d for i, d in enumerate(jax.devices('cpu')) if i < num_devices]

    @partial(
        jax.pmap,
        axis_name="device",
        in_axes=(None, 0, None, None, None),
        out_axes=0,
        devices=devices,
    )
    def fused_train(state, minibatch, valid_X, valid_y, init_es):
        """
        Full training loop fused into one pmap dispatch.

        Args (in-axes):
          state      : TrainState, broadcast  (in_axes=None).
          minibatch  : {X,y} dict, sharded    (in_axes=0).
          valid_X    : broadcast              (in_axes=None).
          valid_y    : broadcast              (in_axes=None).
          init_es    : initial _ScanEarlyStopping, broadcast.

        Returns (stacked along pmap axis 0):
          final_state, losses[num_epochs], val_losses[num_epochs],
          final_es, converged_at_epoch.

        `converged_at_epoch` is set on the step where `should_stop` first
        flips True, else stays at `num_epochs`.  Used host-side to trim
        the loss history.
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
                # Freeze converged_at at the first step where we flip to done.
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

    return fused_train


def _get_or_build_train_pmap(model, model_type, num_devices, num_epochs):
    """Memoised factory — one compile per arch × model_type × devices × epochs."""
    key = (
        _architecture_key(model),
        str(model_type),
        int(num_devices),
        int(num_epochs),
    )
    fn = _TRAIN_PMAP_CACHE.get(key)
    if fn is None:
        fn = _build_fused_train_pmap(model, model_type, num_devices, num_epochs)
        _TRAIN_PMAP_CACHE[key] = fn
    return fn


def train(cfg, model, data, valid_data, model_type):
    """
    Train one NN with device-parallel minibatch SGD + early stopping.

    Uses a single pmap(scan) dispatch for the whole training run — the
    pmap is cached in `_TRAIN_PMAP_CACHE` and reused across K-fold
    retraining of the same architecture.  See the block comment at the
    top of this module for the design rationale.
    """

    # define optimizer
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

    # Create minibatches — host side, once per training run.
    num_devices = jax.local_device_count('cpu')
    minibatches = create_minibatches(data, cfg.batch_size, num_devices)
    minibatches = minibatch_reshape(minibatches)
    actual_num_devices = int(minibatches['X'].shape[0])

    init_es = _ScanEarlyStopping.create(
        min_delta=cfg.min_delta,
        patience=cfg.patience,
        dtype=jnp.asarray(0.0).dtype,
    )

    # Cached pmap — compiles once per (arch, model_type, device_count, num_epochs).
    fused_train = _get_or_build_train_pmap(
        model, model_type, actual_num_devices, int(cfg.num_epochs),
    )

    # One dispatch for the whole training run.
    state_rep, losses, val_losses, es_rep, converged_at_rep = fused_train(
        state,
        minibatches,
        valid_data.X,
        valid_data.y,
        init_es,
    )

    # Unreplicate once; pmap replicas are identical (pmean'd grads ⇒ identical state).
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
        # get classifier performance
        def model_predict(params, data_points):
            return jnp.argmax(model.apply(params, data_points), axis=-1)

        def score(params, data_points, labels):
            return jnp.mean(jnp.equal(model_predict(params, data_points), labels.reshape(-1,)))

        accuracy = score(state.params, data.X, data.y)
        # get classifier false positive rates
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
    return {'params': {layer: {mod: val[0] for mod,val in layer_v.items() } for layer, layer_v in params['params'].items()}}


def minibatch_reshape(batches):
    return {'X': jnp.vstack([batch['X'].reshape(1,-1,batch['X'].shape[-1]) for batch in batches if batch['X'].shape[0]==batches[0]['X'].shape[0]]), 'y': jnp.vstack([batch['y'].reshape(1,-1,batch['y'].shape[-1]) for batch in batches if batch['X'].shape[0]==batches[0]['X'].shape[0]])}

def create_minibatches(dataset, batch_size, num_devices=1):
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


    # If there are leftover examples, create an additional mini-batch
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
    # Match the original `query_unstandardised_classifier` mapping:
    #   0.5 - softmax(y)[0]
    # `<= 0 ⇒ feasible` convention.  The earlier logit-difference variant
    # was a workaround for the macOS XLA-CPU compile crash; that crash has
    # since been fixed (the off-by-one in `build_ann`'s positional partial
    # was returning garbage from every surrogate, so the SQP was solving a
    # nonsense kernel).
    if y.ndim < 2: y = y.reshape(1, -1)
    return jnp.array([0.5]) - softmax(y, axis=-1)[0, 0]

def _ann_forward_std_regressor(params, x, model):
    if x.ndim < 2: x = x.reshape(1, -1)
    return model.apply(params, x)

def _ann_forward_unstd_regressor(params, x, x_mean, x_std, y_mean, y_std, model):
    if x.ndim < 2: x = x.reshape(1, -1)
    return model.apply(params, (x - x_mean) / x_std) * y_std + y_mean

def _ann_forward_std_classifier(params, x, model):
    if x.ndim < 2: x = x.reshape(1, -1)
    if x.shape[0] >= x.shape[1]: x = x.T
    return _ann_mapp(model.apply(params, x))

def _ann_forward_unstd_classifier(params, x, x_mean, x_std, model):
    if x.ndim < 2: x = x.reshape(1, -1)
    return _ann_mapp(model.apply(params, (x - x_mean) / x_std))

_ann_forward_std_regressor_jit    = jit(_ann_forward_std_regressor,    static_argnames=('model',))
_ann_forward_unstd_regressor_jit  = jit(_ann_forward_unstd_regressor,  static_argnames=('model',))
_ann_forward_std_classifier_jit   = jit(_ann_forward_std_classifier,   static_argnames=('model',))
_ann_forward_unstd_classifier_jit = jit(_ann_forward_unstd_classifier, static_argnames=('model',))


def build_ann(cfg, model_data, model_class):

    # Determine the input and output dimensions
    x_standardisation = model_data['standardisation_metrics_input']
    x_mean = x_standardisation.mean
    x_std = x_standardisation.std

    # Create the neural network estimator
    model = NeuralNetworkEstimator(hidden_units=model_data['hidden_units'], output_units=model_data['output_units'], activation_functions=model_data['activation_function'])
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

    