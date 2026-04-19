import time
from typing import Callable
import functools

import jax.numpy as jnp
import numpy as np
from casadi import DM, Callback, Function, Sparsity
from jax import jvp, jit, vjp
from functools import partial


def _shape_sig(v):
    """Recursively compute a hashable shape descriptor for a value"""
    if hasattr(v, 'shape') and hasattr(v, 'dtype'):
        return ('array', v.shape, str(v.dtype))
    elif isinstance(v, dict):
        return ('dict', tuple(sorted((_shape_sig(k), _shape_sig(vv)) for k, vv in v.items())))
    elif isinstance(v, (list, tuple)):
        return (type(v).__name__, tuple(_shape_sig(vv) for vv in v))
    else:
        return ('other', type(v).__name__)


def _fn_shape_signature(fn):
    """Compute a hashable signature for a callable based on function identity + arg shapes.

    For a functools.partial (the common case from build_svm / build_ann), the
    signature is (id of the base function, shapes of positional args, shapes of
    keyword args).  Two partials wrapping the same module-level @jit function
    with the same array shapes will hash identically, allowing _vjp_fun /
    _jvp_fun to reuse the same compiled XLA kernel rather than recompiling per
    evaluator.
    """
    if isinstance(fn, functools.partial):
        return (
            id(fn.func),
            tuple(_shape_sig(a) for a in fn.args),
            tuple((k, _shape_sig(v)) for k, v in sorted(fn.keywords.items())),
        )
    return id(fn)


class HashableCallable:
    """Wraps a callable so it can be used as a JAX static argument.

    functools.partial objects that close over numpy arrays are not hashable,
    which causes _vjp_fun / _jvp_fun (static_argnums=0) to crash.

    Hashing is based on function identity + argument shapes rather than a
    unique counter.  Two surrogates with the same architecture (same underlying
    @jit function, same array shapes) share a single compiled VJP/JVP kernel
    instead of triggering one 2-minute XLA recompilation per evaluator.
    """
    def __init__(self, fn: Callable):
        self._fn = fn
        self._sig = _fn_shape_signature(fn)

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __hash__(self):
        return hash(self._sig)

    def __eq__(self, other):
        return isinstance(other, HashableCallable) and self._sig == other._sig


@partial(jit, static_argnums=(0,))
def _vjp_fun(forward_fn, primals, tangents):
    return vjp(forward_fn, primals)[1](tangents)[0]


@partial(jit, static_argnums=(0,))
def _jvp_fun(forward_fn, primals, tangents):
    return jvp(forward_fn, (primals,), (tangents,))[1]


# --- JAX-CasADi Callback Wrappers ---
# These classes wrap JAX functions to be used within CasADi optimization problems
# They support both forward and reverse mode automatic differentiation
# Forward or reverse is selected based on the decision variable and constraints dimensions


class JaxCasADiEvaluator(Callback):
    def __init__(self, functn: Callable, nd: int, name="TensorFlowEvaluatorwReverse", opts={}, initial_x=None):
        super().__init__()
        self.nd = nd
        self.output_shape = None
        self.refs = [] # List to hold references to reverse callbacks
        self._forward_pass = HashableCallable(functn)
        # Determine output shape and dimension by tracing (dummy run)
        dummy_x = jnp.array(initial_x).reshape(nd, 1) if initial_x is not None else jnp.zeros((nd, 1))
        dummy_y = self._forward_pass(dummy_x)
        self.output_shape = dummy_y.shape
        self.n_out_dim = self.output_shape[0]

        # Construct CasADi's base class
        self.construct(name, opts)

    # --- CasADi Interface Methods ---

    def get_n_in(self): return 1 # Single input variable (x)
    def get_n_out(self): return 1 # Single output variable (y)
 
    def get_sparsity_in(self, i):
        assert i == 0
        return Sparsity.dense(self.nd, 1)

    def get_sparsity_out(self, i):
        assert i == 0
        return Sparsity.dense(self.n_out_dim, 1)

    def eval(self, arg):
        """ Computes the forward pass y = f(x). """
        x_jnp = jnp.array(arg[0]).reshape(self.nd, 1)
        y_jnp = self._forward_pass(x_jnp).reshape(-1)
        return [DM(np.array(y_jnp))]

    def has_reverse(self, nadj): return nadj == 0
    
    def has_forward(self, nfwd): return nfwd == 0


class JaxCallbackForward(JaxCasADiEvaluator):
    def __init__(self, functn: Callable, nd: int, name="JaxEvaluatorwForward", opts={}):
        super().__init__(functn, nd, name, opts)
        self.counter = 0
        self.time = 0
        self.forward_sensitivities = partial(_jvp_fun, self._forward_pass)

    def eval(self, arg):
        self.counter += 1
        t0 = time.time()
        ret = JaxCasADiEvaluator.eval(self, arg)
        self.time += time.time() - t0
        return ret
    
    def has_forward(self, nfwd): return nfwd == 1
    
    def get_forward(self, nfwd, name, inames, onames, opts):
        """ Returns a CasADi Callback that computes the forward mode AD
        of the JAX function.
        """
        assert(nfwd==1)
        nd = self.nd
        forward_sens = self.forward_sensitivities

        class ForwardEvaluator(Callback):
            def __init__(self, forward_fn, nd, n_out_dim, name, opts):
                super().__init__()
                self._forward_sensitivities = forward_fn
                self.nd = nd
                self.n_out_dim = n_out_dim
                self.construct(name, opts)

            def get_n_in(self): return 3 # x, y (nominal_out, unused), fwd_seed
            
            def get_n_out(self): return 1 # fwd_y

            def get_sparsity_in(self, i):
                if i == 0: return Sparsity.dense(self.nd, 1)        # x
                if i == 1: return Sparsity.dense(self.n_out_dim, 1) # y (nominal_out)
                if i == 2: return Sparsity.dense(self.nd, 1)        # fwd_seed
            
            def get_sparsity_out(self, i):
                return Sparsity.dense(self.n_out_dim, 1) # fwd_y

            def eval(self, arg):
                # CasADi passes: arg[0]=x, arg[1]=y (ignored), arg[2]=fwd_seed
                x_jnp = jnp.array(arg[0]).reshape(self.nd,)
                tangents = jnp.array(arg[2]).reshape(self.nd,)
                fwd_sens_list = self._forward_sensitivities(x_jnp, tangents)
                return [DM(np.array(fwd_sens_list))]

        # Instantiate the forward evaluator callback
        fwd_callback = ForwardEvaluator(forward_sens, self.nd, self.n_out_dim, f"{name}_forward", opts)
        self.refs.append(fwd_callback)

        # Construct the CasADi Function wrapper
        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        fwd_seed = self.mx_in()

        # CasADi function signature: (x, y, fwd_seed) -> (fwd_y)
        return Function(
            name,
            nominal_in + nominal_out + fwd_seed,
            fwd_callback.call(nominal_in + nominal_out + fwd_seed),
            inames,
            onames
        )
    

class JaxCallbackReverse(JaxCasADiEvaluator):
    def __init__(self, functn: Callable, nd: int, name="JaxEvaluatorwReverse", opts={}, initial_x=None):
        super().__init__(functn, nd, name, opts, initial_x=initial_x)
        self.counter = 0
        self.time = 0
        self.reverse_pass = partial(_vjp_fun, self._forward_pass)

    def eval(self, arg):
        self.counter += 1
        t0 = time.time()
        ret = JaxCasADiEvaluator.eval(self, arg)
        self.time += time.time() - t0
        return ret
    
    def has_reverse(self, nadj): return nadj == 1
    
    def get_reverse(self, nadj, name, inames, onames, opts):
        """ Returns a CasADi Callback that computes the forward mode AD
        of the JAX function.
        """
        assert(nadj==1)
        nd = self.nd
        reverse_pass = self.reverse_pass

        class ReverseEvaluator(Callback):
            def __init__(self, reverse_fn, nd, n_out_dim, name, opts):
                super().__init__()
                self._reverse_pass = reverse_fn
                self.nd = nd
                self.n_out_dim = n_out_dim
                self.construct(name, opts)

            def get_n_in(self): return 3 # x, y (nominal_out, unused), adj_seed 
            def get_n_out(self): return 1 # grad_x

            # Sparsity definitions mirror the main callback's definitions
            def get_sparsity_in(self, i):
                if i == 0: return Sparsity.dense(self.nd, 1)        # x
                if i == 1: return Sparsity.dense(self.n_out_dim, 1) # y (nominal_out)
                if i == 2: return Sparsity.dense(self.n_out_dim, 1) # adj_seed
            
            def get_sparsity_out(self, i):
                return Sparsity.dense(self.nd, 1) # grad_x  
            
            def eval(self, arg):
                # CasADi passes: arg[0]=x, arg[1]=y (ignored), arg[2]=adj_seed
                x_jnp = jnp.array(arg[0]).reshape(self.nd,1)
                adj_seed_jnp = jnp.array(arg[2]).reshape(self.n_out_dim,1)
                # Execute the jit compiled reverse
                vjp_jax = self._reverse_pass(x_jnp, adj_seed_jnp)
                # Convert back to CasADi DMatrix
                return [DM(np.array(vjp_jax))]

        # Instantiate the forward evaluator callback
        rev_callback = ReverseEvaluator(reverse_pass, nd, self.n_out_dim, f"{name}_reverse", opts)
        self.refs.append(rev_callback)

        # Construct the CasADi Function wrapper
        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        rev_seed = self.mx_out()

        # CasADi function signature: (x, y, rev_seed) -> (rev_y)
        return Function(
            name,
            nominal_in + nominal_out + rev_seed,
            rev_callback.call(nominal_in + nominal_out + rev_seed),
            inames,
            onames
        )


def casadify_forward(functn, nd):
    """Casadify a JAX function via JAX-CasADi wrappers
    functn: a JAX function
    nd: the number of input dimensions to the function
    ny: the number of output dimensions
    """

    return JaxCallbackForward(functn, nd)
    

def casadify_reverse(functn, nd, initial_x=None):
    """Casadify a JAX function via JAX-CasADi wrappers
    functn: a JAX function
    nd: the number of input dimensions to the function
    ny: the number of output dimensions
    """

    return JaxCallbackReverse(functn, nd, initial_x=initial_x)
