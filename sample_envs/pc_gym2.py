"""
A general interface for pc-gym problems
"""

from functools import partial
from dataclasses import dataclass
from flax.core import FrozenDict, unfreeze
import jax.numpy as jnp
import numpy as np
import jax

ENV_PARAMS = FrozenDict(
    {
        "cstr_ode": {
            "case_study": "cstr_ode",
            "N": 100,
            "tsim": 25,
            "SP": {"Ca": tuple([0.85] * 10 + [0.9] * 10)},
            "a_space": {"low": (295,), "high": (302,)},
            "o_space": {"low": (0.7, 300, 0.8), "high": (1.0, 350, 0.9)},
            "x0": (0.8, 330, 0.8),
            "model": "cstr_ode",
            "integration_method": "jax",
            "maximise_reward": False,
            "normalise_a": False,
            "normalise_o": False,
            "disturbance_bounds": {"low": (325, 0.95), "high": (375, 1.05)},
            "disturbance_keys": ('Ti', 'Caf'),
            't_lower': 319,
            't_upper': 331,
        }
    }
)

SHAPE_DICTS = {
    "cstr_ode": {
        "X_SIZE": 2,
        "U_SIZE": 1,
        "F_SIZE": 2,
        "G_SIZE": 2,
        "Z_SIZE": 2,
        "L_SIZE": 1,
        "PHI_SIZE": 0,
    },
}


@dataclass(frozen=False, kw_only=True)
class BaseModel:
    int_method: str = "jax"

    def info(self) -> dict:
        info = {
            "parameters": self.__dict__.copy(),
            "states": self.states,
            "inputs": self.inputs,
            "disturbances": self.disturbances,
            "uncertainties": list(self.uncertainties.keys()) if self.uncertainties else [],
        }
        info["parameters"].pop("int_method", None)
        return info

@dataclass(frozen=False, kw_only=True)
class cstr(BaseModel):
    q: float = 100
    V: float = 100
    rho: float = 1000
    C: float = 0.239
    deltaHr: float = -5e4
    EA_over_R: float = 8750
    k0: float = 7.2e10
    UA: float = 5e4
    Ti: float = 350
    Caf: float = 1
    int_method: str = 'jax'
    states: list = None
    inputs: list = None
    disturbances: list = None
    uncertainties: dict = None

    def __post_init__(self):
        self.states = ["Ca", "T"]
        self.inputs = ["Tc"]
        self.disturbances = ["Ti", "Caf"]

    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # Diffrax/JAX tracing can hand in shape (1, n) or (n, 1) views; flatten first.
        x = jnp.ravel(x)
        u = jnp.ravel(u)
        ca, T = x[0], x[1]
        Tc = u[0]
        rA = self.k0 * jnp.exp(-self.EA_over_R / T) * ca
        dxdt = jnp.array([
            self.q / self.V * (self.Caf - ca) - rA,
            self.q / self.V * (self.Ti - T)
            + ((-self.deltaHr) * rA) * (1 / (self.rho * self.C))
            + self.UA * (Tc - T) * (1 / (self.rho * self.C * self.V)),
        ])
        return dxdt
            
def simulator(
    param_dict, node, x: jnp.ndarray, u: jnp.ndarray, z: jnp.ndarray = None
) -> jnp.ndarray:
    import importlib
        
    #mod = importlib.import_module("pcgym.model_classes")
    #ModelCls = getattr(mod, param_dict["case_study"])   # e.g. "cstr"
    model = cstr(int_method="jax")

    # Normalize traced inputs to 1D vectors before model/constraint math.
    x = jnp.ravel(x)
    u = jnp.ravel(u)

    dxdt = model(x, u).squeeze()
    dgdt = jnp.concatenate([jnp.atleast_1d(c(param_dict, x, u)) for c in CONS_HOLDER[param_dict["case_study"]]], axis=0)
    rwd = jnp.square(param_dict["SP"]["Ca"][node] - x[0]) 
    

    # Jacobian of RHS wrt state x=[Ca, T], useful for diagnosing stiffness during integration.
    ca, T = x[0], x[1]
    rA = model.k0 * jnp.exp(-model.EA_over_R / T) * ca
    drA_dca = model.k0 * jnp.exp(-model.EA_over_R / T)
    drA_dT = rA * (model.EA_over_R / (T * T))
    dfdx = jnp.array(
        [
            [-model.q / model.V - drA_dca, -drA_dT],
            [
                ((-model.deltaHr) / (model.rho * model.C)) * drA_dca,
                -model.q / model.V
                + ((-model.deltaHr) / (model.rho * model.C)) * drA_dT
                - model.UA * (1 / (model.rho * model.C * model.V)),
            ],
        ]
    )

    return jnp.concatenate([jnp.ravel(dxdt), jnp.ravel(dgdt), jnp.ravel(rwd)], axis=0)



# -------------------------------------------------------------------------------- #
# ----------------------------- Constraint Functions ----------------------------- #
# -------------------------------------------------------------------------------- #
@partial(jax.jit, static_argnums=(0,))
def cstr_cons_1(param_dict, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    return jnp.square(jnp.minimum(0.0, (x[1] - param_dict["t_lower"]) / param_dict["t_upper"])) 

@partial(jax.jit, static_argnums=(0,))
def cstr_cons_2(param_dict, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    return jnp.square(jnp.minimum(0.0, (param_dict["t_upper"] - x[1]) / param_dict["t_upper"]))


# -------------------------------------------------------------------------------- #
# ------------------------------ Constraint Helpers ------------------------------ #
# -------------------------------------------------------------------------------- #


def make_pc_constraints(constraint_fns, param_dict):
    def cons(state, control):
        ctrl = jnp.ravel(control)
        vals = [cf(param_dict, state, ctrl) for cf in constraint_fns]
        return jnp.concatenate(vals, axis=0).reshape(
            -1,
        )

    return cons


CONS_HOLDER = {"cstr_ode": [cstr_cons_1, cstr_cons_2]}
