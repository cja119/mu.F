"""
Base environment configuration - all environments should inherit from this.

NOTE:
    - This environment will be used for both dynamic programming evalutation and testing. 
    - The step interfact will be used in simulation, whilst F and G will be used in Mu_F.
"""


from functools import partial

import jax.numpy as jnp
from jax import jit

from omegaconf import OmegaConf, DictConfig
from abc import ABC, abstractmethod
from operator import ge, le

from mu_F.control.utils import HYDRA_CONFIG_FILE, OUTPUTS_DIR

class MarkovEnvironment(ABC):
    X_SIZE = None  # Size of state vector
    U_SIZE = None  # Size of action vector
    F_SIZE = None  # Size of output observation vector
    G_SIZE = None  # Size of constraint space vector
    Z_SIZE = None  # Size of uncertainty vector
    L_SIZE = 1     # Size of stage cost 
    PHI_SIZE = 1   # Size of terminal cost

    """
    Here we will use control theory nomenclature

    Some notes:
        - x is the system state (node input, u in mu_F)
        - u is the control action (node output, v in mu_F)
        - z is the uncertainty (theta in mu_F)
        - F is the observation (y in mu_F)
        - G is the constraint space
        - L is the stage cost
        - ϕ is the terminal cost

    """
    def __init__(self, **kwargs):

        # Build the configuration for the environment
        self.cfg = self._build_cfg(**kwargs)
        self._initialise_env(self.cfg)
        self._set_dim(**kwargs)

        simulator_fn = lambda x, u, z: kwargs['simulator'](kwargs['env_params'], x, u, z)
        self._simulate_fn = lambda x, u, z, node=None: self.simulate(simulator_fn, x, u, z, node=node)
        
        self.model_cfg = self.cfg.model if hasattr(self.cfg, "model") else self.cfg
        self.current_step = 0
        self.max_steps = self.model_cfg.number_repeats
        self._cache = self.model_cfg.memory
        

    # ---- Mu-F Interface ---- #
    @classmethod
    def F(cls, output: jnp.ndarray) -> jnp.ndarray:
        return output[..., :cls.F_SIZE]

    @classmethod
    def G(cls, output: jnp.ndarray) -> jnp.ndarray:
        return output[..., cls.F_SIZE:cls.F_SIZE + cls.G_SIZE]

    @classmethod
    def R(cls, output: jnp.ndarray) -> jnp.ndarray:
        return output[..., cls.F_SIZE + cls.G_SIZE:cls.F_SIZE + cls.G_SIZE + cls.L_SIZE]

    @classmethod
    def phi(cls, output: jnp.ndarray) -> jnp.ndarray:
        return output[..., cls.F_SIZE + cls.G_SIZE + cls.L_SIZE:]
    
    def __call__(
        self, x: jnp.ndarray,u: jnp.ndarray, z=jnp.ndarray, node=None
    ) -> jnp.ndarray:
        
        assert (
            x.shape[-1] == self.X_SIZE
        ), f"Expected last dimension {self.X_SIZE}, got {x.shape[-1]}"
        assert (
            u.shape[-1] == self.U_SIZE
        ), f"Expected last dimension {self.U_SIZE}, got {u.shape[-1]}"
        assert (
            z.shape[-1] == self.Z_SIZE
        ), f"Expected last dimension {self.Z_SIZE}, got {z.shape[-1]}"

        return self._simulate_fn(x, u, z, node=node)


    # ---- Environment Evaluation ---- #
    @staticmethod
    def simulate(fn: callable, x: jnp.ndarray, u: jnp.ndarray, z: jnp.ndarray, node = None) -> jnp.ndarray:
        """
        Simulate the environment dynamics and return the output vector.

        Parameters:
            - fn: The function to apply to the inputs (e.g. the graph function in Mu_F)
            - x: The state vector (node input)
            - u: The action vector (node output)
            - z: The uncertainty vector 
            - node: The current node (optional, for non-Markovian environments)
        Returns:
            - output: The output vector containing observations, constraint values, and costs
        """
        return fn(x, u, z)

    # ---- Rollout Methods ---- #
    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = 0
        self._initialise_env(self.cfg)
        return jnp.array(self.cfg.model.root_node_inputs)
    
    def step(self, u, v, z):
        """
        Step method to take action v given observation u.

        Parameters:
            - u : observations
            - v : actions

        Returns:
            - y : outputs
            - x : constraint spaces
        """
        
        output = self(u, v, z)

        y = self.F(output)
        x = self.G(output)
        reward = self.R(output)
        
        self._tick()
        term, trunc = self._termination_conditions(x)
        
        if term and not trunc: reward = 1000

        return  y, reward, term, trunc, {'constraint_values': x}
    
    def _termination_conditions(self, x):
        """Termination conditions for the environment"""
        test = [self._infeas_sign(x_i, self._feas_thresh) for x_i in x]
        if jnp.any(jnp.array(test)):
            term = True
            trunc = False
        elif self.current_step >= self.max_steps:
            term = True
            trunc = True
        else:
            term = False
            trunc = False

        return term, trunc

    def _tick(self):
        """Increment the current step"""
        self.current_step += 1
        return self.current_step
    
    def _build_cfg(self, **kwargs):
        """Build the configuration for the environment"""
        if 'cfg' in kwargs:
            cfg = kwargs['cfg']
            if not isinstance(cfg, DictConfig):
                cfg = OmegaConf.create(cfg)
        elif 'solve_date' in kwargs and 'solve_id' in kwargs:
            cfg = OmegaConf.load(OUTPUTS_DIR.format(solve_date=kwargs['solve_date'], solve_id=kwargs['solve_id']) + HYDRA_CONFIG_FILE)
        else:
            raise ValueError('No configuration provided for environment')
        return cfg

    def _set_infeas_sign(self):
        """Sets the notion of infeasibility"""
        if hasattr(self.cfg, 'samplers') and self.cfg.samplers.notion_of_feasibility == 'positive':
            self._infeas_sign = le
            self._feas_thresh = - self._feas_thresh
        else:
            self._infeas_sign = ge

    def _initialise_env(self, cfg):
        """Initialise the environment """
        model_cfg = cfg.model 
        self._feas_thresh = model_cfg.feas_thresh
        self._set_infeas_sign()
        return model_cfg
    
    def _set_dim(self, **kwargs):
        """Set the shape dict for the environment """
        for attr in ('X_SIZE', 'U_SIZE', 'F_SIZE', 'G_SIZE', 'Z_SIZE', 'L_SIZE', 'PHI_SIZE'):
            val = kwargs.get(attr, getattr(self.__class__, attr))
            setattr(self.__class__, attr, val)