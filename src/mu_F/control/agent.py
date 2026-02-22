"""
Base agent class implementation
"""
# Standard library imports
import logging
import os
import pickle
from collections import deque
from pathlib import Path
from typing import Optional, List
from functools import partial
from contextlib import (
    contextmanager, redirect_stdout, redirect_stderr
)

# Third party imports
import jax.numpy as jnp
from omegaconf import OmegaConf

# Local application imports
from mu_F.control.utils import ALL_CONSTANTS
from mu_F.constraints.casadi_evaluator import cost_to_go_evaluator as CTG_Surrogate
from mu_F.constraints.casadi_evaluator import current_constraint_evaluator as Constraint_Surrogate
from mu_F.control.utils import _trajectory_plots, _add_policy_to_plot

from graph.graph_assembly import build_graph_structure

# --- Global Constantss ---
globals().update(ALL_CONSTANTS)

# --- Context Helper ------
@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


class Agent:
    _node = 0
    _actions = deque()

    def __init__(self, solve_date: str, solve_id: str):
        # Loading graph config and setting output directory
        self._out_dir = OUTPUTS_DIR.format(solve_date=solve_date, solve_id=solve_id)
        self.cfg = OmegaConf.load(self._out_dir + HYDRA_CONFIG_FILE)
        
        # Init helper functions
        self._update_cfg()
        self.graph = self._load_pickle()
        self.ctg_network, self.constraint_surrogate = self._init_policy()



    # ---- Public methods ---- #
    def act(self, u: jnp.ndarray):
        """Take an action based on the current state u"""

        # If this is the root node, there are no observations, so we pass an empty array to the surrogate models
        if self._node == 0:
            u = jnp.empty((u.shape[0], u.shape[1], 0))

        # First try the Q-network
        with suppress_output():
            v, cost, status = self.q_network(node=self._node)(u[jnp.newaxis, :], None)
            cons = False

        # If Q-network optimisation fails for this node, fall back to constraint surrogate
        if not bool(jnp.all(status)):
            v, status = self.constraint_surrogate(node=self._node)(u[jnp.newaxis, :], None)
            cons = True

        # Logging the action taken
        logging.info((f"Action taken at node {self._node}: {v.T[0]}.",
                      f"Using {f'value function. With expected cost {cost}' if not cons else 'constraint surrogate'}"))

        # Internal state update
        self._actions.append((self._node, v.T))
        self._node += 1

        return v.T

    def add_policy_to_plot(self):
        return _add_policy_to_plot(
            self.cfg, self.graph, self._actions, self._out_dir
            )

    def plot_trajectories(self, action_names: Optional[List] = None):
        return _trajectory_plots(
            self.cfg, self._actions, action_names=action_names
            )


    # ---- Private methods ---- #

    def _load_pickle(self):

        target_file = PICKLE_FILE.format(
            case_study=self.cfg.case_study,
            i=0,
            n=0,
            mode=self.cfg.case_study.mode[0],
        )

        graph = pickle.load(open(Path(self._out_dir) / target_file, "rb"))

        return graph

    def _update_cfg(self):
        """Update the cfg with any new values provided at call time. 
        This is useful for updating the cfg with new surrogate parameters after training."""

        self.cfg = build_graph_structure(self.cfg)
        self.cfg.solvers.evaluation_mode.reward = POOL

        for key, value in CFG_UPDATE.items():
            setattr(self.cfg.solvers.forward_coupling, key, value)
        return None

    def _init_policy(self):
        # Initialise the surrogate models for the agent
        q_network = partial(
            CTG_Surrogate, cfg=self.cfg, graph=self.graph, pool=POOL
            )
        constraint_surrogate = partial(
            Constraint_Surrogate, cfg=self.cfg, graph=self.graph, pool=POOL
            )
        return q_network, constraint_surrogate