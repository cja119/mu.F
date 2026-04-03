"""
Classes for solving the problem as a monolithic NLP by single shooting
"""


import logging

import pandas as pd
from jax import lax
import jax.numpy as jnp
import networkx as nx

from mu_F.direct.base import SolveDirect
from mu_F.solvers.functions import callable_casadi_nlp_optimizer_mono
from mu_F.direct.utils import *


class SingleShooting(SolveDirect):
    def __init__(self, cfg, G):
        super().__init__(cfg, G)
        self._solver = callable_casadi_nlp_optimizer_mono
        assert (
            cfg.formulation.lower() == "deterministic"
        ), "Stochastic optimistaion is unsupported. Run in deterministic setting"


    # --- Public Methods --- #

    def solve(self):
        """
        Solves the problem using the loaded solver
        """
        problem_data = self._prepare_model(self.G)
        solver = self._load_solver()
        solver, solution =  solver(
            problem_data["objective_fn"],
            problem_data["constraints"],
            problem_data["var_bounds"],
            initial_guess(problem_data["var_bounds"]),
            problem_data["eq_lhs"], 
            problem_data["eq_rhs"],
        )

        self._log_outputs(solution, solver.stats())
        return solver, solution

    # --- Private Methods --- #
    
    def _prepare_model(self, graph):
        """
        Prepare the model for solving. We will build the model to be solved as a monolithic NLP.
        """

        constraints = []
        rewards = []
        composed_eval = {}

        # Slicing for the global aux args
        n_aux = graph.graph["n_aux_args"]
        aux_slice = 0, n_aux
        curr_idx = n_aux

        for node in nx.topological_sort(graph):

            # ------- Evaluation Vector -------

            # Slicing for the node's design args
            n_des = graph.nodes[node]["n_design_args"]
            des_slice = curr_idx, n_des
            curr_idx += n_des

            # Extracting input functions from predecessor nodes
            input_fns = input_functions(node, graph, self.cfg, composed_eval)

            # Building the node's evaluation function
            node_fn = graph.nodes[node]["forward_evaluator"].evaluate
            uncer = jnp.array(graph.nodes[node]["parameters_best_estimate"])
            composed_eval[node] = evaluate_node(
                node_fn, input_fns, des_slice, aux_slice, uncer
            )

            # ------- Constraint Functions -------
            cons = list(graph.nodes[node]["constraints"].copy())

            # Then process the constraint functions
            cons_fns = process_constraints(cons, composed_eval[node], self.pos_feas, self.cfg)

            # Extending the constraint functions with new funcitons
            constraints.extend(cons_fns)

            # ------- Objective Function -------
            reward_extractor = make_reward_extractor(graph, node)
            reward_fn = compose(reward_extractor, composed_eval[node])
            rewards.append(reward_fn)

        problem_data = {}

        n_g = len(constraints)

        problem_data["objective_fn"] = make_objective(rewards)
        problem_data["constraints"] = constraints #make_constraints(constraints)
        problem_data["var_bounds"] = get_bounds(self.cfg)
        problem_data["eq_rhs"] = jnp.inf * jnp.ones((n_g, 1))
        problem_data["eq_lhs"] = jnp.zeros((n_g, 1))
        problem_data["num_vars"] = curr_idx

        assert single_idx_check(curr_idx, graph), (
            "The number of variables in the problem does not match the ",
            "number of design and aux args in the graph",
        )

        return problem_data
    
    def _log_outputs(self, solution, solver_stats=None):
        solved = bool(solver_stats.get("success", False)) if isinstance(solver_stats, dict) else self._get_status(solution)
        return log_outputs(self.cfg, self.G, solution, solved)

    def _get_solution(self, solver_output):
        return solver_output['x'], solver_output['f']

    def _get_status(self, solver_output):
        return 1 if all(out >= 0 for out in solver_output['g'].nz) else 0

    def _load_solver(self):
        """
        Loads in solver object
        """
        return self._solver
        

