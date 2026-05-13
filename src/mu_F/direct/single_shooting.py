"""
Classes for solving the problem as a monolithic NLP by single shooting
"""


import logging

import pandas as pd
from jax import lax
import jax.numpy as jnp
import networkx as nx

from mu_F.direct.base import SolveDirect
from mu_F.solvers.septal import septal_monolithic_solver
from mu_F.direct.utils import *


class SingleShooting(SolveDirect):
    def __init__(self, cfg, G):
        super().__init__(cfg, G)
        self._solver = septal_monolithic_solver
        assert (
            cfg.formulation.lower() == "deterministic"
        ), "Stochastic optimistaion is unsupported. Run in deterministic setting"


    # --- Public Methods --- #

    def solve(self):
        """
        Solves the problem using the loaded solver.  Returns septal's native
        `SQPResult` unchanged — no adapter tuple.
        """
        from dataclasses import replace as dc_replace

        problem_data = self._prepare_model(self.G)
        solver = self._load_solver()
        x0 = initial_guess(problem_data["var_bounds"])

        if bool(getattr(self.cfg.solvers, "scale_variables", False)):
            s_obj, s_cons, s_bounds, s_x0, s_lhs, s_rhs, to_real = scale_problem(
                problem_data["objective_fn"], problem_data["constraints"],
                problem_data["var_bounds"], x0,
                problem_data["eq_lhs"], problem_data["eq_rhs"],
            )
            result = solver(
                s_obj, s_cons, s_bounds, s_x0, s_lhs, s_rhs,
                config=self._monolithic_sqp_config(),
            )
            result = dc_replace(
                result,
                decision_variables=to_real(jnp.asarray(result.decision_variables)),
            )
        else:
            result = solver(
                problem_data["objective_fn"],
                problem_data["constraints"],
                problem_data["var_bounds"],
                x0,
                problem_data["eq_lhs"],
                problem_data["eq_rhs"],
                config=self._monolithic_sqp_config(),
            )
        self._log_outputs(result)
        return result

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
    
    def _log_outputs(self, result):
        """Hand the septal `SQPResult` to the shared logger."""
        return log_outputs(self.cfg, self.G, result, bool(result.success))

    def _get_solution(self, result):
        """Return `(decision_variables, objective)` straight off the SQPResult."""
        return result.decision_variables, result.objective

    def _get_status(self, result):
        """`1` if septal reports converged KKT, else `0`."""
        return 1 if bool(result.success) else 0

    def _load_solver(self):
        """
        Loads in solver object
        """
        return self._solver
        

