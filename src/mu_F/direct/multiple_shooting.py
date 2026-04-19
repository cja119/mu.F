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


class MultipleShooting(SolveDirect):
    def __init__(self, cfg, G):
        super().__init__(cfg, G)
        self._solver = callable_casadi_nlp_optimizer_mono
        assert (
            cfg.formulation.lower() == "deterministic"
        ), "Stochastic optimisation is unsupported. Run in deterministic setting"

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
        eql_cons = []
        composed_eval = {}
        inp_idx_map = {}
        inp_fn_map = {}

        # Slicing for the global aux args
        n_aux = graph.graph["n_aux_args"]
        aux_slice = 0, n_aux

        # Variable layout: [aux | des_0, des_1, ..., des_n | inp_0, inp_1, ..., inp_n]
        total_des = sum(graph.nodes[node]["n_design_args"] for node in graph.nodes)
        total_inp = sum(graph.nodes[node]["n_input_args"] for node in graph.nodes)
        des_curr = n_aux
        inp_curr = n_aux + total_des

        for node in nx.topological_sort(graph):

            # Slicing for this node's design and input args
            n_des = graph.nodes[node]["n_design_args"]
            n_inp = graph.nodes[node]["n_input_args"]
            des_slice = des_curr, n_des
            des_curr += n_des

            # Root nodes take fixed inputs from config; all others slice from the decision vector
            if graph.in_degree(node) == 0:
                input_fn_or_slice = lambda ctrl, n=node: jnp.array(self.cfg.model.root_node_inputs[n]).reshape(1, 1, -1)
            else:
                input_slice = inp_curr, n_inp
                inp_curr += n_inp
                input_fn_or_slice = input_slice

            # Building the node evaluation function
            node_fn = graph.nodes[node]["forward_evaluator"].evaluate
            uncer = jnp.array(graph.nodes[node]["parameters_best_estimate"])
            composed_eval[node] = evaluate_node(node_fn, input_fn_or_slice, des_slice, aux_slice, uncer)

            # ------- Constraint Functions -------
            cons = list(graph.nodes[node]["constraints"].copy())
            cons_fns = process_constraints(cons, composed_eval[node], self.pos_feas, self.cfg)
            constraints.extend(cons_fns)

            # ------- Objective Function -------
            reward_extractor = make_reward_extractor(graph, node)
            reward_fn = compose(reward_extractor, composed_eval[node])
            rewards.append(reward_fn)

            # ------- Equality Constraints -------
            if graph.in_degree(node) > 0:
                inp_idx_map = input_index_map(node, graph, inp_idx_map, input_slice)
            inp_fn_map = build_input_fn(node, graph, composed_eval[node], inp_fn_map)
            eql_cons = build_equality_constraints(node, graph, inp_fn_map, inp_idx_map, eql_cons)

        curr_idx = inp_curr

        assert multiple_idx_check(curr_idx, graph), (
            "The number of variables in the problem does not match the ",
            "number of design, aux, and input args in the graph",
        )

        problem_data = {}

        n_g = len(constraints)
        n_e = total_inp  # one scalar constraint per input variable, but one fn per edge

        problem_data["objective_fn"] = make_objective(rewards)
        problem_data["constraints"] = constraints + eql_cons
        problem_data["var_bounds"] = get_bounds_ms(self.cfg, graph, total_inp)
        problem_data["eq_rhs"] = jnp.concatenate([jnp.inf * jnp.ones((n_g, 1)), jnp.zeros((n_e, 1))], axis=0)
        problem_data["eq_lhs"] = jnp.zeros((n_g+n_e, 1))
        problem_data["num_vars"] = curr_idx

        return problem_data

    def _get_solution(self, solver_output):
        """
        Extracts the solution from the solver output. This is where any necessary post-processing of the solution will be done.
        """
        return solver_output['x'], solver_output['f']

    def _load_solver(self):
        """
        Loads in solver object
        """
        return self._solver

    def _get_status(self, solver_output):
        """
        Extracts the status of the solution from the solver output. This is where any necessary post-processing of the status will be done.
        """
        return 1 if all(out >= 0 for out in solver_output['g'].nz) else 0
    
    def _log_outputs(self, solution, solver_stats=None):
        solved = bool(solver_stats.get("success", False)) if isinstance(solver_stats, dict) else self._get_status(solution)
        return log_outputs(self.cfg, self.G, solution, solved)