"""Solves the problem as a monolithic NLP by multiple shooting."""
import logging

import pandas as pd
from jax import lax
import jax.numpy as jnp
import networkx as nx

from mu_F.direct.base import SolveDirect
from mu_F.solvers.septal import septal_monolithic_solver
from mu_F.direct.utils import *


class MultipleShooting(SolveDirect):
    """Monolithic NLP solver using a multiple-shooting transcription.

    Builds one decision vector spanning aux, design and inter-node input
    variables and adds defect equality constraints per edge, then hands the
    assembled problem to septal's monolithic SQP solver.

    """

    # ---- External Methods ----

    def __init__(self, cfg, G):
        super().__init__(cfg, G)
        self._solver = septal_monolithic_solver
        assert (
            cfg.formulation.lower() == "deterministic"
        ), "Stochastic optimisation is unsupported. Run in deterministic setting"

    def solve(self):
        """
        Solve the problem with the loaded solver and return septal's
        native SQPResult unchanged.
        """
        from dataclasses import replace as dc_replace

        problem_data = self._prepare_model(self.G)
        solver = self._load_solver()
        x0 = self._initial_point(problem_data)

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

    # ---- Private Methods ----

    def _initial_point(self, problem_data):
        """
        Pick the SQP start: midpoint, a single forward rollout, or an
        L1-screened Sobol-rollout multi-start (n_sobol_screen > 1).
        """
        s = self.cfg.solvers
        if str(getattr(s, "warm_start", "rollout")) != "rollout":
            return initial_guess(problem_data["var_bounds"])
        n_screen = int(getattr(s, "n_sobol_screen", 1))
        if n_screen > 1:
            return best_rollout_start(self.cfg, self.G, problem_data, n_screen,
                                      float(getattr(s, "screen_penalty", 1.0e3)))
        return forward_rollout_guess(self.cfg, self.G, problem_data["var_bounds"])

    def _prepare_model(self, graph):
        """
        Assemble the monolithic NLP (objective, constraints, bounds)
        from the node graph for the multiple-shooting transcription.
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

        # Variable layout: [aux | des_0, ..., des_n | inp_0, ..., inp_n]
        total_des = sum(graph.nodes[node]["n_design_args"] for node in graph.nodes)
        total_inp = sum(graph.nodes[node]["n_input_args"] for node in graph.nodes)
        des_curr = n_aux
        inp_curr = n_aux + total_des

        # Per-edge coupling widths normalising defect residuals (None disables scaling)
        scale_per_edge = None
        if bool(getattr(self.cfg.solvers, "scale_variables", False)):
            graph = ensure_init_bounds(self.cfg, graph)
            scale_per_edge = edge_defect_scales(graph)

        for node in nx.topological_sort(graph):

            # Slicing for this node's design and input args
            n_des = graph.nodes[node]["n_design_args"]
            n_inp = graph.nodes[node]["n_input_args"]
            des_slice = des_curr, n_des
            des_curr += n_des

            # Root nodes take fixed config inputs; others slice from the decision vector
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
            eql_cons = build_equality_constraints(
                node, graph, inp_fn_map, inp_idx_map, eql_cons,
                scale_per_edge=scale_per_edge,
            )

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

    def _get_solution(self, result):
        """
        Return the decision variables and objective off the SQPResult.
        """
        return result.decision_variables, result.objective

    def _load_solver(self):
        """
        Return the loaded monolithic solver.
        """
        return self._solver

    def _get_status(self, result):
        """
        Return 1 if septal reports converged KKT, else 0.
        """
        return 1 if bool(result.success) else 0

    def _log_outputs(self, result):
        """
        Hand the septal SQPResult to the shared logger.
        """
        return log_outputs(self.cfg, self.G, result, bool(result.success))