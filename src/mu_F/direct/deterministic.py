"""
Classes for solving the problem as a monolithic NLP
"""


import logging

import pandas as pd
from jax import lax
import jax.numpy as jnp
import networkx as nx

from mu_F.direct.base import SolveDirect
from mu_F.solvers.functions import callable_casadi_nlp_optimizer_gcons


class DeterministicMonolithic(SolveDirect):
    def __init__(self, cfg, G):
        super().__init__(cfg, G)
        self._solver = callable_casadi_nlp_optimizer_gcons
        assert (
            cfg.formulation.lower() == "deterministic"
        ), "Stochastic optimistaion is unsupported. Run in deterministic setting"


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
            reward_fn = _compose(rwd, composed_eval[node])
            rewards.append(reward_fn)

        problem_data = {}

        n_g = len(constraints)

        problem_data["objective_fn"] = make_objective(rewards)
        problem_data["constraints"] = constraints #make_constraints(constraints)
        problem_data["var_bounds"] = _get_bounds(self.cfg)
        problem_data["eq_rhs"] = jnp.inf * jnp.ones((n_g, 1))
        problem_data["eq_lhs"] = jnp.zeros((n_g, 1))
        problem_data["num_vars"] = curr_idx

        assert _idx_check(curr_idx, graph), (
            "The number of variables in the problem does not match the ",
            "number of design and aux args in the graph",
        )

        return problem_data
    
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
            _initial_guess(problem_data["var_bounds"]),
            problem_data["eq_lhs"], 
            problem_data["eq_rhs"],
        )

        self._log_outputs(solution)
        return solver, solution
    
    def _log_outputs(self, solution):

        status = 'succesfully' if self._get_status(solution) else 'unsuccessfully'

        logging.info(f"Monolithic solver finished {status}, objective value {solution['f']}")

        cols = list(self.cfg.case_study.design_space_dimensions)
        rollout_row = {c: float('nan') for c in cols}

        des_0 = 0
        for node in self.G.nodes():
            n_des = self.G.nodes[node]["n_design_args"]
            des_vals = solution['x'][des_0:des_0 + n_des].full().flatten()
            logging.info(f"Design variables for node {node}: {des_vals}")

            # Mirror _get_rollout_action_columns priority from integration.py
            process_names = self.cfg.case_study.process_space_names
            node_dims = process_names[node] if isinstance(process_names, (list, tuple)) else process_names
            if not isinstance(node_dims, (list, tuple)):
                node_dims = [node_dims]
            if len(node_dims) == n_des:
                action_cols = [str(c) for c in node_dims]
            else:
                node_ds_cols = [c for c in cols if f"N{node+1}" in str(c)]
                if len(node_ds_cols) == n_des:
                    action_cols = node_ds_cols
                elif len(cols) == n_des:
                    action_cols = list(cols)
                else:
                    action_cols = [f"node_{node}_action_{i}" for i in range(n_des)]

            named = {col: float(val) for col, val in zip(action_cols, des_vals)}

            # Store on graph nodes so _rollout_policy_from_graph works on this graph
            self.G.nodes[node]["rollout_action"] = [float(v) for v in des_vals]
            self.G.nodes[node]["rollout_action_columns"] = action_cols
            self.G.nodes[node]["rollout_action_named"] = named

            # Assign positionally into rollout_row (process_space_names and
            # design_space_dimensions use different naming conventions after make_markov)
            for idx, val in enumerate(des_vals):
                col_idx = des_0 + idx
                if col_idx < len(cols):
                    rollout_row[cols[col_idx]] = float(val)

            des_0 += n_des

        rollout_df = pd.DataFrame([rollout_row])
        fname = 'monolithic_policy.xlsx'
        rollout_df.to_excel(fname)
        logging.info(f"Saved monolithic policy ({len(cols)}-d) to {fname}")

        return None

    def _get_solution(self, solver_output):
        return solver_output['x'], solver_output['f']

    def _get_status(self, solver_output):
        return 1 if all(out >= 0 for out in solver_output['g'].nz) else 0

    def _load_solver(self):
        """
        Loads in solver object
        """
        return self._solver

# -------------------------------------------------------------------------------- #
# ------------------------------- Core Functions --------------------------------- #
# -------------------------------------------------------------------------------- #


def evaluate_node(node_fn, in_fn, des_slice, aux_slice, uncer):
    des_0, des_len = des_slice
    aux_0, aux_len = aux_slice

    def node_eval(ctrl):
        des = _to_rank3(_slice_1d(ctrl, des_0, des_len))
        aux = _to_rank3(_slice_1d(ctrl, aux_0, aux_len))
        unc = _to_rank3(uncer)
        ins = in_fn(ctrl) if in_fn is not None else None
        return node_fn(des, ins, aux, unc)

    return node_eval


def input_functions(node, graph, cfg, composed_eval):

    input_fns = None

    for prec in sorted(graph.predecessors(node)):

        prec_eval = composed_eval[prec]
        edge_fn = graph.edges[prec, node]["edge_fn"]
        input_fn = _compose(edge_fn, prec_eval)
        input_fns = _extend(input_fns, input_fn)

    if graph.in_degree()[node] == 0:
        input_fns = lambda ctrl: _to_rank3(jnp.array(cfg.model.root_node_inputs[node]))

    return input_fns


def rwd(node_output):
    return node_output[..., -1]


def process_constraints(constraints, node_outs, pos_feas, cfg):
    fns = []
    for cons in constraints:
        f_cons = _apply_feasibility(cons, pos_feas)

        def cons_outer(y, f_cons=f_cons):
            return jnp.ravel(f_cons(y, cfg))

        fns.append(_compose(cons_outer, node_outs))

    return fns


def make_constraints(cons_fns):
    def cons(ctrl):
        ctrl = jnp.ravel(ctrl)
        return jnp.concatenate([cf(ctrl) for cf in cons_fns], axis=0).reshape(-1, 1)

    return cons


def make_objective(reward_fns):
    def obj(ctrl):
        ctrl = jnp.ravel(ctrl)
        vals = [rf(ctrl) for rf in reward_fns]
        return jnp.sum(jnp.concatenate(vals, axis=0)).reshape(1, 1)

    return obj


# -------------------------------------------------------------------------------- #
# ------------------------------ Helper Functions -------------------------------- #
# -------------------------------------------------------------------------------- #


def _apply_feasibility(constraint, pos_feas):
    if pos_feas:
        return constraint
    return lambda x: -constraint(x)


def _slice_1d(x, start: int, length: int):
    x = jnp.ravel(x)
    return lax.dynamic_slice(x, (start,), (length,))


def _to_rank3(v):
    while v.ndim < 3:
        v = jnp.expand_dims(v, axis=0)
    return v


def _extend(fn, fn_new):
    if fn is None:
        return fn_new
    return lambda *args: jnp.concatenate([fn(*args), fn_new(*args)], axis=-1)


def _compose(outer_fn, inner_fn):
    return lambda *args: outer_fn(inner_fn(*args))


def _idx_check(curr_idx, graph):
    total_des = sum([graph.nodes[node]["n_design_args"] for node in graph.nodes])
    n_aux = graph.graph["n_aux_args"]
    return curr_idx == total_des + n_aux


def _get_bounds(cfg):

    design_bds = [
        bound for node in cfg.case_study["KS_bounds"]["design_args"] for bound in node
    ]
    aux_bds = [
        bound for node in cfg.case_study["KS_bounds"]["aux_args"] for bound in node
    ]

    if any(["None" in b for b in design_bds]):
        des_lbs = []
        des_ubs = []
    else:
        des_lbs = [i[0] for i in design_bds]
        des_ubs = [i[1] for i in design_bds]

    if any(["None" in b for b in aux_bds]):
        aux_lbs = []
        aux_ubs = []
    else:
        aux_lbs = [i[0] for i in aux_bds]
        aux_ubs = [i[1] for i in aux_bds]

    bounds = [aux_lbs + des_lbs, aux_ubs + des_ubs]

    return jnp.array(bounds)


def _initial_guess(bounds):
    return (bounds[0] + bounds[1]) / 2


# -------------------------------------------------------------------------------- #
# ------------------------------- Core Interface  -------------------------------- #
# -------------------------------------------------------------------------------- #


def monolithic_problem(cfg, G, node, pool):
    """
    Solves the problem as a monolithic NLP
    """

    logging.warning((
        "Solving the problem as a monolithic NLP. This is not core functionality, ",
        "and is only intended for testing/diagnostic purposes."
    ))

    if cfg.formulation.lower() == "deterministic":
        solver = DeterministicMonolithic(cfg, G)
    else:
        raise NotImplementedError("Stochastic optimization is not (yet) supported in the monolithic formulation.")
    
    problem_data = solver._prepare_model(G)
    solver, solution = solver.solve(problem_data)

    if solver.stats()['success']:
        logging.info(f"Optimal solution found at {solution['x']}, with cost: {solution['f']}")
    
    else:
        logging.warning("Solver failed to find an optimal solution.")
        logging.warning(f"Solver message: {solver.stats()['return_status']}")
    return
