"""
Classes for solving the problem as a monolithic NLP
"""

from functools import partial

from mu_F.solvers.functions import callable_casadi_nlp_optimizer_gcons
import jax.numpy as jnp
import networkx as nx
import logging
from jax import lax


class SolveMonolithic:
    def __init__(self, cfg, G):
        self.cfg = cfg
        self.G = G
        self.pos_feas = (
            True if cfg.samplers.notion_of_feasibility.lower() == "positive" else False
        )
        self._prepare_model(G)

        assert (
            cfg.formulation.lower() == "deterministic"
        ), "Stochastic optimistaion is unsupported. Run in deterministic setting"

    # --- Public Interface --- #
    def solve(self, problem_data, x0=None):
        """
        Solves the problem using the loaded solver
        """
        solver = self._load_solver()

        if x0 is None:
            x0 = _initial_guess(problem_data["var_bounds"])

        return solver(
            problem_data["objective_fn"],
            problem_data["constraints"],
            problem_data["var_bounds"],
            x0,
            problem_data["eq_rhs"],
            problem_data["eq_lhs"],
            
        )

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
            composed_eval[node] = evaluate_node(
                node_fn, input_fns, des_slice, aux_slice
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
        problem_data["constraints"] = make_constraints(constraints)
        problem_data["var_bounds"] = _get_bounds(self.cfg)
        problem_data["eq_rhs"] = jnp.zeros((n_g, 1))
        problem_data["eq_lhs"] = jnp.inf * jnp.ones((n_g, 1))
        problem_data["num_vars"] = curr_idx

        assert _idx_check(curr_idx, graph), (
            "The number of variables in the problem does not match the ",
            "number of design and aux args in the graph",
        )

        return problem_data

    def _load_solver(self):
        """
        Loads in solver object
        """
        return callable_casadi_nlp_optimizer_gcons


# -------------------------------------------------------------------------------- #
# ------------------------------- Core Functions --------------------------------- #
# -------------------------------------------------------------------------------- #


def evaluate_node(node_fn, in_fn, des_slice, aux_slice):
    des_0, des_len = des_slice
    aux_0, aux_len = aux_slice

    def node_eval(ctrl):
        des = _to_rank3(_slice_1d(ctrl, des_0, des_len))
        aux = _to_rank3(_slice_1d(ctrl, aux_0, aux_len))
        ins = in_fn(ctrl) if in_fn is not None else None
        return node_fn(des, ins, aux)

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

    logging.warning(
        "Solving the problem as a monolithic NLP. This is not core functionalitu, and is only intended for testing purposes."
    )

    solver = SolveMonolithic(cfg, G)
    problem_data = solver._prepare_model(G)
    solver, solution = solver.solve(problem_data)

    if solver.stats()['success']:
        logging.info(f"Optimal solution found at {solution['x']}, with cost: {solution['f']}")
    
    else:
        logging.warning("Solver failed to find an optimal solution.")
        logging.warning(f"Solver message: {solver.stats()['return_status']}")
    return
