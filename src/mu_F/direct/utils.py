
import logging

import pandas as pd
from jax import lax
import jax.numpy as jnp
import networkx as nx


# ---- General utils ----
def evaluate_node(node_fn, inp_slice_or_fn, des_slice, aux_slice, uncer):
    des_0, des_len = des_slice
    aux_0, aux_len = aux_slice
    inp_0, inp_len = inp_slice_or_fn if isinstance(inp_slice_or_fn, tuple) else (0, 0)

    def node_eval(ctrl):
        des = _to_rank3(_slice_1d(ctrl, des_0, des_len))
        aux = _to_rank3(_slice_1d(ctrl, aux_0, aux_len))
        unc = _to_rank3(uncer)
        if isinstance(inp_slice_or_fn, tuple):
            ins = _to_rank3(_slice_1d(ctrl, inp_0, inp_len))
        elif callable(inp_slice_or_fn):
            ins = inp_slice_or_fn(ctrl)
        else:
            ins = None
        return node_fn(des, ins, aux, unc)

    return node_eval

def input_index_map(node, graph, index_map, input_slice):

    inp_0, _ = input_slice

    index_map = index_map or {}
    
    for prec in graph.predecessors(node):
        
        prec_indices = graph.edges[prec, node]["input_indices"]
        index_map[(prec, node)] = [inp_0 + prec_idx for prec_idx in prec_indices]

    return index_map
    
def build_input_fn(node, graph, node_eval, fn_map):

    fn_map = fn_map or {}

    for succ in graph.successors(node):
        edge_fn = graph.edges[node, succ]["edge_fn"]
        fn_map[(node, succ)] = compose(edge_fn, node_eval)

    return fn_map

def build_equality_constraints(node, graph, fn_map, index_map, eql_cons):

    eql_cons = eql_cons or []

    def cons_fn(edge_fn, inp_indices):
        return lambda ctrl: (jnp.ravel(edge_fn(ctrl)) - _slice_index_1d(ctrl, inp_indices)).reshape(-1, 1)

    for prec in graph.predecessors(node):
        eql_cons.append(cons_fn(fn_map[(prec, node)], index_map[(prec, node)]))

    return eql_cons


def input_functions(node, graph, cfg, composed_eval):

    input_fns = None

    for prec in sorted(graph.predecessors(node)):

        prec_eval = composed_eval[prec]
        edge_fn = graph.edges[prec, node]["edge_fn"]
        input_fn = compose(edge_fn, prec_eval)
        input_fns = _extend(input_fns, input_fn)

    if graph.in_degree()[node] == 0:
        input_fns = lambda ctrl: _to_rank3(jnp.array(cfg.model.root_node_inputs[node]))

    return input_fns


def make_reward_extractor(graph, node):

    cost_fns = list(graph.nodes[node].get("node_cost", []))

    def node_cost_sum(node_output):
        vals = [jnp.ravel(cf(node_output)) for cf in cost_fns]
        return jnp.sum(jnp.concatenate(vals, axis=0)).reshape(1, 1)

    return node_cost_sum


def process_constraints(constraints, node_outs, pos_feas, cfg):
    fns = []
    for cons in constraints:
        f_cons = _apply_feasibility(cons, pos_feas)

        def cons_outer(y, f_cons=f_cons):
            return jnp.ravel(f_cons(y, cfg))

        fns.append(compose(cons_outer, node_outs))

    return fns


def make_constraints(cons_fns):
    def cons(ctrl):
        ctrl = jnp.ravel(ctrl)
        return jnp.concatenate([cf(ctrl) for cf in cons_fns], axis=0).reshape(-1, 1)

    return cons


def make_objective(reward_fns):
    def obj(ctrl):
        ctrl = jnp.ravel(ctrl)
        vals = [jnp.sum(jnp.ravel(rf(ctrl))) for rf in reward_fns]
        return jnp.sum(jnp.array(vals)).reshape(1, 1)

    return obj


def log_outputs(cfg, graph, solution, solved):
    
    status = 'succesfully' if solved else 'unsuccessfully'
    logging.info(f"Monolithic solver finished {status}, objective value {solution['f']}")
    cols = list(cfg.case_study.design_space_dimensions)
    rollout_row = {c: float('nan') for c in cols}

    des_0 = 0
    for node in graph.nodes():
        n_des = graph.nodes[node]["n_design_args"]
        des_vals = solution['x'][des_0:des_0 + n_des].full().flatten()
        logging.info(f"Design variables for node {node}: {des_vals}")

        # Mirror _get_rollout_action_columns priority from integration.py
        process_names = cfg.case_study.process_space_names
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
        graph.nodes[node]["rollout_action"] = [float(v) for v in des_vals]
        graph.nodes[node]["rollout_action_columns"] = action_cols
        graph.nodes[node]["rollout_action_named"] = named

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


def compose(outer_fn, inner_fn):
    return lambda *args: outer_fn(inner_fn(*args))


def multiple_idx_check(curr_idx, graph):
    total_des = sum([graph.nodes[node]["n_design_args"] for node in graph.nodes])
    total_inp = sum([len(graph.edges[prec, node]["input_indices"]) for node in graph.nodes for prec in graph.predecessors(node)])
    n_aux = graph.graph["n_aux_args"]
    return curr_idx == total_des + n_aux + total_inp

def single_idx_check(curr_idx, graph):
    total_des = sum([graph.nodes[node]["n_design_args"] for node in graph.nodes])
    n_aux = graph.graph["n_aux_args"]
    return curr_idx == total_des + n_aux


def get_bounds(cfg):

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


def get_bounds_ms(cfg, total_inp):

    base = get_bounds(cfg)
    inp_lbs = -jnp.inf * jnp.ones(total_inp)
    inp_ubs = jnp.inf * jnp.ones(total_inp)

    return jnp.concatenate([base, jnp.stack([inp_lbs, inp_ubs])], axis=1)


def initial_guess(bounds):
    midpoint = (bounds[0] + bounds[1]) / 2
    return jnp.where(jnp.isfinite(midpoint), midpoint, jnp.zeros_like(midpoint))


def _apply_feasibility(constraint, pos_feas):
    if pos_feas:
        return constraint
    return lambda x: -constraint(x)

def _slice_index_1d(x, indices):
    return jnp.take(jnp.ravel(x), jnp.array(indices))

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

