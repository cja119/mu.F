"""Builders and helpers for the monolithic single/multiple-shooting NLPs."""
import logging

import numpy as np
import pandas as pd
from jax import lax
import jax.numpy as jnp
import networkx as nx


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def evaluate_node(node_fn, inp_slice_or_fn, des_slice, aux_slice, uncer):
    """
    Wrap a node forward evaluation, slicing design/aux/input args out
    of the flat decision vector before calling it.
    """
    des_0, des_len = des_slice
    aux_0, aux_len = aux_slice
    inp_0, inp_len = inp_slice_or_fn if isinstance(inp_slice_or_fn, tuple) else (0, 0)

    def node_eval(ctrl):
        des = _slice_1d(ctrl, des_0, des_len).reshape(1, -1)   # (N=1, U)
        aux = _slice_1d(ctrl, aux_0, aux_len).reshape(1, -1)   # (N=1, A)
        unc = jnp.reshape(uncer, (1, -1))                      # (S=1, Z)
        if isinstance(inp_slice_or_fn, tuple):
            ins = _to_rank3(_slice_1d(ctrl, inp_0, inp_len))   # (N=1, S=1, F)
        elif callable(inp_slice_or_fn):
            ins = inp_slice_or_fn(ctrl)                        # (N=1, S=1, F)
        else:
            ins = None
        return node_fn(des, ins, aux, unc)

    return node_eval

def input_index_map(node, graph, index_map, input_slice):
    """
    Map each incoming edge to the decision-vector indices holding that
    node's input variables (used by the multiple-shooting defects).
    """
    inp_0, _ = input_slice

    index_map = index_map or {}

    for prec in graph.predecessors(node):

        prec_indices = graph.edges[prec, node]["input_indices"]
        index_map[(prec, node)] = [inp_0 + prec_idx for prec_idx in prec_indices]

    return index_map


def build_input_fn(node, graph, node_eval, fn_map):
    """
    Register, per outgoing edge, the composed edge function mapping this
    node's output to its successor's input.
    """
    fn_map = fn_map or {}

    for succ in graph.successors(node):
        edge_fn = graph.edges[node, succ]["edge_fn"]
        fn_map[(node, succ)] = compose(edge_fn, node_eval)

    return fn_map

def build_equality_constraints(node, graph, fn_map, index_map, eql_cons,
                               scale_per_edge=None):
    """
    Build the per-edge defect constraints edge_fn(prev) - x_input,
    optionally dividing by scale_per_edge so feasibility_tol applies
    in relative units.
    """
    eql_cons = eql_cons or []

    def cons_fn(edge_fn, inp_indices, scale_inp):
        if scale_inp is None:
            return lambda ctrl: (
                jnp.ravel(edge_fn(ctrl)) - _slice_index_1d(ctrl, inp_indices)
            ).reshape(-1, 1)
        return lambda ctrl: (
            (jnp.ravel(edge_fn(ctrl)) - _slice_index_1d(ctrl, inp_indices)) / scale_inp
        ).reshape(-1, 1)

    for prec in graph.predecessors(node):
        scale_inp = scale_per_edge.get((prec, node)) if scale_per_edge else None
        eql_cons.append(cons_fn(fn_map[(prec, node)], index_map[(prec, node)], scale_inp))

    return eql_cons


def input_functions(node, graph, cfg, composed_eval):
    """
    Compose the input function for a node from its predecessors' edge
    functions; root nodes take fixed inputs from config.
    """
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
    """
    Return a function summing a node's cost terms into the scalar
    reward contributed to the objective.
    """
    cost_fns = list(graph.nodes[node].get("node_cost", []))

    def node_cost_sum(node_output):
        vals = [jnp.ravel(cf(node_output)) for cf in cost_fns]
        return jnp.sum(jnp.concatenate(vals, axis=0)).reshape(1, 1)

    return node_cost_sum


def process_constraints(constraints, node_outs, pos_feas, cfg):
    """
    Apply the feasibility sign convention and compose each node
    constraint onto the node's output evaluation.
    """
    fns = []
    for cons in constraints:
        f_cons = _apply_feasibility(cons, pos_feas)

        def cons_outer(y, f_cons=f_cons):
            return jnp.ravel(f_cons(y, cfg))

        fns.append(compose(cons_outer, node_outs))

    return fns


def make_constraints(cons_fns):
    """
    Stack the individual constraint functions into one vector-valued
    callable over the decision vector.
    """
    def cons(ctrl):
        ctrl = jnp.ravel(ctrl)
        return jnp.concatenate([cf(ctrl) for cf in cons_fns], axis=0).reshape(-1, 1)

    return cons


def make_objective(reward_fns):
    """
    Sum the per-node reward functions into one scalar objective over
    the decision vector.
    """
    def obj(ctrl):
        ctrl = jnp.ravel(ctrl)
        vals = [jnp.sum(jnp.ravel(rf(ctrl))) for rf in reward_fns]
        return jnp.sum(jnp.array(vals)).reshape(1, 1)

    return obj


# ---------------------------------------------------------------------------
# Output and logging
# ---------------------------------------------------------------------------

def log_outputs(cfg, graph, result, solved):
    """
    Log the solved objective and per-node design variables, and write the
    recovered policy to graph nodes and an Excel file.
    """
    status = 'successfully' if solved else 'unsuccessfully'
    logging.info(
        f"Monolithic solver finished {status}, "
        f"objective value {float(result.objective):.6e}"
    )
    cols = list(cfg.case_study.design_space_dimensions)
    rollout_row = {c: float('nan') for c in cols}

    x_star = np.asarray(result.decision_variables).reshape(-1)

    # Decision layout is [aux | designs | inputs]: x_pos reads x_star past the aux
    # block, col_pos writes into the (aux-free) design-column order.
    x_pos, col_pos = graph.graph["n_aux_args"], 0
    for node in graph.nodes():
        n_des = graph.nodes[node]["n_design_args"]
        des_vals = x_star[x_pos:x_pos + n_des]
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

        # Assign positionally; process and design naming conventions differ after make_markov
        for idx, val in enumerate(des_vals):
            col_idx = col_pos + idx
            if col_idx < len(cols):
                rollout_row[cols[col_idx]] = float(val)

        x_pos += n_des
        col_pos += n_des

    rollout_df = pd.DataFrame([rollout_row])

    # Persist the shared aux block so a forward re-simulation of the policy
    # reproduces the optimised dynamics (read by the trajectory plot).
    n_aux = graph.graph["n_aux_args"]
    aux_vals = [float(v) for v in x_star[:n_aux]]
    for idx, val in enumerate(aux_vals):
        rollout_df[f"aux_{idx}"] = val
    graph.graph["aux_star"] = aux_vals

    fname = 'monolithic_policy.xlsx'
    rollout_df.to_excel(fname)
    logging.info(f"Saved monolithic policy ({len(cols)}-d, aux={aux_vals}) to {fname}")

    return None


def compose(outer_fn, inner_fn):
    """
    Functional composition helper used throughout the builders.
    """
    return lambda *args: outer_fn(inner_fn(*args))


def multiple_idx_check(curr_idx, graph):
    """
    Sanity check that the decision vector length matches aux, design
    and input args for the multiple-shooting layout.
    """
    total_des = sum([graph.nodes[node]["n_design_args"] for node in graph.nodes])
    total_inp = sum([len(graph.edges[prec, node]["input_indices"]) for node in graph.nodes for prec in graph.predecessors(node)])
    n_aux = graph.graph["n_aux_args"]
    return curr_idx == total_des + n_aux + total_inp


def single_idx_check(curr_idx, graph):
    """
    Sanity check that the decision vector length matches aux and
    design args for the single-shooting layout.
    """
    total_des = sum([graph.nodes[node]["n_design_args"] for node in graph.nodes])
    n_aux = graph.graph["n_aux_args"]
    return curr_idx == total_des + n_aux


# ---------------------------------------------------------------------------
# Bounds and scaling
# ---------------------------------------------------------------------------

def get_bounds(cfg):
    """
    Assemble the aux and design variable bounds for the single-shooting
    decision vector from the case study config.
    """
    design_bds = [
        bound for node in cfg.case_study["KS_bounds"]["design_args"] for bound in node
    ]
    aux_bds = [
        bound for node in cfg.case_study["KS_bounds"]["aux_args"] for bound in node
    ]

    if any([(None in b or "None" in b) for b in design_bds]):
        des_lbs = []
        des_ubs = []
    else:
        des_lbs = [i[0] for i in design_bds]
        des_ubs = [i[1] for i in design_bds]

    if any([(None in b or "None" in b) for b in aux_bds]):
        aux_lbs = []
        aux_ubs = []
    else:
        aux_lbs = [i[0] for i in aux_bds]
        aux_ubs = [i[1] for i in aux_bds]

    bounds = [aux_lbs + des_lbs, aux_ubs + des_ubs]

    return jnp.array(bounds)


# Fractional widening of the init operating-range coupling box. A coupling sits
# on its box bound iff its value equals an envelope edge (e.g. the H accumulator
# pinned by a fixed-input root); there the box-bound gradient is parallel to the
# defect gradient and LICQ fails, so the SQP stalls. The pad lifts couplings off
# the bounds; the defect scale stays the unpadded width, so conditioning is O(1).
_COUPLING_BOX_PAD = 0.25


def get_bounds_ms(cfg, graph, total_inp):
    """
    Extend the single-shooting bounds with per-edge input-variable
    bounds for the multiple-shooting decision vector.
    """
    from omegaconf import OmegaConf, ListConfig, DictConfig

    base = get_bounds(cfg)

    non_root = next(n for n in graph.nodes if graph.in_degree(n) > 0)
    n_inp_per = graph.nodes[non_root]["n_input_args"]
    reps = total_inp // n_inp_per

    coupling_box = _chainwide_coupling_box(graph)
    ext_set = graph.nodes[non_root]["extendedDS_bounds"] not in (None, "None")

    if coupling_box is not None:
        # Priority 1: extendedDS box, or the init operating range — shared with
        # the defect scale so the scaled coupling columns stay O(1).
        inp_lbs_per = jnp.array(coupling_box[0])
        inp_ubs_per = jnp.array(coupling_box[1])
        if not ext_set:
            # extendedDS is already an overshoot; the init envelope is tight, so pad it.
            pad = _COUPLING_BOX_PAD * jnp.maximum(inp_ubs_per - inp_lbs_per, 1e-8)
            inp_lbs_per = inp_lbs_per - pad
            inp_ubs_per = inp_ubs_per + pad
    elif getattr(cfg.case_study, "edge_clip", None) not in (None, "None"):
        # Priority 2: physical state-component limits declared in the case study.
        ec = cfg.case_study.edge_clip
        if isinstance(ec, (ListConfig, DictConfig)):
            ec = OmegaConf.to_container(ec, resolve=True)
        inp_lbs_per = jnp.array([p[0] for p in ec])
        inp_ubs_per = jnp.array([p[1] for p in ec])
    else:
        # No bounds known.  The monolithic SQP will be poorly conditioned.
        inp_lbs_per = -jnp.inf * jnp.ones(n_inp_per)
        inp_ubs_per =  jnp.inf * jnp.ones(n_inp_per)

    inp_lbs = jnp.tile(inp_lbs_per, reps)
    inp_ubs = jnp.tile(inp_ubs_per, reps)

    return jnp.concatenate([base, jnp.stack([inp_lbs, inp_ubs])], axis=1)


def ensure_init_bounds(cfg, graph):
    """
    Populate per-edge input_data_bounds for the monolithic: skip when
    extendedDS_bounds is pre-set or the bounds already exist, else run the
    forward init pass (the same one the decomposition uses).
    """
    from mu_F.decomposition import _extended_bounds_are_set
    in_edges = [e for e in graph.edges if graph.in_degree(e[1]) > 0]
    have = len(in_edges) > 0 and all("input_data_bounds" in graph.edges[e] for e in in_edges)
    if _extended_bounds_are_set(cfg) or have:
        return graph
    from mu_F.initialisation.methods import run_initialisation
    return run_initialisation(cfg, graph)


def edge_defect_scales(graph):
    """
    Per-edge defect scale = the chain-wide coupling operating-range width, the
    same box the input variables are scaled over (see get_bounds_ms) so the
    scaled defect columns stay O(1). Shared across edges; None if no box is known.
    """
    box = _chainwide_coupling_box(graph)
    if box is None:
        return None
    width = jnp.asarray(np.maximum(box[1] - box[0], 1e-8))
    edges = [(p, n) for n in graph.nodes for p in graph.predecessors(n)]
    return {e: width for e in edges}


def _chainwide_coupling_box(graph):
    """Chain-wide [lb, ub] per coupling state: the receiving node's extendedDS
    input slots when set, else the envelope of the init input_data_bounds across
    edges, else None. A state that collapses at one node keeps the range it has
    elsewhere, so the envelope removes the need for a separate degeneracy floor."""
    edges = [(p, n) for n in graph.nodes for p in graph.predecessors(n)]
    n = edges[0][1]
    n_inp = int(graph.nodes[n]["n_input_args"])

    ext = graph.nodes[n].get("extendedDS_bounds", None)
    if ext not in (None, "None"):
        return np.asarray(ext[0]).reshape(-1)[-n_inp:], np.asarray(ext[1]).reshape(-1)[-n_inp:]

    with_bounds = [e for e in edges if "input_data_bounds" in graph.edges[e]]
    if with_bounds:
        los = np.stack([np.asarray(graph.edges[e]["input_data_bounds"][0]).reshape(-1) for e in with_bounds])
        his = np.stack([np.asarray(graph.edges[e]["input_data_bounds"][1]).reshape(-1) for e in with_bounds])
        return los.min(axis=0), his.max(axis=0)
    return None


def _safe_scale_offset(lb, ub):
    """
    Per-element (scale, offset) for the [0, 1]^n affine transform,
    falling back to (1, 0) when a slot is unbounded or zero-width.
    """
    lb_arr = jnp.asarray(lb, dtype=jnp.float64).reshape(-1)
    ub_arr = jnp.asarray(ub, dtype=jnp.float64).reshape(-1)
    width = ub_arr - lb_arr
    finite = jnp.isfinite(lb_arr) & jnp.isfinite(ub_arr) & (width > 0)
    scale = jnp.where(finite, width, jnp.ones_like(width))
    offset = jnp.where(finite, lb_arr, jnp.zeros_like(lb_arr))
    return scale, offset


def scale_problem(objective, constraints, var_bounds, x0, eq_lhs, eq_rhs):
    """
    Wrap a monolithic NLP in scaled coordinates x_s = (x - lo) / (hi - lo),
    returning the scaled callables, bounds, guess and a to_real un-scaler.
    Unbounded slots pass through (scale 1, offset 0).
    """
    lb = jnp.asarray(var_bounds[0]).reshape(-1)
    ub = jnp.asarray(var_bounds[1]).reshape(-1)
    scale, offset = _safe_scale_offset(lb, ub)

    def to_real(x_s):
        return jnp.asarray(x_s).reshape(-1) * scale + offset

    def wrap(fn):
        return lambda x_s: fn(to_real(x_s))

    s_obj = wrap(objective)
    s_cons = [wrap(c) for c in constraints]

    s_lb = jnp.where(jnp.isfinite(lb), jnp.zeros_like(lb), lb)
    s_ub = jnp.where(jnp.isfinite(ub), jnp.ones_like(ub),  ub)
    s_x0 = (jnp.asarray(x0).reshape(-1) - offset) / scale

    return s_obj, s_cons, [s_lb, s_ub], s_x0, eq_lhs, eq_rhs, to_real


def initial_guess(bounds):
    """
    Midpoint initial guess, falling back to zero on unbounded slots.
    """
    midpoint = (bounds[0] + bounds[1]) / 2
    return jnp.where(jnp.isfinite(midpoint), midpoint, jnp.zeros_like(midpoint))


def forward_rollout_guess(cfg, graph, var_bounds, seed=None):
    """
    Physically-consistent warm start for multiple shooting: roll the chain
    forward from the root under the `seed` controls (mid-box by default) so each
    node's real output seeds the next node's shooting state, leaving the defects
    near zero.  Only aux/design slots of `seed` are read; inputs come from the roll.
    """
    lb = jnp.asarray(var_bounds[0]).reshape(-1)
    ub = jnp.asarray(var_bounds[1]).reshape(-1)
    base = initial_guess(var_bounds) if seed is None else jnp.asarray(seed).reshape(-1)
    aux = base[:graph.graph["n_aux_args"]].reshape(1, -1)

    total_des = sum(graph.nodes[n]["n_design_args"] for n in graph.nodes)
    des_curr, inp_curr = graph.graph["n_aux_args"], graph.graph["n_aux_args"] + total_des

    x0, outputs = base, {}
    for node in nx.topological_sort(graph):
        n_des = graph.nodes[node]["n_design_args"]
        des = _slice_1d(base, des_curr, n_des).reshape(1, -1)
        des_curr += n_des
        unc = jnp.reshape(jnp.asarray(graph.nodes[node]["parameters_best_estimate"]), (1, -1))

        if graph.in_degree(node) == 0:
            ins = _to_rank3(jnp.asarray(cfg.model.root_node_inputs[node]))
        else:
            n_inp = graph.nodes[node]["n_input_args"]
            node_input = _rollout_node_input(graph, node, outputs)
            x0 = x0.at[inp_curr:inp_curr + n_inp].set(node_input)
            inp_curr += n_inp
            ins = _to_rank3(node_input)

        outputs[node] = graph.nodes[node]["forward_evaluator"].evaluate(des, ins, aux, unc)

    return jnp.clip(jnp.where(jnp.isfinite(x0), x0, base), lb, ub)   # seed fallback per NaN slot


def best_rollout_start(cfg, graph, problem_data, n_screen, penalty):
    """
    Sobol-sample the controls, roll each forward to a near-feasible start and keep
    the lowest L1-merit one (objective + penalty * path-constraint violation), the
    same scoring the decomposition's Sobol screen uses.
    """
    from mu_F.solvers.utilities import generate_initial_guess

    var_bounds = problem_data["var_bounds"]
    seeds = generate_initial_guess(int(n_screen), int(problem_data["num_vars"]), var_bounds, seed=42)
    merit = _make_l1_merit(problem_data, penalty)

    best_x, best_score = None, jnp.inf
    for seed in seeds:
        x0 = forward_rollout_guess(cfg, graph, var_bounds, seed)
        score = float(merit(x0))
        if score < best_score:
            best_x, best_score = x0, score
    logging.info(f"Sobol-rollout screen: {n_screen} starts, best L1 merit={best_score:.4g}")
    return best_x


def _make_l1_merit(problem_data, penalty):
    """L1 merit obj + penalty * violation under the monolithic lhs <= g <= rhs bounds."""
    obj = problem_data["objective_fn"]
    cons = list(problem_data["constraints"])
    lhs = jnp.asarray(problem_data["eq_lhs"]).reshape(-1)
    rhs = jnp.asarray(problem_data["eq_rhs"]).reshape(-1)

    def merit(x):
        g = jnp.concatenate([jnp.ravel(c(x)) for c in cons])
        viol = (jnp.maximum(0.0, lhs - g) + jnp.maximum(0.0, g - rhs)).sum()
        return jnp.asarray(obj(x)).reshape(()) + float(penalty) * viol

    return merit


def _rollout_node_input(graph, node, outputs):
    """Place each predecessor's edge-mapped output into this node's input block."""
    node_input = jnp.zeros(graph.nodes[node]["n_input_args"])
    for prec in graph.predecessors(node):
        edge_out = jnp.ravel(graph.edges[prec, node]["edge_fn"](outputs[prec]))
        idxs = jnp.asarray(graph.edges[prec, node]["input_indices"])
        node_input = node_input.at[idxs].set(edge_out)
    return node_input


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _apply_feasibility(constraint, pos_feas):
    """
    Flip the constraint sign unless positive feasibility is requested.
    """
    if pos_feas:
        return constraint
    return lambda x: -constraint(x)


def _slice_index_1d(x, indices):
    """
    Gather the given indices from the flattened vector.
    """
    return jnp.take(jnp.ravel(x), jnp.array(indices))


def _slice_1d(x, start: int, length: int):
    """
    Take a contiguous slice from the flattened vector.
    """
    x = jnp.ravel(x)
    return lax.dynamic_slice(x, (start,), (length,))


def _to_rank3(v):
    """
    Promote an array to rank 3 by prepending singleton axes.
    """
    while v.ndim < 3:
        v = jnp.expand_dims(v, axis=0)
    return v


def _extend(fn, fn_new):
    """
    Concatenate the outputs of two functions along the last axis.
    """
    if fn is None:
        return fn_new
    return lambda *args: jnp.concatenate([fn(*args), fn_new(*args)], axis=-1)

