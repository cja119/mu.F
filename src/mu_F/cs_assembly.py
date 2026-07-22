"""Assemble the case-study graph: nodes, edges, constraints, costs and solvers."""
from functools import partial
from itertools import chain
import jax.numpy as jnp
import numpy as np
import pandas as pd
from omegaconf import open_dict


from mu_F.unit_evaluators.constructor import UnitEvaluation, PostProcessEvaluation
from mu_F.constraints.functions import COST_holder, CS_holder, post_process_visualiser
from mu_F.graph.graph_assembly import GraphConstructor, MarkovGraphConstructor
from mu_F.graph.methods import CS_edge_holder, vmap_CS_edge_holder
from mu_F.surrogate.surrogate import Surrogate
from mu_F.constraints.constructor import ConstraintEvaluator
from mu_F.post_processes.constructor import PostProcessSamplingScheme, PostProcessLocalSipScheme
from mu_F.post_processes.methods import post_process_regressor_data_function
from mu_F.unit_evaluators.utils import arrhenius_kinetics_fn, arrhenius_kinetics_fn_2
from mu_F.visualisation.visualiser import Visualiser


def resolve_aux_block(cfg):
    """
    Derive every aux quantity from the single `aux` block, so the block is the
    only place an auxiliary is declared.  Each entry is one slot of the per-node
    aux vector; every node carries the whole block, so this is already the Markov
    template and needs no tiling.  A case study without an `aux` key is untouched.
    """
    specs = cfg.case_study.get('aux', None)
    if specs is None:
        return cfg

    types = [str(s['type']) for s in specs]
    unknown = set(types) - {'global_var', 'global_param', 'local_param'}
    if unknown:
        raise ValueError(f"Unknown aux type(s): {sorted(unknown)}")

    coupled = [i for i, t in enumerate(types) if t.startswith('global')]

    with open_dict(cfg.case_study):          # these keys are derived, not authored
        cfg.case_study.global_n_aux_args = len(specs)
        cfg.case_study.aux_var_type      = types
        cfg.case_study.aux_default       = [float(s['default']) for s in specs]
        cfg.case_study.KS_bounds.aux_args = [[list(s['bounds'])] for s in specs]
        cfg.case_study.n_aux_args = {'node_0': list(range(len(specs))), '(0,1)': coupled}

    # The aux slots are the tail of the DEUS sampling box, so their bounds come
    # from the block rather than being restated (and left to drift) alongside it.
    ext = cfg.case_study.get('extendedDS_bounds', None)
    if ext not in (None, 'None'):
        with open_dict(cfg.case_study):
            cfg.case_study.extendedDS_bounds = list(ext) + [list(s['bounds']) for s in specs]
    return cfg


def case_study_constructor(cfg):
    """
    Build and return the assembled case-study graph from the config.
    """
    cfg = resolve_aux_block(cfg)
    constraint_dictionary = CS_holder[cfg.case_study.case_study]
    cost_dictionary = COST_holder[cfg.case_study.case_study] if cfg.case_study.eval_cost else None

    # Edge functions: vmap'd variants when batched evaluation is enabled.
    if cfg.case_study.vmap_evaluations:
        dict_of_edge_fn = vmap_CS_edge_holder[cfg.case_study.case_study]
    else:
        dict_of_edge_fn = CS_edge_holder[cfg.case_study.case_study]

    # Markov-style case study: chained DAG of identical cfg-driven nodes.
    if cfg.case_study.get('make_markov', False):
        G = MarkovGraphConstructor(cfg)
    else:
        G = GraphConstructor(cfg, cfg.case_study.adjacency_matrix)

    # Dummy dataframe for the initial forward pass.
    init_df_samples = pd.DataFrame({col: np.zeros((2,)) for i,col in enumerate(cfg.case_study.design_space_dimensions)})

    G = case_study_allocation(G, cfg, dict_of_edge_fn, constraint_dictionary, solvers=solver_constructor(cfg, G), unit_params_fn=unit_params_fn(cfg, G), initial_forward_pass=init_df_samples, cost_dictionary=cost_dictionary)

    return G.get_graph()

def _process_bounds(cfg):
    """
    Split the extended design-space bounds into lower/upper row vectors.
    """
    raw_extended = cfg.case_study.extendedDS_bounds
    if raw_extended in (None, "None"):
        return "None"
    else:
        ext_arr = np.asarray(raw_extended)
        return [ext_arr[:, 0].reshape(1, -1), ext_arr[:, 1].reshape(1, -1)]
   

def case_study_allocation(G, cfg, dict_of_edge_fn, constraint_dictionary, solvers, unit_params_fn, initial_forward_pass, cost_dictionary=None):
    """
    Attach node, edge and graph attributes (constraints, costs, bounds,
    solvers, post-processing) onto the graph constructor.
    """
    # Node properties.
    G.add_arg_to_nodes('n_design_args', cfg.case_study.n_design_args)
    G.add_arg_to_nodes('n_theta', cfg.case_study.n_theta)
    G.add_arg_to_nodes('KS_bounds', cfg.case_study.KS_bounds.design_args)
    G.add_arg_to_nodes('parameters_best_estimate', cfg.case_study.parameters_best_estimate)
    G.add_arg_to_nodes('parameters_samples', cfg.case_study.parameters_samples)
    G.add_arg_to_nodes('fn_evals', cfg.case_study.fn_evals)
    G.add_arg_to_nodes('unit_op', cfg.case_study.unit_op)
    G.add_arg_to_nodes('unit_params_fn', unit_params_fn)
    G.add_arg_to_nodes('extendedDS_bounds', _process_bounds(cfg))
    G.add_arg_to_nodes('constraints', constraint_dictionary)

    if cost_dictionary is not None:
        G.add_arg_to_nodes('node_cost', cost_dictionary)

    if cfg.method != 'decomposition_constraint_tuner':
        n_nodes = cfg.case_study.num_nodes if cfg.case_study.get('make_markov', False) else len(cfg.case_study.adjacency_matrix)
        b_off = [0 for _ in range(n_nodes)]
        G.add_arg_to_nodes('constraint_backoff', b_off)

    # Input / auxiliary argument counts and indices.
    G.add_n_input_args(cfg.case_study.n_input_args)
    G.add_n_aux_args(cfg.case_study.n_aux_args)
    G.add_input_aux_indices()

    # Graph-level arguments.
    G.add_arg_to_graph('aux_bounds', cfg.case_study.KS_bounds.aux_args)
    G.add_arg_to_graph('n_aux_args', cfg.case_study.global_n_aux_args)
    G.add_arg_to_graph('initial_forward_pass', initial_forward_pass)
    G.add_arg_to_graph('solve_post_processing_problem', False)
    G.add_arg_to_graph('post_process_decision_indices', cfg.reconstruction.post_process_decision_indices if hasattr(cfg, 'reconstruction') else [])
    # Defaults so downstream evaluators always find keys even when post_process is disabled.
    G.add_arg_to_graph('solve_post_processing_problem', False)
    G.add_arg_to_graph('post_process_decision_indices', cfg.reconstruction.post_process_decision_indices if hasattr(cfg, 'reconstruction') else [])
    G.add_arg_to_graph('solve_post_processing_problem', False)
    if cfg.case_study.eval_cost:
        n_nodes_eval = cfg.case_study.num_nodes if cfg.case_study.get('make_markov', False) else len(cfg.case_study.adjacency_matrix)
        G.add_arg_to_graph('n_design_args', cfg.case_study.n_design_args * n_nodes_eval)
        G.add_arg_to_graph('bounds', list(chain.from_iterable([cfg.case_study.KS_bounds.design_args]* n_nodes_eval + cfg.case_study.KS_bounds.aux_args)))
    else:
        G.add_arg_to_graph('n_design_args', sum(cfg.case_study.n_design_args))
        G.add_arg_to_graph('bounds', list(chain.from_iterable(cfg.case_study.KS_bounds.design_args + cfg.case_study.KS_bounds.aux_args)))

   
    G.add_arg_to_graph('classifier_x_scalar', None)  # initialisation
    # Dummy scalarising classifiers so jit tracing has a callable to compile.
    G.add_arg_to_graph('post_process_classifier', lambda x: jnp.sum(x))
    G.add_arg_to_graph('post_process_lower_classifier', lambda x: jnp.sum(x))
    if cfg.reconstruction.post_process:
        G.add_arg_to_nodes('post_process_constraints', post_process_visualiser[cfg.case_study.case_study])
        G.add_arg_to_graph('post_process', PostProcessSamplingScheme if cfg.reconstruction.post_process_sampler 
                           else PostProcessLocalSipScheme)
        G.add_arg_to_graph('post_process_training_methods', Surrogate)
        # TODO: allow flexibility between sampling scheme and local SIP scheme
        G.add_arg_to_graph('post_process_solver_methods',
                           {'upper_level_solver': partial(ConstraintEvaluator, node=None, constraint_type=cfg.reconstruction.post_process_solver.upper_level), 'lower_level_solver': partial(ConstraintEvaluator, node=None, constraint_type=cfg.reconstruction.post_process_solver.lower_level)} if cfg.reconstruction.post_process_sampler 
                           else {'relaxation_a_solver': partial(ConstraintEvaluator, node=None, constraint_type=cfg.reconstruction.post_process_solver.relaxation_a), 'relaxation_b_solver': partial(ConstraintEvaluator, node=None, constraint_type=cfg.reconstruction.post_process_solver.relaxation_b)})
        G.add_arg_to_graph('post_process_decision_indices', cfg.reconstruction.post_process_decision_indices)
        G.add_arg_to_graph('solve_post_processing_problem', False)  # overwritten in the post_process function
        G.add_arg_to_graph('post_process_solution_evaluator', partial(PostProcessEvaluation, constraint_evaluator=ConstraintEvaluator))
        G.add_arg_to_graph('post_process_solution_visualiser', Visualiser)
        if cfg.surrogate.post_process_lower.model_class == 'regression':
            G.add_arg_to_graph('global_regressor_function',post_process_regressor_data_function[cfg.case_study.case_study])
    # Edge properties and auxiliary filters.
    G.add_arg_to_edges('edge_fn', dict_of_edge_fn)
    G.add_arg_to_edges('aux_filter', aux_filter(cfg, G))

    graph = G.get_graph()

    for node in graph.nodes:
        G.add_node_object(node, UnitEvaluation(cfg, graph, node), "forward_evaluator")

    return G


def unit_params_fn(cfg, G):
    """
    Per-node unit-parameter callables (e.g. Arrhenius kinetics) for the case study.
    """
    if cfg.case_study.case_study == 'batch_reaction_network' or (cfg.case_study.case_study == 'serial_mechanism_batch'):
        return {node: partial(arrhenius_kinetics_fn_2,Ea=jnp.array(cfg.model.arrhenius.EA[node]), R=jnp.array(cfg.model.arrhenius.R)) for node in G.G.nodes}
    elif cfg.case_study.case_study == 'serial_mechanism_batch':
        return {node: partial(arrhenius_kinetics_fn,Ea=jnp.array(cfg.model.arrhenius.EA[node]), A=jnp.array(cfg.model.arrhenius.A[node]), R=jnp.array(cfg.model.arrhenius.R)) for node in G.G.nodes}
    elif cfg.case_study.case_study in ['tablet_press', 'convex_estimator', 'estimator', 'convex_underestimator', 'affine_study', ]:
        return {node: lambda x, y: jnp.empty((0,)) for node in G.G.nodes}
    elif cfg.case_study.get('make_markov', False):
        return lambda x, y: jnp.empty((0,))
    else :
        raise ValueError('Invalid case study')
    

def aux_filter(cfg, G):
    """
    Per-edge filter trimming input_data_bounds to the successor's
    n_input_args columns; a no-op when the count already matches.
    """
    def make_filter(successor):
        n_inp = int(G.G.nodes[successor]['n_input_args'])

        def _filter(x):
            n_cols = int(x[0].shape[-1])
            if n_cols <= n_inp:
                return x
            return [x[0][:, :n_inp], x[1][:, :n_inp]]
        return _filter

    return {edge: make_filter(edge[1]) for edge in G.G.edges}

def solver_constructor(cfg, G):
    """
    Legacy hook — returned a dict of `SolverConstruction` factories per node.
    Retained as a no-op stub so call sites don't break; the dict is consumed
    into a graph attribute that no evaluator reads after Phase 3f.
    """
    return {
        'forward_coupling_solver':  {node: None for node in G.G.nodes},
        'backward_coupling_solver': {node: None for node in G.G.nodes},
    }

def make_markov(cfg):
    """
    Broadcast single-node case-study config across num_nodes for Markov chains.
    """
    if cfg.case_study.get('make_markov', False):
        cfg.case_study.parameters_best_estimate = [cfg.case_study.parameters_best_estimate for _ in range(cfg.case_study.num_nodes)]
        cfg.case_study.KS_bounds.design_args = [cfg.case_study.KS_bounds.design_args for _ in range(cfg.case_study.num_nodes)]
        cfg.case_study.design_space_dimensions = [f'{dim}_node_{i}' for i in range(cfg.case_study.num_nodes) for dim in cfg.case_study.design_space_dimensions]
        cfg.case_study.n_design_args = [cfg.case_study.n_design_args for _ in range(cfg.case_study.num_nodes)]
        cfg.case_study.n_input_args = [cfg.case_study.n_input_args for _ in range(cfg.case_study.num_nodes)]
        cfg.case_study.unit_op = [cfg.case_study.unit_op for _ in range(cfg.case_study.num_nodes)]
        cfg.case_study.extendedDS_bounds = [cfg.case_study.extendedDS_bounds for _ in range(cfg.case_study.num_nodes)]
        cfg.model.node_aux = [cfg.model.node_aux for _ in range(cfg.case_study.num_nodes)]
        cfg.case_study.n_aux_args = build_aux_args(cfg)
        cfg.case_study.process_space_names = [[f'n{i}_{name}'for name in cfg.case_study.process_space_names] for i in range(cfg.case_study.num_nodes)]
        cfg.samplers.unit_wise_target_reliability = [cfg.samplers.unit_wise_target_reliability for _ in range(cfg.case_study.num_nodes)]
    return cfg

def build_aux_args(cfg):
    """
    Expand per-node and per-edge auxiliary-argument counts across the chain.
    """
    n_aux_per_node = cfg.case_study.n_aux_args['node_0']
    n_aux_per_edge = cfg.case_study.n_aux_args['(0,1)']

    n_aux_args = {}
    for node in range(cfg.case_study.num_nodes):
        n_aux_args[f'node_{node}'] = n_aux_per_node

    for i in range(cfg.case_study.num_nodes-1):
        n_aux_args[f'({i},{i+1})'] = n_aux_per_edge
    return n_aux_args
