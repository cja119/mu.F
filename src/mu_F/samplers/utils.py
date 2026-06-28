"""Bounds construction and DEUS problem-description builders for the samplers."""
from omegaconf import DictConfig
import networkx as nx
import hydra
import logging
import jax.numpy as jnp
from jax.random import PRNGKey, choice


def _phases_setup_block(cfg: DictConfig, n_live, n_replacements):
    """
    Build DEUS's phases_setup from cfg.samplers.ns.phases toggles
    (INITIAL is unconditional; NMVP_SEARCH / DETERMINISTIC /
    PROBABILISTIC each carry a skip flag).
    """
    pcfg = cfg.samplers.ns.get('phases', None) or {}
    nmvp   = bool(pcfg.get('nmvp_search', False))
    determ = bool(pcfg.get('deterministic', True))
    prob   = bool(pcfg.get('probabilistic', True))

    schedule = [(0.00, n_live, n_replacements)]
    block = {
        "initial":       {"nlive": n_live, "nproposals": n_replacements},
        "nmvp_search":   {"skip": not nmvp},
        "deterministic": {"skip": not determ},
        "probabilistic": {"skip": not prob},
    }
    if determ:
        block["deterministic"]["nlive_change"] = {
            "mode": "user_given", "schedule": schedule,
        }
    if prob:
        block["probabilistic"]["nlive_change"] = {
            "mode": "user_given", "schedule": schedule,
        }
    return block


def _replacement_block(cfg: DictConfig):
    """
    Build DEUS's algorithms.replacement sub-tree from
    cfg.samplers.ns.rejector ('suob-ellipsoid' single ellipsoid, or
    'sumb-xmeans' x-means clustered ellipsoids).
    """
    rejector = getattr(cfg.samplers.ns, 'rejector', 'suob-ellipsoid')
    if rejector == 'suob-ellipsoid':
        return {"sampling": {"algorithm": "suob-ellipsoid"}}
    if rejector == 'sumb-xmeans':
        return {
            "clustering": {
                "algorithm": "clustering-xmeans",
                "settings": {"metric": "euclidean",
                             "score_criterion": "bic_ellipsoid"},
            },
            "sampling": {"algorithm": "sumb-mix"},
        }
    raise ValueError(f"Unrecognised rejector: {rejector!r}")


# ---------------------------------------------------------------------------
# Bounds construction
# ---------------------------------------------------------------------------

def design_list_constructor(bounds_for_design):
    """
    Construct a dict of bounds for the design space.
    """

    bounds = {}
    for i, bound in enumerate(bounds_for_design):
        if bound[0] not in (None, 'None'): bounds[f'd{i+1}'] = {f'd{i+1}': [bound[0], bound[1]]}

    return bounds

def extended_design_list_constructor(bounds_for_input, bounds_for_design):
    """
    Extend the design-space bounds with the coupling-input bounds
    (bounds_for_design is a nested list, bounds_for_input a dict).
    """
    if len(bounds_for_input) > 0:
        bounds = {}
        n_index = len(bounds_for_design)
        for j in range(len(bounds_for_input)): # iterate over input streams
            n_input_shape = max(bounds_for_input[j][0].shape)
            for i in range(n_input_shape): # iterate over input variables for each stream
                bounds[f'd{n_index+1}'] = {f'd{n_index+1}': [bounds_for_input[j][0][0,i].squeeze(), bounds_for_input[j][1][0,i].squeeze()]}
                n_index +=1
        return bounds_for_design | bounds
    else:
        return bounds_for_design

def add_global_aux(bounds, G, node=None):
    """
    Append the global aux bounds a node carries (its aux_ids slots); the full
    global set when node is None (the whole-network box).
    """
    n_index = len(bounds) if len(bounds) > 0 else 0

    aux_bounds = G.graph['aux_bounds']
    slots = G.nodes[node]['aux_ids'] if node is not None else range(len(aux_bounds))
    for j in slots: # only the global aux slots this node carries
        for n in aux_bounds[j]: # iterate over variables for each auxiliary arg
            if n[0] not in (None, 'None'):
                bounds[f'd{n_index+1}'] = {f'd{n_index+1}': [n[0], n[1]]}
                n_index +=1
    return bounds


def resolve_aux_override(cfg):
    """Read by the rollout and recovery: the pinned aux value, or None when the
    aux is to be optimised at the root node."""
    v = cfg.get('aux_override', None)
    return None if v in (None, 'None') else list(v)


def aux_optimised_at_root(cfg, graph, node):
    """True when the global aux is a free decision (no override) and this is the
    root node — the single place it is solved before being carried forward."""
    return resolve_aux_override(cfg) is None and graph.in_degree(node) == 0


def global_aux_bounds(graph, node):
    """Lower/upper bounds for the global aux slots a node carries, used to widen
    the recovery's free-decision box when the aux is optimised."""
    aux_bounds = graph.graph['aux_bounds']
    pairs = [n for j in graph.nodes[node]['aux_ids'] for n in aux_bounds[j]
             if n[0] not in (None, 'None')]
    lb = jnp.asarray([p[0] for p in pairs]).reshape(-1)
    ub = jnp.asarray([p[1] for p in pairs]).reshape(-1)
    return lb, ub


def get_unit_bounds(G: nx.DiGraph, unit_index: int):
    """
    Assemble design, coupling-input and aux bounds for a single unit.
    """
    if isinstance(G.nodes[unit_index]['extendedDS_bounds'], str) or unit_index == 0:
        design_var = design_list_constructor(G.nodes[unit_index]['KS_bounds'])
        if G.in_degree()[unit_index] > 0:
            bounds_for_input = [G.edges[predec,unit_index]['aux_filter'](G.edges[predec,unit_index]["input_data_bounds"]) for predec in G.predecessors(unit_index)]
            bounds =  extended_design_list_constructor(bounds_for_input, design_var) # operates on data already in the graph
        else:
            bounds = design_var
        bounds = add_global_aux(bounds, G, unit_index)
    else:
        bounds = { f'd{index+1}': {f'd{index+1}': [ G.nodes[unit_index]['extendedDS_bounds'][0][0,index],  G.nodes[unit_index]['extendedDS_bounds'][1][0,index]]} for index in range(len(G.nodes[unit_index]['extendedDS_bounds'][0].squeeze()))}

    return _sanitise_bounds_for_deus(bounds, unit_index)


def _sanitise_bounds_for_deus(bounds, unit_index, *, rel_eps=1.0e-6, abs_eps=1.0e-9):
    """
    Widen any collapsed (lb >= ub) interval by a tiny epsilon so DEUS's
    strict `lb < ub` check_problem passes.  Logs each widening so the
    underlying degeneracy isn't silently hidden.
    """
    for k, inner in bounds.items():
        for kk, v in list(inner.items()):
            lb, ub = float(v[0]), float(v[1])
            if lb < ub:
                continue
            mid   = 0.5 * (lb + ub)
            width = max(abs_eps, rel_eps * max(abs(lb), abs(ub), 1.0))
            new_lb, new_ub = mid - 0.5 * width, mid + 0.5 * width
            logging.warning(
                "Bounds collapsed at unit %s, dim %s: [%g, %g] → [%g, %g] "
                "(widened for DEUS).",
                unit_index, kk, lb, ub, new_lb, new_ub,
            )
            inner[kk] = [new_lb, new_ub]
    return bounds


def expand_bounds(bounds, frac, n_design):
    """
    Widen the first n_design decision intervals by frac of their range
    (frac/2 per side) so corner-living designs gain surrounding volume;
    coupling-input and aux dims are left unchanged.
    """
    if frac <= 0.0:
        return bounds
    widened = {}
    for i, (key, spec) in enumerate(bounds.items()):
        (inner, (lo, hi)), = spec.items()
        if i < n_design:
            delta = frac / 2.0 * (hi - lo)
            lo, hi = lo - delta, hi + delta
        widened[key] = {inner: [lo, hi]}
    return widened


def get_network_bounds(G: nx.DiGraph):
    """
    Concatenate every node's design bounds into a single network box.
    """

    bounds = []
    for node in G.nodes():
        bounds = bounds + G.nodes[node]['KS_bounds']

    dict_bounds = design_list_constructor(bounds)
    bounds = add_global_aux(dict_bounds, G)

    return dict_bounds


# ---------------------------------------------------------------------------
# DEUS problem-description builders
# ---------------------------------------------------------------------------

def create_problem_description_deus(cfg: DictConfig, the_model: object, G:nx.DiGraph, unit_index:float, forward_mode:bool = False):
    """
    Build the per-unit DEUS activity form (bounds, parameter scenarios,
    solver and sampling settings) for a single graph node.
    """

    bounds = get_unit_bounds(G, unit_index)
    bounds = expand_bounds(bounds, float(cfg.samplers.get('design_expansion', 0.0)),
                           int(G.nodes[unit_index]['n_design_args']))

    logging.info(f"Bounds: {bounds}")
    logging.info(f'EXTENDED DS DIM.: {len(bounds)}')

    if cfg.formulation == 'deterministic':
        parameter_samples = [{'c': jnp.array(G.nodes[unit_index]['parameters_best_estimate']).reshape(-1,), 'w': 1.0}]
    elif cfg.formulation == 'probabilistic':
        from omegaconf import OmegaConf
        raw_samples = OmegaConf.to_container(
            G.nodes[unit_index]['parameters_samples'], resolve=True,
        )
        # accepts standard (list of {'c','w'} scenarios) or compact (single dict
        # of parallel 'c'/'w' arrays) layouts
        parameter_samples = []
        for dict_ in raw_samples:
            w = dict_['w']
            if isinstance(w, (list, tuple)):
                for ci, wi in zip(dict_['c'], w):
                    parameter_samples.append({'c': [ci], 'w': float(wi)})
            else:
                parameter_samples.append({'c': dict_['c'], 'w': float(w)})
        parameter_samples = parameter_samples[:cfg.max_uncertain_samples]
        sum_weights = sum(param['w'] for param in parameter_samples)  # normalise to sum to 1
        for param in parameter_samples:
            param['w'] = param['w'] / sum_weights
    else:
        raise ValueError('Formulation not recognised')

    # probabilistic-surrogate path: model hands DEUS P_feas(d) directly instead
    # of per-scenario constraints (probabilistic formulation only)
    direct_probability = bool(
        cfg.samplers.deus.get('direct_probability', False)
        and cfg.formulation == 'probabilistic'
    )

    the_activity_form = {
        "activity_type": cfg.samplers.deus.activity_type,
        "activity_settings": {
            "case_name": f"{cfg.case_study.case_study}_{unit_index}_fwd_{forward_mode}",
            "case_path": hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            "resume": False, #' in general following the implementation here, we cannot load the problem via pickle
            "save_period": 1
        },


        "problem": {
            "user_script_filename": "none",
            "constraints_func_name": "none",
            "parameters_best_estimate": [el for el in G.nodes[unit_index]['parameters_best_estimate']],
            "parameters_samples": parameter_samples,
            "target_reliability": cfg.samplers.unit_wise_target_reliability[unit_index],
            "design_variables": [bound for bound in bounds.values()]
        },

        "solver": {
            "name": "dsc-ns",
            "settings": {
            "log_evidence_estimation": {"enabled": cfg.samplers.ns.log_evidence_estimation}, # must ve a boolean
            "score_evaluation": {
                "method": "serial",
                "score_type": "sigmoid",  # "indicator",
                #"constraints_func_ptr": the_model.g,
                # "constraints_func_ptr": None,
                "store_constraints": False,
            },
            # "score_evaluation": {
            #     "method": "mppool",
            #     "score_type": "sigmoid",  # "indicator",
            #     "pool_size": -1,
            #     "store_constraints": False
            # },
            "efp_evaluation": {
                "method": "serial",
                #"constraints_func_ptr": the_model.g,
                # "constraints_func_ptr": None,
                "store_constraints": False,
                "direct_probability": direct_probability,
            },
            #"efp_evaluation": {
            #    "method": "mppool",
            #    "pool_size": -1,
            #    "store_constraints": False
            #},
            "phases_setup": _phases_setup_block(
                cfg, cfg.samplers.ns.n_live, cfg.samplers.ns.n_replacements,
            )
        },
        "algorithms": {
            "sampling": {
                "algorithm": "mc_sampling-ns_global",
                "settings": {
                     "nlive": cfg.samplers.ns.n_live,  # This is overridden by points_schedule
                     "nproposals": cfg.samplers.ns.n_replacements,  # This is overriden by points_schedule
                     "prng_seed": 1989,
                     "f0": cfg.samplers.ns.f0,
                     "alpha": cfg.samplers.ns.alpha,
                     "stop_criteria": [
                         {"max_iterations": 100000}
                     ],
                     "debug_level": 0,
                     "monitor_performance": False
                 },
                "algorithms": {
                    "replacement": _replacement_block(cfg)
                    }
                }
            }
        }
    }

    return the_activity_form


def get_network_uncertain_params(cfg):
    """
    Build the joint uncertain-parameter scenarios (and nominal point)
    for the whole network from the per-unit case-study samples.
    """
    param_dict = cfg.case_study.parameters_samples

    # uncertain parameters
    max_parameter_samples = min(cfg.max_uncertain_samples, min([len(param) for param in param_dict]))
    list_of_params, list_of_weights = {i: {} for i in range(len(param_dict)) }, {i: {} for i in range(len(param_dict)) }

    for i, param in enumerate(param_dict):
        for k in range(max_parameter_samples):
            list_of_params[i][k] = param[k]['c']
            list_of_weights[i][k] = param[k]['w']

    concat_params = [jnp.hstack([jnp.array(list_of_params[i][k]).reshape(1,-1) for i in range(len(param_dict))]) for k in range(max_parameter_samples)]
    prod_weights = [jnp.prod(jnp.hstack([jnp.array(list_of_weights[i][k]).reshape(1,1) for i in range(len(param_dict))])) for k in range(max_parameter_samples)]
    sum_prod_weights = jnp.sum(jnp.array(prod_weights))
    prod_weights = [prod_weights[i]/sum_prod_weights for i in range(max_parameter_samples)]

    list_ = []

    for i in range(max_parameter_samples):
        list_.append({'c': concat_params[i].squeeze(), 'w': prod_weights[i]})

    # nominal parameters
    nom_params = jnp.hstack([jnp.array(p).reshape(1,-1) for p in cfg.case_study.parameters_best_estimate]).squeeze()

    nom_p = [{'c': nom_params.squeeze(), 'w': 1.0}]

    return list_, nom_params, nom_p


def create_problem_description_deus_direct(cfg: DictConfig, G:nx.DiGraph):
    """
    Build the whole-network DEUS activity form (direct mode) over the
    concatenated network bounds and joint uncertain parameters.
    """

    bounds = get_network_bounds(G)
    uncertain_params, nom_params, n_p_samples = get_network_uncertain_params(cfg)

    if cfg.formulation == 'deterministic':
        parameter_samples = n_p_samples
    elif cfg.formulation == 'probabilistic':
        parameter_samples = uncertain_params
    else:
        raise ValueError('Formulation not recognised')

    logging.info(f"Bounds: {bounds}")
    logging.info(f'DS DIM.: {len(bounds)}')


    the_activity_form = {
        "activity_type": cfg.samplers.deus.activity_type,
        "activity_settings": {
            "case_name": f"{cfg.case_study.case_study}_direct_mode",
            "case_path": hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            "resume": False, #' in general following the implementation here, we cannot load the problem via pickle
            "save_period": 1
        },

        "problem": {
            "user_script_filename": "none",
            "constraints_func_name": "none",
            "parameters_best_estimate": [nom_params[i] for i in range(len(nom_params))],
            "parameters_samples": parameter_samples,
            "target_reliability": cfg.samplers.target_reliability,
            "design_variables": [bound for bound in bounds.values()]
        },

        "solver": {
            "name": "dsc-ns",
            "settings": {
            "log_evidence_estimation": {"enabled": cfg.samplers.ns.log_evidence_estimation},
            "score_evaluation": {
                "method": "serial",
                "score_type": "sigmoid",  # "indicator",
                #"constraints_func_ptr": the_model.g,
                # "constraints_func_ptr": None,
                "store_constraints": False
            },
            # "score_evaluation": {
            #     "method": "mppool",
            #     "score_type": "sigmoid",  # "indicator",
            #     "pool_size": -1,
            #     "store_constraints": False
            # },
            "efp_evaluation": {
                "method": "serial",
                #"constraints_func_ptr": the_model.g,
                # "constraints_func_ptr": None,
                "store_constraints": False,
            },
            #"efp_evaluation": {
            #    "method": "mppool",
            #    "pool_size": -1,
            #    "store_constraints": False
            #},
            "phases_setup": _phases_setup_block(
                cfg, cfg.samplers.ns.final_sample_live, cfg.samplers.ns.n_replacements,
            )
        },
        "algorithms": {
            "sampling": {
                "algorithm": "mc_sampling-ns_global",
                "settings": {
                     "nlive": cfg.samplers.ns.final_sample_live,  # This is overridden by points_schedule
                     "nproposals": cfg.samplers.ns.n_replacements,  # This is overriden by points_schedule
                     "prng_seed": 1989,
                     "f0": cfg.samplers.ns.f0,
                     "alpha": cfg.samplers.ns.alpha,
                     "stop_criteria": [
                         {"max_iterations": 100000}
                     ],
                     "debug_level": 0,
                     "monitor_performance": False
                 },
                "algorithms": {
                    "replacement": _replacement_block(cfg)
                    }
                }
            }
        }
    }

    return the_activity_form
