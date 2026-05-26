from abc import ABC

import pandas as pd
from jax.random import PRNGKey, choice

import jax.numpy as jnp
import numpy as np
import logging
import jax.profiler as profiler
from jax import clear_caches

import time
import gc
import os
import contextlib
import io


class _StdoutToLog(io.TextIOBase):
    """Pipe writes into a logger so DEUS prints land in main.log.

    DEUS calls bare `print(...)` from inside its solver loop (one line per
    iteration plus per-phase headers).  Redirecting stdout into this
    TextIOBase lets the iteration trace land in the hydra-managed log
    instead of polluting the terminal.
    """
    def __init__(self, logger):
        self.logger, self._buf = logger, ""
    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.logger.info(line.rstrip())
        return len(s)
    def flush(self):
        if self._buf.strip():
            self.logger.info(self._buf.rstrip())
        self._buf = ""


from mu_F.constraints.constructor import constraint_evaluator
from mu_F.unit_evaluators.constructor import subproblem_unit_wrapper
from mu_F.surrogate.surrogate import surrogate
from mu_F.surrogate.augmentation import augment_training_data
from mu_F.samplers.constructor import construct_deus_problem
from mu_F.samplers.appproximators import calculate_box_outer_approximation
from mu_F.samplers.utils import create_problem_description_deus
from deus import DEUS
from mu_F.utils import dataset_object as dataset_holder
from mu_F.utils import dataset as dataset
from mu_F.utils import data_processing as data_processor
from mu_F.utils import apply_feasibility
from mu_F.utils import save_graph


def _sqp_evaluators_for_node(graph, node):
    """`[(label, instance), ...]` for every SQP-running evaluator cached
    against `(graph, node)`.  Each evaluator carries the base-class
    `n_sqp_calls` / `n_sqp_failed` counters."""
    from mu_F.constraints.evaluators.backward              import _BACKWARD_EVALUATOR_CACHE
    from mu_F.constraints.evaluators.cost_to_go            import _CTG_EVALUATOR_CACHE
    from mu_F.constraints.evaluators.probability           import _BACKWARDPMAP_EVALUATOR_CACHE
    from mu_F.constraints.evaluators.forward               import _FORWARD_EVALUATOR_CACHE
    from mu_F.constraints.evaluators.forward_decentralised import _FWDDEC_EVALUATOR_CACHE

    g = id(graph)
    return [(label, ev) for label, ev in (
        ('bwd-cls', _BACKWARD_EVALUATOR_CACHE.get((g, node, False))),
        ('bwd-CTG', _CTG_EVALUATOR_CACHE.get((g, node))),
        ('prob',    _BACKWARDPMAP_EVALUATOR_CACHE.get((g, node))),
        ('fwd',     _FORWARD_EVALUATOR_CACHE.get((g, node))),
        ('fwd-dec', _FWDDEC_EVALUATOR_CACHE.get((g, node))),
    ) if ev is not None]


def _log_sqp_convergence(graph, node):
    """One-shot per-node summary of cumulative SQP non-convergence across
    every evaluator that participated in this node's DEUS sweep."""
    for label, ev in _sqp_evaluators_for_node(graph, node):
        n_calls, n_failed = int(ev.n_sqp_calls), int(ev.n_sqp_failed)
        if n_calls == 0:
            continue
        rate = n_failed / n_calls
        msg = (f"Node {node}: {label} — {n_failed}/{n_calls} SQPs did not "
               f"converge ({rate*100:.1f}%)")
        if rate >= 0.05:
            logging.warning(msg + " — consider bumping cfg.solvers.max_iter "
                                  "or cfg.solvers.n_starts.")
        else:
            logging.info(msg)


def _get_rollout_action_columns(cfg, node, n_actions):
    process_names = cfg.case_study.process_space_names
    node_dims = process_names[node] if isinstance(process_names, (list, tuple)) else process_names
    if not isinstance(node_dims, (list, tuple)):
        node_dims = [node_dims]

    if len(node_dims) == n_actions:
        return [str(c) for c in node_dims]

    ds_cols = list(cfg.case_study.design_space_dimensions)
    node_ds_cols = [c for c in ds_cols if f"N{node+1}" in str(c)]
    if len(node_ds_cols) == n_actions:
        return [str(c) for c in node_ds_cols]

    if len(ds_cols) == n_actions:
        return [str(c) for c in ds_cols]

    return [f"node_{node}_action_{i}" for i in range(n_actions)]



class apply_decomposition:
    def __init__(self, cfg, graph, precedence_order, mode:str="forward", iterate=0, max_devices=1, total_iterates=1):
        self.cfg = cfg
        self.graph = graph
        self.precedence_order = precedence_order
        self.mode = mode
        self.iterate = iterate
        self.max_devices = max_devices
        self.total_iterates = total_iterates

    def run(self):

        cfg, graph = self.cfg, self.graph
        precedence_order, mode, iterate, max_devices, total_iterates = self.precedence_order, self.mode, self.iterate, self.max_devices, self.total_iterates


        if mode == "backward" or mode == "forward-backward":
            nodes = reversed(precedence_order.copy())
        elif mode == "forward" or mode == "backward-forward" or mode == 'rollout':
            nodes = self.precedence_order.copy()
        else:
            raise ValueError(f"Mode {mode} not recognized. Please use 'forward', 'backward' or 'forward-backward'.")
        
        node_input = None
        r_cost = 0.0
        
        # Iterate over the nodes and apply nested sampling
        for node in nodes:
            logging.info(f'------- Characterising node {node} according to precedence order -------')
            model = subproblem_model(node, cfg, graph, mode=mode, max_devices=max_devices)
            if mode == 'rollout':
                if node_input is not None:
                
                    if node_input.ndim < 3: node_input = jnp.expand_dims(node_input, axis=0)

                    node_input = jnp.squeeze(node_input, axis=1) # Remove uncertainty dimensions for uncertain case
                    
                outputs, n_cost, decision = model.rollout(node_input)
                decision_vec = np.ravel(np.asarray(decision, dtype=float))
                action_cols = _get_rollout_action_columns(cfg, node, decision_vec.shape[0])
                graph.nodes[node]["rollout_action"] = decision_vec
                graph.nodes[node]["rollout_action_columns"] = action_cols
                graph.nodes[node]["rollout_action_named"] = {
                    col: float(value) for col, value in zip(action_cols, decision_vec)
                }
                
                if graph.out_degree(node) > 0: 
                    node_input = graph.edges[node, list(graph.successors(node))[0]]["edge_fn"](outputs) 
                
                r_cost += n_cost
                logging.info("Rollout cost thus far: " + str(r_cost))

            else:
                # create problem sheet according to cfg

                problem_sheet = create_problem_description_deus(cfg, model, graph, node, mode) 
                # solve extended DS using NS
                solver =  construct_deus_problem(DEUS, problem_sheet, model)
                with contextlib.redirect_stdout(_StdoutToLog(logging.getLogger())):
                    solver.solve()
                if cfg.method == 'decomposition_constraint_tuner': graph.nodes[node]['log_evidence'] = solver.get_log_evidence()
                feasible, infeasible = solver.get_solution()
                feasible_set, feasible_set_prob = feasible[0], feasible[1]
                if feasible_set.size == 0:
                    logging.warning(f"No feasible set found for node {node}. Terminating simulation.")
                    graph.graph['terminate'] = True
                    return graph
                # update the graph with the number of function evaluations
                graph.nodes[node]["fn_evals"] += model.function_evaluations
                # Augment training data around the converged feasibles BEFORE
                # process_data_forward so the augmented samples land in both
                # classifier_training and ctg_values via the same model.s()
                # pipeline DEUS used. No-op when augmentation.enabled is False.
                if mode != 'backward-forward':
                    augment_training_data(cfg, graph, node, model)
                # Per-node SQP convergence summary across every evaluator
                # that fired during this node's DEUS sweep (incl. augmentation).
                _log_sqp_convergence(graph, node)
                # estimate box for bounds for DS downstream
                process_data_forward(cfg, graph, node, model, feasible_set, mode)
                # train constraints for DS downstream using data now stored in the graph
                if (mode in ['forward'] and graph.out_degree(node) != 0) or (mode in ['backward'] and graph.in_degree(node) == 0): 
                    if cfg.surrogate.forward_evaluation_surrogate: surrogate_training_forward(cfg, graph, node)

                    # train reward function
                if cfg.case_study.eval_cost:
                    graph = get_ctg_training_data(graph, node, model, cfg)

                # classifier construction for current unit
                if (cfg.surrogate.classifier and mode != 'backward-forward'):
                    # Multi-head (K-cluster) classifier when rejector=sumb-xmeans;
                    # standard single-output classifier otherwise.
                    if cfg.samplers.ns.get('rejector', '') == 'sumb-xmeans':
                        cluster_classifier_construction(cfg, graph, node, iterate)
                    else:
                        classifier_construction(cfg, graph, node, iterate)
                if (cfg.surrogate.probability_map and mode != 'backward-forward'): probability_map_construction(cfg, graph, node, iterate) #  NOTE this is a study specific condition
                if (cfg.case_study.eval_cost and mode != 'backward-forward'): ctg_surrogate_construction(cfg, graph, node, iterate)

                del model, problem_sheet, solver, infeasible, feasible_set_prob, feasible_set, feasible
                
                
                save_graph(graph.copy(), mode + '_iterate_' + str(iterate)+ '_node_' + str(node))
                graph = del_data(graph, node)
                gc.collect()
                profiler.save_device_memory_profile(f"memory{node}.prof")
                clear_caches()
                profiler.save_device_memory_profile(f"memory{node}_post_backend_clear.prof")

        if mode == 'rollout':
            from mu_F.visualisation.methods import _rollout_policy_from_graph
            cols = list(cfg.case_study.design_space_dimensions)
            policy_vec = _rollout_policy_from_graph(cfg, graph)
            rollout_df = pd.DataFrame([dict(zip(cols, policy_vec))])
            fname = f'rollout_policy_iterate_{iterate}.xlsx'
            rollout_df.to_excel(fname)
            logging.info(f"Saved rollout policy ({len(cols)}-d) to {fname}: {dict(zip(cols, policy_vec))}")

        return graph

    
def del_data(graph, node):

    del graph.nodes[node]["classifier_training"]
    graph.nodes[node]["classifier_training"] = None

    for successor in graph.successors(node):
        if 'surrogate_training' in graph.edges[node, successor]:
            del graph.edges[node, successor]["surrogate_training"]
            graph.edges[node, successor]["surrogate_training"] = None
    return graph



def surrogate_training_forward(cfg, graph, node, iterate:int=0):
    """
    Train the surrogate model for the forward pass.
    - only train the node if it has successors
    - train the node with the input data from the current node
    - store the trained model in the graph

    """

    # Check if the node has successors
    if graph.out_degree()[node] == 0:
        return
    
    for successor in graph.successors(node):
        # train the model
        forward_evaluator_surrogate = surrogate(graph, node, cfg, ('regression', cfg.surrogate.regressor_selection, 'forward_evaluation_surrogate'), iterate, data_str='None') # data_str is None as forward_regressor data handling independent of data_str
        forward_evaluator_surrogate.fit(node=successor)
        # Always store the self-scaling variant: takes real-world inputs,
        # standardises internally with its own trained scaler.  Evaluator
        # code no longer needs to coordinate scalers across surrogates —
        # every callable on the graph speaks real units.
        query_model = forward_evaluator_surrogate.get_model('unstandardised_model')

        # store the trained model in the graph
        graph.edges[node, successor]["forward_surrogate"] = query_model
        graph.nodes[node]['x_scalar'] = forward_evaluator_surrogate.trainer.get_model_object('standardisation_metrics_input')
        graph.edges[node,successor]['y_scalar'] = forward_evaluator_surrogate.trainer.get_model_object('standardisation_metrics_output')
        graph.edges[node,successor]["forward_surrogate_serialised"] = forward_evaluator_surrogate.get_serailised_model_data()
    
    return query_model



def probability_map_construction(cfg, graph, node, iterate):
    """
    Construct the probability map for the forward pass.

    Parameters:
    cfg (object): The configuration object with a 'probability_map' attribute.
    graph (object): The graph object.
    node (object): The current node in the graph.

    Returns:
    None
    """
    # train the model
    ls_surrogate = surrogate(graph, node, cfg, ('regression', 'ANN', 'probability_map_surrogate'), iterate, data_str='probability_map_training')
    ls_surrogate.fit(node=None)
    # Always use the self-scaling variant; see `surrogate_training_forward`.
    query_model = ls_surrogate.get_model('unstandardised_model')
    
    # store the trained model in the graph
    graph.nodes[node]["probability_map"] = query_model
    graph.nodes[node]['probability_map_x_scalar'] = ls_surrogate.trainer.get_model_object('standardisation_metrics_input')
    graph.edges[node]['probability_map_y_scalar'] = ls_surrogate.trainer.get_model_object('standardisation_metrics_output')

    return

def get_classifier_data(graph, node, model, cfg):
    x_d, y_d = model.constraint_data.d[cfg.surrogate.index_on:], model.constraint_data.y[cfg.surrogate.index_on:]
    x_classifier, y_classifier, feasible_indices = apply_feasibility(x_d , y_d , cfg, node, cfg.formulation).get_feasible(return_indices = True)
    graph.nodes[node]["classifier_training"] = dataset(X=x_classifier, y=y_classifier) 

    return graph, feasible_indices

def get_probability_map_data(graph, node, model, cfg):
    x_d, y_d = model.probability_map_data.d[cfg.surrogate.index_on:], model.probability_map_data.y[cfg.surrogate.index_on:]
    x_classifier, y_classifier, feasible_indices = apply_feasibility(x_d , y_d, cfg, node, cfg.formulation).probabilistic_feasibility(return_indices = True)
    graph.nodes[node]["probability_map_training"] = dataset(X=x_classifier, y=y_classifier) 

    return graph

def process_data_forward(cfg, graph, node, model, live_set, mode, notion_of_feasibility='positive'):
    """
    Process the data in the forward direction.

    Parameters:
    cfg (object): The configuration object with a 'classification_threshold' attribute.
    graph (object): The graph object.
    node (object): The current node in the graph.
    model (object): The model object with 'input_output_data' and 'constraint_data' attributes.
    live_set (jnp.array): The live set data.

    Returns:
    None
    """
    # Select a subset of the data based on the classifier
    if (mode != 'backward-forward'):
        if cfg.surrogate.classifier:
            graph, feasible_indices = get_classifier_data(graph, node, model, cfg)
        elif cfg.surrogate.probability_map:
            graph = get_probability_map_data(graph, node, model, cfg)
        else:
            pass

        # Apply the selected function to the y data and store forward evaluations on the graph for surrogate training
        for successor in graph.successors(node):

            # --- apply edge function to output data --- #
            io_fn = graph.edges[node, successor]["edge_fn"]

            # --- select the approximation method
            if cfg.surrogate.forward_evaluation_surrogate:
                # Extract the input-output and classifier data from the model
                x_io, y_io, selected_y_io = data_processor(model.input_output_data, index_on = cfg.surrogate.index_on).transform_data_to_matrix(io_fn, feasible_indices) 
                if cfg.formulation == 'deterministic':
                    n_args = graph.nodes[node]['n_design_args'] + graph.nodes[node]['n_input_args'] + graph.nodes[node]['n_auxiliary_args']
                    x_io = x_io[:,:n_args]
            
            
                # --- apply the function to the selected output data --- #
                # ensure the output data is rank 2
                if y_io.ndim > 2: y_io= y_io.squeeze()
                if selected_y_io.ndim < 2: selected_y_io= selected_y_io.reshape(-1,1)
                # --- select the approximation method
                if cfg.samplers.ku_approximation == 'box': 
                    feasible_outer_approx = calculate_box_outer_approximation
                elif cfg.samplers.ku_approximation == 'ellipsoid':
                    raise NotImplementedError("Ellipsoid approximation not implemented yet.")

                # --- find box bounds on inputs
                graph.edges[node, successor][
                    "input_data_bounds"
                ] = feasible_outer_approx(selected_y_io, cfg, ndim=2)

                # store the forward evaluations on the graph for surrogate training
                forward_evals = dataset(X=x_io, y=y_io)
                graph.edges[node, successor]["surrogate_training"] = forward_evals 

                del x_io, y_io, selected_y_io, forward_evals
    # add live set to the node    
    graph.nodes[node]["live_set_inner"] = live_set    
    # Store the classifier data and the live set data to the node
    update_aux_bounds(live_set, graph, node, cfg)
    update_node_bounds_iplus1(graph, node, cfg)

    del live_set

    return

def update_aux_bounds(live_set, graph, node, cfg):
    # AUX SET IS THE INTERSECTION SO CAN DO THIS AFTER EACH NODE
    if cfg.case_study.n_aux_args[f'node_{node}'] != 0:
        aux = live_set[:,-cfg.case_study.n_aux_args[f'node_{node}']:]
        aux_bounds = calculate_box_outer_approximation(aux, cfg, ndim=2)
        graph.graph['aux_bounds'] = [[[aux_bounds[0][0,i], aux_bounds[1][0,i]]] for i in range(aux_bounds[0].shape[1])]
    else: pass

    return 


def transform_vmap_output(vmap_output):
    return jnp.vstack([vmap_output[:,i,:].squeeze() for i in range(vmap_output.shape[1])]) # transform rank3 tensor to rank 2 tensor ammenable for box approximation and surrogate training

def update_node_bounds_iplus1(graph, node, cfg):
    """
    Update the bounds of the node in the graph for iterate i+1 based on the liveset of the node at iterate i.

    Parameters:
    graph (object): The graph object.
    node (object): The current node in the graph.

    Returns:
    None
    """

    # Get the bounds of the node i
    new_bounds = calculate_box_outer_approximation(graph.nodes[node]["live_set_inner"], cfg, ndim=2)
    graph.nodes[node]['extendedDS_bounds'] = new_bounds

    return


def classifier_construction(cfg, graph, node, iterate):
    """
    Construct the classifier for the forward pass.

    Parameters:
    cfg (object): The configuration object with a 'classifier' attribute.
    graph (object): The graph object.
    node (object): The current node in the graph.

    Returns:
    classifier: The trained classifier. (-1 belongs to feasible region, 1 does not belong to feasible region)
    """
    # train the model
    ls_surrogate = surrogate(graph, node, cfg, ('classification', cfg.surrogate.classifier_selection, 'live_set_surrogate'), iterate, data_str='classifier_training')
    ls_surrogate.fit(node=node)

    serialised = ls_surrogate.get_serailised_model_data()

    if cfg.surrogate.classifier_selection == 'ANN':
        from mu_F.surrogate.nn_utils import build_ann
        query_model = build_ann(cfg, serialised, 'classifier')
    else:
        query_model = ls_surrogate.get_model('unstandardised_model')

    # store the trained model in the graph
    graph.nodes[node]["classifier"] = query_model
    graph.nodes[node]['classifier_x_scalar'] = ls_surrogate.trainer.get_model_object('standardisation_metrics_input')
    graph.nodes[node]['classifier_serialised'] = serialised

    del ls_surrogate

    return


def cluster_classifier_construction(cfg, graph, node, iterate):
    """Build a K-head ANN classifier from this node's training data.

    Single shared-trunk network with K linear output heads. Each head k is
    regressed to a ±1 target: −1 if the point is "in cluster k" (a feasible
    point assigned to cluster k by x-means), +1 otherwise. MSE loss across
    the K heads with per-(point, head) class-balanced weights.

    Clustering is x-means with the BIC-ellipsoid criterion — the same
    primitive DEUS uses during sumb-xmeans sampling, so the partition K
    reflects the sampler's view of feasibility geometry rather than a
    fixed K guess.

    Stored on the graph:
        graph.nodes[node]['cluster_classifier_head']      — callable f(x) → (K,)
        graph.nodes[node]['cluster_classifier_n_heads']   — K
    Convention: f(x)[k] ≤ 0 ⇔ "x is in cluster k".
    """
    from mu_F.surrogate.nn_utils import train_multihead_regressor

    # If the pre-augmentation x-means already found K=1, the natural
    # feasibility region is connected — multi-head would only be
    # modelling integer-corner augmentation artefacts.  Fall back to the
    # single-head binary classifier; `_resolve_classifier` in each
    # evaluator then routes through the standard `classifier` path.
    pre_aug_K = graph.nodes[node].get('pre_augmentation_n_clusters', None)
    if pre_aug_K is not None and pre_aug_K <= 1:
        logging.info(
            f"Cluster classifier: pre-augmentation x-means K={pre_aug_K} at "
            f"node {node} — feasibility is connected; using single-head classifier."
        )
        classifier_construction(cfg, graph, node, iterate)
        return

    ds = graph.nodes[node].get('classifier_training', None)
    if ds is None:
        logging.warning(f"Cluster classifier: no classifier_training data on node {node}")
        return

    X = np.asarray(ds.X)
    if X.ndim > 2:
        X = X.squeeze()
    y = np.asarray(ds.y).reshape(-1)
    # Positive-notion feasibility: min(g) >= 0 ⇔ all constraints satisfied.
    feas_mask = (y >= 0)
    X_feas = X[feas_mask]

    if X_feas.shape[0] < 2:
        logging.warning(f"Cluster classifier: too few feasible points at node {node}")
        return

    from mu_F.surrogate.augmentation import xmeans_cluster_indices
    cluster_indices = xmeans_cluster_indices(X_feas)
    n_clusters = len(cluster_indices)

    # Defence-in-depth: also short-circuit if the on-training-data x-means
    # finds K=1 (covers the augmentation-disabled path where pre_aug_K
    # isn't cached).
    if n_clusters <= 1:
        logging.info(
            f"Cluster classifier: x-means K={n_clusters} on classifier_training "
            f"at node {node} — using single-head classifier."
        )
        classifier_construction(cfg, graph, node, iterate)
        return
    cluster_labels = np.empty(X_feas.shape[0], dtype=int)
    for k, ids in enumerate(cluster_indices):
        cluster_labels[ids] = k

    # Per-row K-vector target. Feasible point in cluster k: -1 at slot k,
    # +1 elsewhere. Infeasible point: +1 everywhere ("not in any cluster").
    N = X.shape[0]
    Y = np.ones((N, n_clusters), dtype=np.float32)
    feas_idx = np.where(feas_mask)[0]
    for j, k in zip(feas_idx, cluster_labels):
        Y[j, k] = -1.0

    head_callable = train_multihead_regressor(
        cfg, X, Y, num_folds=int(cfg.surrogate.num_folds),
    )
    graph.nodes[node]['cluster_classifier_head']    = head_callable
    graph.nodes[node]['cluster_classifier_n_heads'] = n_clusters
    logging.info(f"Cluster classifier: trained {n_clusters}-head ANN at node {node}")




class subproblem_model(ABC):
    def __init__(self, unit_index, cfg, G, mode, max_devices):     
        """
        Class to construct the subproblem model for the DEUS solver.
        """
          
        # function evaluations, graph and unit-index intiialisation
        self.function_evaluations = 0
        self.unit_index = unit_index
        self.cfg, self.G = cfg, G

        # subproblem construction
        self.process_constraints = constraint_evaluator(cfg, G, unit_index, constraint_type='process')
        if cfg.case_study.eval_cost:
            self.node_costs = constraint_evaluator(cfg, G, unit_index, constraint_type='node_cost')
            self.backward_cost_to_go = constraint_evaluator(cfg, G, unit_index, constraint_type='backward_cost_to_go')
        if mode == 'forward':
            self.forward_constraints = constraint_evaluator(cfg, G, unit_index, constraint_type='forward')
            self.backward_constraints = None
            self.forward_decentralised = None
            self.root_node_constraint = None
        elif mode == 'rollout':
            self.rollout_costs = constraint_evaluator(cfg, G, unit_index, constraint_type='cost_rollout')
            self.rollout_constraints = constraint_evaluator(cfg, G, unit_index, constraint_type='constraint_rollout')
        elif mode == 'backward':
            self.backward_constraints = constraint_evaluator(cfg, G, unit_index, constraint_type='backward')
            self.forward_constraints = None
            self.forward_decentralised = None
            self.root_node_constraint = None
        elif (mode in ['forward-backward', 'backward-forward']):
            self.forward_constraints = constraint_evaluator(cfg, G, unit_index, constraint_type='forward')
            if self.cfg.method == 'decomposition_constraint_tuner':
                self.backward_constraints = None
                self.forward_decentralised = constraint_evaluator(cfg, G, unit_index, constraint_type='forward_decentralized')
                if mode == 'backward-forward':
                    self.root_node_constraint = constraint_evaluator(cfg, G, unit_index, constraint_type='root_node_decentralized')
                else:
                    self.root_node_constraint = None
            else:
                self.backward_constraints = constraint_evaluator(cfg, G, unit_index, constraint_type='backward')
                self.forward_decentralised = None
                self.root_node_constraint = None
        else:
            raise ValueError(f"Mode {mode} not recognized. Please use 'forward', 'backward', 'forward-backward' or 'backward-forward' .")
        
        # decentralised?

        # subproblem unit construction
        self.unit_forward_evaluator = subproblem_unit_wrapper(cfg, G, unit_index)
        if (cfg.method == 'decomposition_constraint_tuner') and (mode != 'backward'):
            self.unit_forward_evaluator.get_constraints = lambda d, p: None
        else:
            pass

        # dataset initialisation 
        self.input_output_data = None 
        self.constraint_data = None 
        self.ctg_values = None
        self.probability_map_data = None
        self.mode = mode
        self.max_devices = max_devices

    def determine_batches(self, data, batch_size):
        """ Method to determine the number of batches"""
        n_batches = data.shape[0] // batch_size
        if data.shape[0] % batch_size != 0:
            n_batches += 1
        return n_batches
    
    def evaluate_subproblem_batch(self, data, batch_size, p):
        """ Method to evaluate the subproblem in batches"""
        n_batches = self.determine_batches(data, batch_size)
        constraints = []
        for i in range(n_batches):
            batch = data[i*batch_size:(i+1)*batch_size,:]
            output = self.subproblem_constraint_evals(batch, p)
            constraints.append(output)
            del batch

            if (data.shape[0]>50) and (n_batches % (i+1) == 0): logging.info(f'Batch {i} of {n_batches} evaluated')

        return np.concatenate(constraints, axis=0)


    def subproblem_constraint_evals(self, d, p):
        # unit forward pass
        outputs = self.unit_forward_evaluator.get_constraints(d, p) # outputs (rank 3 tensor if we have parametric uncertainty in the unit, n_d \times n_theta \times n_g)

        # get design/inputs/aux parameters split
        unit_design, unit_inputs, aux_args = self.unit_forward_evaluator.get_auxilliary_input_decision_split(d) # decisions, inputs (both rank 2 tensors)

        # evaluate process constraints 
        if outputs is not None:
            process_constraint_evals = self.process_constraints.evaluate(unit_design, unit_inputs, aux_args, outputs) # process constraints (rank 3 tensor n_d \times n_theta \times n_g)
            node_cost_evals = self.node_costs.evaluate(unit_design, unit_inputs, aux_args, outputs) if self.cfg.case_study.eval_cost else None
        else:
            process_constraint_evals = None
            node_cost_evals = None

        # evaluate feasibility upstream
        if (self.forward_constraints is not None) and (self.G.in_degree(self.unit_index) > 0):
            start_time = time.time()
            forward_constraint_evals = self.forward_constraints.evaluate(unit_inputs, aux_args) # forward constraints (rank 3 tensor n_d \times n_theta \times n_g)
            end_time = time.time()
            execution_time = end_time - start_time
            del start_time, end_time, execution_time
            if forward_constraint_evals.ndim == 1:
                forward_constraint_evals = forward_constraint_evals.reshape(-1,1)
            if forward_constraint_evals.ndim == 2:
                forward_constraint_evals = np.expand_dims(forward_constraint_evals, axis=1)
            forward_constraint_evals = np.repeat(forward_constraint_evals, len(p), axis=1)
        else:
            forward_constraint_evals = None
        
        # evaluate feasibility downstream
        if (self.backward_constraints is not None) and (self.G.out_degree(self.unit_index) > 0):
            start_time = time.time()
            # backward returns `(evals, _)` — the second slot is reserved
            # for parity with older signatures and is now unconditionally
            # None.
            backward_constraint_evals, _ = self.backward_constraints.evaluate(outputs, aux_args) # backward constraints (rank 3 tensor, n_d \times n_theta \times n_g)
            end_time = time.time()
            execution_time = end_time - start_time
            if backward_constraint_evals.ndim == 1:
                backward_constraint_evals = backward_constraint_evals.reshape(-1,1)
            if backward_constraint_evals.ndim == 2:
                backward_constraint_evals = np.expand_dims(backward_constraint_evals, axis=1)

            logging.info(f"Backward constraint evaluation time for node {self.unit_index}: {execution_time:.4f} seconds")
        else:
            backward_constraint_evals = None

        # evaluate feasibility decentralised
        if self.forward_decentralised is not None and self.G.in_degree(self.unit_index) > 0:
            start_time = time.time()
            decentralised_constraint_evals = self.forward_decentralised.evaluate(unit_design, aux_args) 
            if decentralised_constraint_evals.ndim == 1:
                decentralised_constraint_evals = decentralised_constraint_evals.reshape(-1,1)
            if decentralised_constraint_evals.ndim == 2:
                decentralised_constraint_evals = np.expand_dims(decentralised_constraint_evals, axis=1)
            decentralised_constraint_evals= np.repeat(decentralised_constraint_evals, len(p), axis=1)
            end_time = time.time()
            execution_time = end_time - start_time
        else:
            decentralised_constraint_evals = None

        # evaluate root node feasibility 
        if self.root_node_constraint is not None and self.G.in_degree(self.unit_index) == 0:
            start_time = time.time()
            decentralised_root_constraint_evals = self.root_node_constraint.evaluate(unit_design, aux_args) 
            if decentralised_root_constraint_evals.ndim == 1:
                decentralised_root_constraint_evals = decentralised_root_constraint_evals.reshape(-1,1)
            if decentralised_root_constraint_evals.ndim == 2:
                decentralised_root_constraint_evals = np.expand_dims(decentralised_root_constraint_evals, axis=1)
            decentralised_root_constraint_evals= np.repeat(decentralised_root_constraint_evals, len(p), axis=1)
            end_time = time.time()
            execution_time = end_time - start_time
        else:
            decentralised_root_constraint_evals = None

        # Evaluate rewards
        if node_cost_evals is not None:
            if self.G.out_degree(self.unit_index) > 0:
                pre_ctg_cons = jnp.concatenate([c for c in [process_constraint_evals, forward_constraint_evals, backward_constraint_evals, decentralised_constraint_evals, decentralised_root_constraint_evals] if c is not None], axis=-1)
                _, _, cond = apply_feasibility([outputs], [pre_ctg_cons], self.cfg, self.unit_index, self.cfg.formulation).get_feasible(return_indices=True)
                feasible_mask = jnp.asarray(cond[0]).ravel()
                feasible_idx = jnp.where(feasible_mask)[0]

                start_time = time.time()
                if len(feasible_idx) > 0:
                    outputs_feasible = outputs[feasible_idx]
                    aux_feasible = aux_args[feasible_idx] if aux_args.shape[0] == outputs.shape[0] else aux_args
                    ctg_evals_feasible, ctg_success = self.backward_cost_to_go.evaluate(outputs_feasible, aux_feasible)
                    ctg_evals_flagged = jnp.where(ctg_success.reshape(-1), ctg_evals_feasible.reshape(-1), jnp.nan)
                    ctg_function_evals = jnp.full(outputs.shape[0], jnp.nan).at[feasible_idx].set(ctg_evals_flagged)
                else:
                    ctg_function_evals = jnp.full(outputs.shape[0], jnp.nan)
                if ctg_function_evals.ndim == 1:
                    ctg_function_evals = ctg_function_evals.reshape(-1,1)
                if ctg_function_evals.ndim == 2:
                    ctg_function_evals = jnp.expand_dims(ctg_function_evals, axis=1)
                ctg_target = node_cost_evals + self.cfg.case_study.discount_factor * ctg_function_evals
                end_time = time.time()
                execution_time = end_time - start_time
                logging.info(f"Cost-to-go evaluation time for node {self.unit_index}: {execution_time:.4f} seconds")
            else:
                ctg_target = node_cost_evals
            self.ctg_values = update_data(self.ctg_values, d, p, ctg_target)
        # update input output data for forward surrogate model
        # TODO check all this works, turn off data collection for forward on decentrlaised and check backoffs
        if (self.cfg.surrogate.forward_evaluation_surrogate and self.mode != 'backward-forward'):
            self.input_output_data = update_data(self.input_output_data, d, p, outputs)  # updating dataset for surrogate model of forward unit evaluation

        # concatenate constraint evaluations
        concat_obj = [process_constraint_evals, forward_constraint_evals, backward_constraint_evals, decentralised_constraint_evals, decentralised_root_constraint_evals]
        cons_g = jnp.concatenate([c for c in concat_obj if c is not None], axis=-1)  # return raw constraint values (n_d \times n_theta \times n_g)

        # storing classifier data and updating function evaluations
        if (self.cfg.surrogate.classifier and self.mode != 'backward-forward'):
            self.constraint_data = update_data(self.constraint_data, d, p, cons_g)  # updating dataset for surrogate model of forward unit evaluation
        if (self.cfg.surrogate.probability_map and self.mode != 'backward-forward'):
            self.probability_map_data = update_data(self.probability_map_data, d, p, self.SAA(cons_g))  # updating dataset for surrogate model of forward unit evaluation

        del process_constraint_evals, forward_constraint_evals, backward_constraint_evals, concat_obj, decentralised_constraint_evals

        return cons_g
    
    def s(self, d, p):
        # Snapshot SQP counters before this batch so the per-s() delta
        # can be reported without spamming per-iter.
        snapshot = [
            (label, ev, ev.n_sqp_calls, ev.n_sqp_failed)
            for label, ev in _sqp_evaluators_for_node(self.G, self.unit_index)
        ]

        g = self.evaluate_subproblem_batch(d, self.cfg.samplers.pmap_batch, p)

        n_theta, n_g = g.shape[-2], g.shape[-1]
        # adding function evaluations
        self.function_evaluations += g.shape[0]*g.shape[1]

        diag = []
        for label, ev, prev_calls, prev_failed in snapshot:
            d_calls  = ev.n_sqp_calls  - prev_calls
            d_failed = ev.n_sqp_failed - prev_failed
            if d_calls > 0:
                diag.append(f"{label} {d_failed}/{d_calls} ({d_failed/d_calls*100:.0f}%)")
        if diag:
            logging.info("[s] SQP non-conv: " + ", ".join(diag))

        # return information for DEUS
        return [g[i,:,:].reshape(n_theta,n_g) for i in range(g.shape[0])]
        
    def get_constraints(self, d, p):
        return self.s(d, p)

    def rollout(self, inputs, aux=None):

        # Edge handling if inputs or aux are None
        if inputs is None:
            inputs = jnp.empty((1,0))
        if aux is None:
            # Pull the rollout-time aux default from the case_study config.
            # When global_n_aux_args > 0 the case study must define `aux_default`
            # as a list whose length matches the global aux dimension; for
            # case studies with no global aux this is an empty list.
            aux_default = list(self.cfg.case_study.get('aux_default', []) or [])
            if len(aux_default) > 0:
                aux = jnp.tile(
                    jnp.asarray(aux_default, dtype=jnp.float32).reshape(1, -1),
                    (inputs.shape[0], 1),
                )
            else:
                aux = jnp.empty((inputs.shape[0], 0))

        # Make a decision
        decision, opt_ctg, opt_status = self.rollout_costs.evaluate(inputs, aux)

        # Concatenate decision, inputs and aux for constraint evaluation
        d = jnp.concatenate([decision, inputs, aux], axis=-1)
        p = self.get_uncertain_params()

        # Evaluate the graph, get constraints and rewards
        outputs = self.unit_forward_evaluator.get_constraints(d, p)
        p_cons = self.process_constraints.evaluate(decision, inputs, aux, outputs)
        n_cost = self.node_costs.evaluate(decision, inputs, aux, outputs)
        
        # Determine feasibility and status
        feasible = jnp.all(p_cons >= 0)
        status = 'feasible' if feasible else 'infeasible'
        opt_converged = 'converged' if bool(opt_status) else 'NOT_CONVERGED'

        logging.info(f"Rollout decision: {decision}, Cost-to-go: {opt_ctg}, Optimizer: {opt_converged}, Actual: {status}")

        return outputs, n_cost, decision

    def get_uncertain_params(self):
        
        if self.cfg.formulation == 'probabilistic':
            param_dict = self.cfg.case_study.parameters_samples
            if self.cfg.case_study.get('make_markov', False):
                param_dict = [param_dict for _ in range(self.cfg.case_study.num_nodes)]
            
            list_of_params = [jnp.array([p['c'] for p in param]) for param in param_dict]
            list_of_weights = [jnp.array([p['w'] for p in param]).reshape(-1) for param in param_dict]

            max_parameter_samples = self.cfg.max_uncertain_samples
            selected_params = [choice(PRNGKey(0), a, shape=(max_parameter_samples,), replace=True, p=weight, axis=0) for a, weight in zip(list_of_params, list_of_weights)]
        elif self.cfg.formulation == 'deterministic':
            param_best_estimate = self.cfg.case_study.parameters_best_estimate
            
            selected_params = [jnp.array([param]) for param in param_best_estimate]
        
 
        return selected_params[self.unit_index]
    
    def SAA(self, g):
        
        if self.cfg.samplers.notion_of_feasibility == 'positive':
            g_ = jnp.min(g, axis=-1).reshape(g.shape[0],g.shape[1])
            indicator = jnp.where(g_>=0, 1, 0)
        else:
            g_ = jnp.max(g, axis=-1).reshape(g.shape[0],g.shape[1])
            indicator = jnp.where(g_<=0, 1, 0)

        n_s = g.shape[1]
        prob_feasible = jnp.sum(indicator, axis=1)/n_s
            
        return prob_feasible.reshape(g.shape[0],1)
    
def update_data(data, *args):
    """ Method to update the data holder with new data"""
    if data is None:
        data = dataset_holder(*args)
    else:
        data.add(*args)

    return data

def ctg_surrogate_construction(cfg, graph, node, iterate):
    # train the model
    ctg_surrogate = surrogate(graph, node, cfg, ('regression', cfg.surrogate.regressor_selection, 'ctg_surrogate'), iterate)
    ctg_surrogate.fit(node=None)

    serialised = ctg_surrogate.get_serailised_model_data()

    if cfg.surrogate.regressor_selection == 'ANN':
        from mu_F.surrogate.nn_utils import build_ann
        query_model = build_ann(cfg, serialised, 'regressor')
    else:
        # Always use the self-scaling variant; see `classifier_construction`.
        query_model = ctg_surrogate.get_model('unstandardised_model')

    # store the trained model in the graph
    graph.nodes[node]["ctg_surrogate"] = query_model
    # Kept for diagnostics; no evaluator reads it under the new contract.
    graph.nodes[node]['ctg_surrogate_x_scalar'] = ctg_surrogate.trainer.get_model_object('standardisation_metrics_input')
    graph.nodes[node]['ctg_surrogate_serialised'] = serialised

    del ctg_surrogate

    return

def get_ctg_training_data(graph, node, model, cfg):
    x_d_list = model.ctg_values.d[cfg.surrogate.index_on:]
    y_ctg_list = model.ctg_values.y[cfg.surrogate.index_on:]
    y_d_list = model.constraint_data.y[cfg.surrogate.index_on:]

    # Apply feasibility per batch and collect feasible samples
    x_feasible_list = []
    y_feasible_list = []

    for x_batch, y_ctg_batch, y_d_batch in zip(x_d_list, y_ctg_list, y_d_list):
        _, _, feasible_indices = apply_feasibility([x_batch], [y_d_batch], cfg, node, cfg.formulation).get_feasible(return_indices=True)
        feasible_idx = jnp.asarray(feasible_indices[0]).ravel()
        # Convert boolean mask to integer indices if needed
        if feasible_idx.dtype == bool:
            feasible_idx = jnp.where(feasible_idx)[0]
        valid_mask = jnp.all(jnp.isfinite(y_ctg_batch[feasible_idx]), axis=-1).ravel()
        x_feasible_list.append(x_batch[feasible_idx][valid_mask])
        y_feasible_list.append(y_ctg_batch[feasible_idx][valid_mask])

    # Concatenate all feasible samples
    x_feasible = jnp.vstack(x_feasible_list) if x_feasible_list else jnp.empty((0, x_d_list[0].shape[1]))
    y_feasible = jnp.vstack(y_feasible_list) if y_feasible_list else jnp.empty((0, y_ctg_list[0].shape[1]))
    
    graph.nodes[node]["ctg_func_training"] = dataset(X=x_feasible, y=y_feasible)
    return graph
