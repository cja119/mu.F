"""Per-unit forward evaluators and the network simulator over the graph."""

from abc import ABC
from copy import copy
from functools import partial
import jax.numpy as jnp
from jax import vmap, jit
import numpy as np
import pandas as pd
import logging

from mu_F.unit_evaluators.integrators import unit_dynamics
from mu_F.unit_evaluators.steady_state import unit_steady_state
from mu_F.unit_evaluators.utils import arrhenius_kinetics_fn as arrhenius, RegressorData


class BaseUnit(ABC):
    """Abstract unit interface.

    Defines the decision-dependent-parameter and evaluate hooks that every
    concrete unit evaluator in this module implements.

    """

    # ---- External Methods ----

    def __init__(self, cfg, graph, node):
        self.cfg = cfg
        self.graph = graph
        self.node = node

    def get_decision_dependent_params(self, decisions):
        raise NotImplementedError

    def evaluate(self, decisions, x0):
        raise NotImplementedError

class UnitEvaluation(BaseUnit):
    """Forward evaluator for a single graph node.

    Wraps a UnitCfg holding the vmapped unit and decision-dependent-parameter
    functions, and is driven by the NetworkSimulator during a forward pass.

    """

    # ---- External Methods ----

    def __init__(self, cfg, graph, node):
        """
        Build the evaluator for a node and its associated UnitCfg.
        """
        super().__init__(cfg, graph, node)
        self.unit_cfg = UnitCfg(cfg, graph, node)


    def get_decision_dependent_params(self, decisions, uncertain_params=None):
        """
        Return the decision-dependent parameters for the given decisions.
        """
        return self.unit_cfg.decision_dependent_params(decisions, uncertain_params)

    def evaluate(self, design_args, input_args, aux_args, uncertain_params=None):
        """
        Evaluate the unit over the design / input / aux / uncertainty batch.
        Falls back to a row-by-row pass to isolate a failing case on error.
        """

        dd_params = self.get_decision_dependent_params(design_args, uncertain_params)
        dd_params = expand_dims(dd_params, axis=-1)
        input_args = expand_dims(input_args, axis=1)
        design_args = expand_dims(design_args, axis=1)
        aux_args = expand_dims(aux_args, axis=1)

        try:
            return self.unit_cfg.evaluator(
                design_args, input_args, aux_args, dd_params, uncertain_params,
            )
        except Exception as outer_exc:

            n_batch = int(design_args.shape[0])
            logging.error(
                "[unit_eval] vmapped evaluator failed; isolating offender "
                "row-by-row over %d rows.  Outer error: %r",
                n_batch, outer_exc,
            )
            for i in range(n_batch):
                d_i   = design_args[i:i+1]
                u_i   = input_args[i:i+1]
                a_i   = aux_args[i:i+1]
                dd_i  = dd_params[i:i+1]
                try:
                    self.unit_cfg.evaluator(d_i, u_i, a_i, dd_i, uncertain_params)
                except Exception as row_exc:
                    np.set_printoptions(suppress=True, precision=6, linewidth=200)
                    logging.error(
                        "[unit_eval] FAILING ROW %d / %d\n"
                        "  design_args      = %s\n"
                        "  input_args       = %s\n"
                        "  aux_args         = %s\n"
                        "  dd_params        = %s\n"
                        "  uncertain_params = %s\n"
                        "  row exception    = %r",
                        i, n_batch,
                        np.asarray(d_i).squeeze(),
                        np.asarray(u_i).squeeze(),
                        np.asarray(a_i).squeeze(),
                        np.asarray(dd_i).squeeze(),
                        np.asarray(uncertain_params).squeeze(),
                        row_exc,
                    )
                    raise
            # no single row reproduced the failure; re-raise the outer error
            raise

    def evaluate_diagonal(self, design_args, input_args, aux_args, uncertain_params):
        """
        Diagonal evaluation: row i uses realisation i, scenario axis collapsed.
        uncertain_params is (N, param_dim); returns (N, 1, n_out).
        """
        dd_params = self.unit_cfg.decision_dependent_params_diag(design_args, uncertain_params)
        outputs = self.unit_cfg.evaluator_diag(
            design_args, input_args, aux_args, dd_params, uncertain_params,
        )
        return jnp.expand_dims(outputs, axis=1)


def expand_dims(array, axis):
    """Promote an array to at least 3 dims for the batched evaluators."""
    if array.ndim < 3:
        array = jnp.expand_dims(array, axis=axis)
    return array


class SubproblemUnitWrapper(UnitEvaluation):
    """Single-node evaluator exposed to the subproblem solvers.

    Splits a flat decision vector into design / input / aux blocks, supplies
    root-node defaults, and returns the node constraints for DEUS / rollouts.

    """

    # ---- External Methods ----

    def __init__(self, cfg, graph, node):
        """
        Build the wrapper for a node, delegating to UnitEvaluation.
        """
        super().__init__(cfg, graph, node)

    def get_constraints(self, decisions, uncertain_params=None):
        """
        Split the decision vector and evaluate the node constraints over the
        uncertainty set (broadcasting a scenario axis onto the inputs).
        """
        if uncertain_params is None:
            uncertain_params = jnp.empty((1,1))
        
        design_args, input_args, aux_args = self.get_auxilliary_input_decision_split(decisions)
        
        # if no inputs to the unit, use the root node inputs or add empty array
        if input_args.shape[1] == 0: 
            if not (self.cfg.model.root_node_inputs[self.node] == 'None'):
                input_args = jnp.array([self.cfg.model.root_node_inputs[self.node]]*design_args.shape[0])
            else:
                input_args = jnp.empty((design_args.shape[0], 0))
        # if no aux to the unit, use the root node aux or add empty array
        if aux_args.shape[1] == 0:
            if not (self.cfg.model.node_aux[self.node] == 'None'):
                aux_args = jnp.array([self.cfg.model.root_node_aux[self.node]]*design_args.shape[0])
            else:
                aux_args = jnp.empty((design_args.shape[0], 0))
            
        input_args = expand_input_args(input_args, uncertain_params)
        #aux_args = expand_input_args(aux_args, uncertain_params)

        return self.evaluate(design_args, input_args, aux_args, uncertain_params)

    def get_constraints_rollout(self, decisions, uncertain_params):
        """
        Rollout forward pass with parameters paired one-per-trajectory.
        Row i is trajectory i's realisation; returns (N, 1, n_out) with no
        scenario-axis broadcast.
        """
        design_args, input_args, aux_args = self.get_auxilliary_input_decision_split(decisions)
        n = design_args.shape[0]
        if input_args.shape[1] == 0:
            if not (self.cfg.model.root_node_inputs[self.node] == 'None'):
                input_args = jnp.array([self.cfg.model.root_node_inputs[self.node]] * n)
            else:
                input_args = jnp.empty((n, 0))
        if aux_args.shape[1] == 0:
            if not (self.cfg.model.node_aux[self.node] == 'None'):
                aux_args = jnp.array([self.cfg.model.root_node_aux[self.node]] * n)
            else:
                aux_args = jnp.empty((n, 0))
        return self.evaluate_diagonal(design_args, input_args, aux_args, uncertain_params)

    def get_auxilliary_input_decision_split(self, decisions):
        """
        Split a flat decision vector into design / input / auxiliary blocks.
        """
        n_d = self.graph.nodes[self.node]['n_design_args']
        n_u = self.graph.nodes[self.node]['n_input_args']
        design_args, input_args, auxiliary_args = decisions[:,:n_d], decisions[:,n_d:n_d+n_u], decisions[:,n_d+n_u:]
        return design_args, input_args, auxiliary_args

def expand_input_args(array, template):
    """Broadcast input args across the uncertainty scenarios in template."""
    if array.ndim == 1:
        array = jnp.expand_dims(array, axis=1)
    if array.ndim < 3:
        array = jnp.expand_dims(array, axis=1)

    if array.shape[1] != template.shape[0]:
        array = jnp.concatenate([array for _ in range(template.shape[0])], axis=1) # repeat input_args for each uncertain param

    return array

class UnitCfg:
    """Builds the (optionally vmapped) evaluators for a single node.

    Selects the unit-operation function (dynamic / steady-state) and the
    decision-dependent-parameter function from the node's graph attributes,
    constructing both grid and diagonal variants for search and rollout.

    """

    # ---- External Methods ----

    def __init__(self, cfg, graph, node):
        """
        Build the grid and diagonal evaluators and the decision-dependent
        parameter functions from the node's graph attributes.
        """

        self.cfg, self.graph, self.node = cfg, graph, node

        # diagonal evaluators (rollout) built alongside the grid ones below.
        self.evaluator_diag = None
        self.decision_dependent_params_diag = None

        # if vmap is enabled in cfg, set the unit evaluation and decision dependent evaluation functions using vmap
        if cfg.case_study.vmap_evaluations:
            # --- set the unit evaluation fn
            if graph.nodes[node]['unit_op'] == 'dynamic':
                base = jit(partial(unit_dynamics, cfg=cfg, node=node, graph=graph))
            elif graph.nodes[node]['unit_op'] == 'steady_state':
                base = jit(partial(unit_steady_state, cfg=cfg, node=node, graph=graph))
            else:
                raise NotImplementedError(f'Unit corresponding to node {node} is a {graph.nodes[node]["unit_op"]} operation, which is not yet implemented.')
            # grid (search): one weather list shared across the design batch
            self.evaluator = vmap(vmap(base, in_axes=(0, 0, 0, 0, None), out_axes=0), in_axes=(None, 1, None, 1, 0), out_axes=1) # inputs are design args, input args, deicsion_and_uncertainty_dependent_params, uncertain params
            # diagonal (rollout): weather paired one-per-trajectory along the batch
            self.evaluator_diag = vmap(base, in_axes=(0, 0, 0, 0, 0), out_axes=0)

            # --- set the decision dependent evaluation
            fn = graph.nodes[node]['unit_params_fn']
            self.decision_dependent_params = vmap(vmap(fn, in_axes=(0, None), out_axes=0), in_axes=(None, 0), out_axes=1)
            self.decision_dependent_params_diag = vmap(fn, in_axes=(0, 0), out_axes=0)

        # if vmap is not enabled in cfg, set the unit evaluation and decision dependent evaluation functions without using vmap
        else: 
            # --- set the unit evaluation fn
            if graph.nodes[node]['unit_op'] == 'dynamic':
                self.evaluator = lambda x, y, z: jit(partial(unit_dynamics, cfg=cfg, node=node, graph=graph))(x.squeeze(), y.squeeze(), z.squeeze())
            elif graph.nodes[node]['unit_op'] == 'steady_state':
                self.evaluator = lambda x, y, z: jit(partial(unit_steady_state, cfg=cfg, node=node, graph=graph))(x.squeeze(), y.squeeze(), z.squeeze())
            else:
                raise NotImplementedError(f'Unit corresponding to node {node} is a {graph.nodes[node]["unit_op"]} operation, which is not yet implemented.')

            # --- set the decision dependent evaluation fn
            fn = graph.nodes[node]['unit_params_fn']
            self.decision_dependent_params = fn

        return
    
class NetworkSimulator(ABC):
    """Forward simulation of the process graph.

    Walks the nodes in order, propagating each unit's outputs along its edges
    and storing per-node constraints; subclasses specialise the evaluation
    mode (search, direct, post-process).

    """

    # ---- External Methods ----

    def __init__(self, cfg, graph, constraint_evaluator, type_cons='process'):
        self.cfg = cfg
        self.graph = graph.copy()
        self.type = type_cons
        self.constraint_evaluator = constraint_evaluator
        self.function_evaluations = {node: 0 for node in self.graph.nodes}
        self.desired_node_index = self.cfg.surrogate.post_process_lower.desired_node_index
        self.desired_regressor_data = RegressorData(cfg)

    def simulate(self, decisions, uncertain_params=None):
        """
        Walk the graph, evaluating each node and propagating its outputs along
        successor edges. Returns the per-node constraints and per-edge inputs.
        """
        u_p = None
        n_d = 0
        aux_args = decisions[:, sum([self.graph.nodes[node]['n_design_args'] for node in self.graph.nodes]):]
        
        for node in self.graph.nodes:
            if not (uncertain_params == None) :
                u_p = uncertain_params[node]
            

            if self.graph.in_degree()[node] == 0:
                if not (self.cfg.model.root_node_inputs[node] == 'None'):
                    inputs = jnp.tile(jnp.expand_dims(jnp.array([self.cfg.model.root_node_inputs[node]]).reshape(1,-1), axis=1), (decisions.shape[0], u_p.shape[0], 1))
                else:
                    inputs = jnp.empty((decisions.shape[0], u_p.shape[0], 0))
            else:
                inputs = jnp.concatenate([jnp.copy(self.graph.edges[predecessor, node]['input_data_store'])[:,:,:] for predecessor in self.graph.predecessors(node)], axis=-1)

            unit_nd = self.graph.nodes[node]['n_design_args']
            outputs = self.graph.nodes[node]['forward_evaluator'].evaluate(decisions[:, n_d:n_d+unit_nd], inputs, aux_args, u_p)
            
            for successor in self.graph.successors(node):
                edge_data = self.graph.edges[node, successor]['edge_fn'](jnp.copy(outputs))
                if edge_data.ndim==2: edge_data = jnp.expand_dims(edge_data, axis=-1)
                self.graph.edges[node, successor]['input_data_store'] = edge_data

            node_constraint_evaluator = self.constraint_evaluator(self.cfg, self.graph, node, constraint_type=self.type)
            self.graph.nodes[node]['constraint_store'] = node_constraint_evaluator.evaluate(decisions[:, n_d:n_d+unit_nd], inputs, aux_args, outputs)

            n_d += unit_nd

        # constraint evaluation, information for extended KS bounds
        return {node: self.graph.nodes[node]['constraint_store'] for node in self.graph.nodes}, {edge: self.graph.edges[edge[0],edge[1]]['input_data_store'] for edge in self.graph.edges}
    
    def get_constraints(self, decisions, uncertain_params=None):
        """
        Simulate the network and return the per-node constraints, tallying
        function evaluations and optionally collecting regressor data.
        """
        constraints, _ = self.simulate(decisions, uncertain_params)
        for node, g in constraints.copy().items():
            self.function_evaluations[node] += g.shape[0]*g.shape[1]
        if (self.cfg.surrogate.post_process_lower.model_class == 'regression') and (self.cfg.reconstruction.post_process):
            self.process_select_regressor_data(constraints, decisions)
        return constraints
    
    def get_extended_ks_info(self, decisions, uncertain_params=None):
        """
        Simulate the network and return the per-edge input data used for the
        extended KS bounds.
        """
        _, edge_data = self.simulate(decisions, uncertain_params)
        return edge_data

    def get_data(self, decisions, uncertain_params=None):
        """
        Simulate the network and return both per-node constraints and per-edge
        input data.
        """
        constraints, edge_data = self.simulate(decisions, uncertain_params)
        for node, g in constraints.items():
            self.function_evaluations[node] += g.shape[0]*g.shape[1]
        return constraints, edge_data
        
    
    def evaluate_direct(self, decisions, uncertain_params):
        """
        Direct-mode network walk that slices each node's uncertainty block from
        a flat parameter vector. Returns per-node constraints and per-edge inputs.
        """
        n_theta = [self.graph.nodes[node]['n_theta'] for node in self.graph.nodes]
        nu_pk = 0
        nu_pk_1 = 0
        n_d = 0
        aux_args = decisions[:, sum([self.graph.nodes[node]['n_design_args'] for node in self.graph.nodes]):]
        for node in self.graph.nodes:
            if not (uncertain_params.all() == None) :
                nu_pk = nu_pk_1 + n_theta[node]
                u_p = uncertain_params[:,nu_pk_1:nu_pk]
                if u_p.ndim == 1:
                    u_p = jnp.expand_dims(u_p, axis=1)

                nu_pk_1 = nu_pk


            if self.graph.in_degree()[node] == 0:
                if not (self.cfg.model.root_node_inputs[node] == 'None'):
                    inputs = jnp.array([self.cfg.model.root_node_inputs[node]]*decisions.shape[0])
                else:
                    inputs = jnp.empty((decisions.shape[0], u_p.shape[0], 0))
            else:
                inputs = jnp.concatenate([jnp.copy(self.graph.edges[predecessor, node]['input_data_store'])[:,:,:] for predecessor in self.graph.predecessors(node)], axis=-1)

            unit_nd = self.graph.nodes[node]['n_design_args']
            outputs = self.graph.nodes[node]['forward_evaluator'].evaluate(decisions[:, n_d:n_d+unit_nd], inputs, aux_args, u_p)
            
            for successor in self.graph.successors(node):
                edge_data = self.graph.edges[node, successor]['edge_fn'](jnp.copy(outputs))
                if edge_data.ndim==2: edge_data = jnp.expand_dims(edge_data, axis=-1)
                self.graph.edges[node, successor]['input_data_store'] = edge_data

            node_constraint_evaluator = self.constraint_evaluator(self.cfg, self.graph, node, constraint_type=self.type)

            self.graph.nodes[node]['constraint_store'] = node_constraint_evaluator.evaluate(decisions[:, n_d:n_d+unit_nd], inputs, aux_args, outputs)


            n_d += unit_nd

        # constraint evaluation, information for extended KS bounds
        return {node: self.graph.nodes[node]['constraint_store'] for node in self.graph.nodes}, {edge: self.graph.edges[edge[0],edge[1]]['input_data_store'] for edge in self.graph.edges}
    

    def direct_evaluate(self, decisions, uncertain_params):
        """
        Direct-mode evaluation returning the concatenated node constraints as a
        per-decision list (the form the direct solvers consume).
        """
        constraints, _ = self.evaluate_direct(decisions, uncertain_params)
        for node, g in constraints.items():
            self.function_evaluations[node] += g.shape[0]*g.shape[1]
        
        if (self.cfg.surrogate.post_process_lower.model_class == 'regression') and (self.cfg.reconstruction.post_process):
            self.process_select_regressor_data(constraints, decisions)

        cons_ = jnp.concatenate([cons for cons in constraints.values()], axis=-1)

        return [cons_[i,:,:] for i in range(cons_.shape[0])]

    def process_select_regressor_data(self, constraints, candidates):
        """
        Map the candidates and constraints through the global regressor
        function and append the result to the live regressor set.
        """
        fn = self.graph.graph['global_regressor_function']
        inputs, outputs = fn(candidates, constraints, self.desired_node_index)
        self.desired_regressor_data.append_to_live_set(inputs, outputs)
        return 



class PostProcessEvaluation(NetworkSimulator):
    """Post-processing pass over the network simulation.

    Extends NetworkSimulator to sweep the auxiliary space on a grid and map a
    fixed solution to its feasibility surface for plotting.

    """

    # ---- External Methods ----

    def __init__(self, cfg, graph, constraint_evaluator):
        super().__init__(cfg, graph, constraint_evaluator, type_cons='post_process_evals')
        self.type = 'post_process_evals'

    def get_auxiliary_bounds(self):
        aux_bounds = self.cfg.case_study.KS_bounds.aux_args
        aux_lb = jnp.array([bound[0][0] for bound in aux_bounds])
        aux_ub = jnp.array([bound[0][1] for bound in aux_bounds])
        return aux_lb, aux_ub

    def wrap_get_constraints(self, solution):
        """
        Sweep the auxiliary space on a grid for a fixed solution and return the
        feasibility surface as a dict (custom to the current case study).
        """
        logging.warning('This post-process evaluation is set up for a specific case study and is not generalised yet.')
        bounds = self.get_auxiliary_bounds()
        x_range = (bounds[0][0], bounds[1][0])
        y_range = (bounds[0][1], bounds[1][1])
        num_points = 200
        # grid of points for the x and y axes
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)

        X, Y = np.meshgrid(x, y)

        # flatten the grids into a batch of coordinate pairs
        x_coords_batch = X.ravel().reshape(-1, 1)
        y_coords_batch = Y.ravel().reshape(-1, 1)

        solution_batch = np.tile(solution, (num_points * num_points, solution.shape[0]))
        coords_batch = np.hstack((solution_batch, x_coords_batch, y_coords_batch, np.zeros((num_points*num_points,1))))

        uncertain_params = jnp.empty((num_points, 1))
        epsilon = self.get_constraints(coords_batch, uncertain_params)[5]
        df = {
            'x': np.reshape(x_coords_batch, X.shape),
            'y': np.reshape(y_coords_batch, X.shape),
            'z': np.reshape(epsilon.squeeze(), X.shape)
        }

        print(f'Post-process evaluation: {[v.shape for v in df.values()]} shape check.')

        return df
