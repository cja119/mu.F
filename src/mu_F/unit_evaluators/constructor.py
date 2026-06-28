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
from mu_F._types import (typecheck, DesignBatch, StateScen, AuxBatch,
                         UncertainScen, StateBatch, UncertainBatch,
                         OutputScen, OutputDiag)


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

    @typecheck
    def evaluate(self, design_args: DesignBatch, input_args: StateScen,
                 aux_args: AuxBatch, uncertain_params: UncertainScen) -> OutputScen:
        """
        Evaluate the unit over the design x scenario grid; every arg is laid out
        (N, S, .) before the vmap, with shared N / S axes enforced by the
        annotations and a row-by-row fallback to surface a failing design.
        """
        n, s = design_args.shape[0], input_args.shape[1]

        dd_args = self.get_decision_dependent_params(design_args, uncertain_params)               # (N, S, D)
        design_args = _spread(design_args, s)                                                     # (N, S, U)
        aux_args = _spread(aux_args, s)                                                           # (N, S, A)
        unc = jnp.broadcast_to(uncertain_params[None, :, :], (n, s, uncertain_params.shape[-1]))  # (N, S, Z)

        try:
            return self.unit_cfg.evaluator(design_args, input_args, aux_args, dd_args, unc)
        except Exception as outer_exc:
            return self._isolate_failure(design_args, input_args, aux_args, dd_args, unc, outer_exc)

    @typecheck
    def evaluate_diagonal(self, design_args: DesignBatch, input_args: StateBatch,
                          aux_args: AuxBatch, uncertain_params: UncertainBatch) -> OutputDiag:
        """
        Diagonal evaluation: row i uses realisation i, scenario axis collapsed.
        uncertain_params is (N, param_dim); returns (N, 1, n_out).
        """
        dd_params = self.unit_cfg.decision_dependent_params_diag(design_args, uncertain_params)
        outputs = self.unit_cfg.evaluator_diag(
            design_args, input_args, aux_args, dd_params, uncertain_params,
        )
        return jnp.expand_dims(outputs, axis=1)

    # ---- Private Methods ----

    def _isolate_failure(self, design_args, input_args, aux_args, dd_args, unc, outer_exc):
        """Re-run the vmap row by row to surface the offending design on failure."""
        n_batch = int(design_args.shape[0])
        logging.error(
            "[unit_eval] vmapped evaluator failed; isolating offender row-by-row "
            "over %d rows.  Outer error: %r", n_batch, outer_exc,
        )
        for i in range(n_batch):
            try:
                self.unit_cfg.evaluator(
                    design_args[i:i+1], input_args[i:i+1], aux_args[i:i+1],
                    dd_args[i:i+1], unc[i:i+1],
                )
            except Exception as row_exc:
                np.set_printoptions(suppress=True, precision=6, linewidth=200)
                logging.error(
                    "[unit_eval] FAILING ROW %d / %d  design=%s input=%s aux=%s dd=%s unc=%s\n  %r",
                    i, n_batch,
                    np.asarray(design_args[i:i+1]).squeeze(), np.asarray(input_args[i:i+1]).squeeze(),
                    np.asarray(aux_args[i:i+1]).squeeze(), np.asarray(dd_args[i:i+1]).squeeze(),
                    np.asarray(unc[i:i+1]).squeeze(), row_exc,
                )
                raise
        raise outer_exc


def _spread(array, scenarios):
    """Broadcast a per-design array across the uncertainty scenarios: (N, F) -> (N, S, F)."""
    return jnp.broadcast_to(array[:, None, :], (array.shape[0], scenarios, array.shape[-1]))


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
            if self.cfg.model.root_node_inputs[self.node] not in (None, 'None'):
                input_args = jnp.array([self.cfg.model.root_node_inputs[self.node]]*design_args.shape[0])
            else:
                input_args = jnp.empty((design_args.shape[0], 0))
        # if no aux to the unit, use the root node aux or add empty array
        if aux_args.shape[1] == 0:
            if self.cfg.model.node_aux[self.node] not in (None, 'None'):
                aux_args = jnp.array([self.cfg.model.root_node_aux[self.node]]*design_args.shape[0])
            else:
                aux_args = jnp.empty((design_args.shape[0], 0))
            
        input_args = _spread(input_args, uncertain_params.shape[0])

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
            if self.cfg.model.root_node_inputs[self.node] not in (None, 'None'):
                input_args = jnp.array([self.cfg.model.root_node_inputs[self.node]] * n)
            else:
                input_args = jnp.empty((n, 0))
        if aux_args.shape[1] == 0:
            if self.cfg.model.node_aux[self.node] not in (None, 'None'):
                aux_args = jnp.array([self.cfg.model.root_node_aux[self.node]] * n)
            else:
                aux_args = jnp.empty((n, 0))
        return self.evaluate_diagonal(design_args, input_args, aux_args, uncertain_params)

    def get_auxilliary_input_decision_split(self, decisions):
        """
        Split a flat decision vector into design / input / auxiliary blocks. Aux is
        the trailing block, so a root node seeded with its full state (n_input = 0)
        cannot bleed that state into the aux slot.
        """
        n_d = self.graph.nodes[self.node]['n_design_args']
        n_aux = int(self.graph.graph['n_aux_args'])
        design_args = decisions[:, :n_d]
        if n_aux > 0:
            input_args, auxiliary_args = decisions[:, n_d:-n_aux], decisions[:, -n_aux:]
        else:
            input_args, auxiliary_args = decisions[:, n_d:], decisions[:, n_d:n_d]
        return design_args, input_args, auxiliary_args


def _aux_expanded_base(base, cfg, node, graph):
    """
    Wrap a unit base so a node carrying a strict aux subset has its compact block
    scattered into the full global vector (defaults for gaps) before evaluation;
    returns base untouched when the node already holds the full set.
    """
    n_global = int(cfg.case_study.global_n_aux_args)
    ids = tuple(graph.nodes[node]['aux_ids'])
    if n_global == 0 or len(ids) == 0 or ids == tuple(range(n_global)):
        return base

    seats = jnp.asarray(ids)
    defaults = cfg.case_study.get('aux_default', None)
    full = jnp.asarray(defaults) if defaults is not None else jnp.zeros(n_global)

    def base_full_aux(design_args, input_args, aux, dd_args, unc):
        return base(design_args, input_args, full.astype(aux.dtype).at[seats].set(aux), dd_args, unc)

    return base_full_aux


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
            base = _aux_expanded_base(base, cfg, node, graph)   # node aux subset -> full global vector
            # grid (search): every arg laid out (N, S, .); inner sweep = scenarios, outer = designs
            self.evaluator = vmap(vmap(base, in_axes=0, out_axes=0), in_axes=0, out_axes=0)
            # diagonal (rollout): realisation paired one-per-trajectory along the batch
            self.evaluator_diag = vmap(base, in_axes=0, out_axes=0)

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

    def __init__(self, cfg, graph, constraint_evaluator, type_cons='process', cost_type=None):
        self.cfg = cfg
        self.graph = graph.copy()
        self.type = type_cons
        self.cost_type = cost_type
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
                if self.cfg.model.root_node_inputs[node] not in (None, 'None'):
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

            # cost-seeking reads the node cost off the same integration
            if self.cost_type is not None:
                cost_evaluator = self.constraint_evaluator(self.cfg, self.graph, node, constraint_type=self.cost_type)
                self.graph.nodes[node]['cost_store'] = cost_evaluator.evaluate(decisions[:, n_d:n_d+unit_nd], inputs, aux_args, outputs)

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

    def get_data_cost(self, decisions, uncertain_params=None):
        """
        Single-pass variant returning the node cost alongside the constraints
        and inputs, read off the same integration; used by min_cost refinement.
        """
        constraints, edge_data = self.simulate(decisions, uncertain_params)
        for node, g in constraints.items():
            self.function_evaluations[node] += g.shape[0]*g.shape[1]
        costs = {node: self.graph.nodes[node]['cost_store'] for node in self.graph.nodes}
        return constraints, edge_data, costs


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
                if self.cfg.model.root_node_inputs[node] not in (None, 'None'):
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
