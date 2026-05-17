from abc import ABC
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.random import PRNGKey, choice, normal

import logging



# TODO think about ways to fit approximators to the eks data within this class.

class initialisation(ABC):
    def __init__(self, cfg, graph, network_simulator, constraint_evaluator, sampler, approximator):
        self.cfg = cfg
        self.graph = graph
        self.network_simulator = network_simulator(cfg, graph, constraint_evaluator)
        self.sampler = sampler
        self.approximator = approximator

    def run(self):
        samples = self.sample_design_space()
        uncertain_params = self.get_uncertain_params()
        constraints, eks_data = self.network_simulator.get_data(samples, uncertain_params)

        # Evolutionary refinement: select best samples and generate children
        if self.cfg.init.get('evolutionary_refinement', False):
            samples, constraints, eks_data = self.evolutionary_refine(
                samples, constraints, eks_data, uncertain_params
            )

        self.update_eks_data(eks_data)
        for node in self.graph.nodes():
            self.graph.nodes[node]["fn_evals"] = self.network_simulator.function_evaluations[node]
        self.graph.graph["initial_forward_pass"] = pd.DataFrame({col:samples[:,i] for i,col in enumerate(self.cfg.case_study.design_space_dimensions)})
        return self.graph

    # =========================================================================
    # Evolutionary refinement methods
    # =========================================================================

    def evolutionary_refine(self, samples, constraints, eks_data, uncertain_params):
        """
        Refine initial samples using evolutionary local search.

        Supports multiple scoring methods - if a list is provided, each method
        runs independently on the base samples and all results are combined.
        This expands bounds in multiple directions (e.g., toward low cost AND
        toward constraint boundaries).

        Parameters
        ----------
        samples : jnp.ndarray
            Initial Sobol samples, shape (n_samples, n_design)
        constraints : dict
            {node: constraint_array} where constraint_array is (n_samples, n_theta, n_g)
        eks_data : dict
            {edge: input_data} edge input bounds data
        uncertain_params : list
            Uncertain parameters for each node

        Returns
        -------
        combined_samples, combined_constraints, combined_eks
        """
        n_best = self.cfg.init.get('n_best', 32)
        n_children = self.cfg.init.get('n_children_per_best', 4)
        perturbation_scale = self.cfg.init.get('perturbation_scale', 0.1)
        n_iterations = self.cfg.init.get('n_iterations', 1)
        scoring_methods = self.cfg.init.get('scoring_method', 'min_cost')

        # Support both single method and list of methods
        if isinstance(scoring_methods, str):
            scoring_methods = [scoring_methods]
        else:
            scoring_methods = list(scoring_methods)

        bounds = self.get_bounds()

        logging.info(f"Evolutionary refinement: methods={scoring_methods}, n_best={n_best}, "
                     f"n_children={n_children}, scale={perturbation_scale}, iterations={n_iterations}")

        # Start with base samples
        current_samples = samples
        current_constraints = constraints
        current_eks = eks_data
        current_costs = None

        # Run each scoring method independently, accumulating all children
        for scoring_method in scoring_methods:
            logging.info(f"=== Running {scoring_method} refinement ===")

            # Evaluate costs if needed for this method
            if scoring_method == 'min_cost' and current_costs is None:
                if not self.cfg.case_study.get('eval_cost', False):
                    raise ValueError("min_cost scoring requires eval_cost=True in case_study config")
                current_costs = self._evaluate_costs(current_samples, uncertain_params)

            for iteration in range(n_iterations):
                # Score samples
                if scoring_method == 'min_cost':
                    scores = self._cost_seek(current_constraints, current_costs)
                    score_label = "cost"
                elif scoring_method == 'l1_constraint':
                    scores = self._boundary_seek(current_constraints)
                    score_label = "l1_dist"
                else:
                    raise ValueError(f"Unknown scoring_method: {scoring_method}. Use 'min_cost' or 'l1_constraint'.")

                # Select best samples (higher score = better)
                n_select = min(n_best, len(scores))
                best_indices = jnp.argsort(scores)[-n_select:]
                best_samples = current_samples[best_indices]

                # Show score range (negate back to original metric)
                worst_val = -float(scores[best_indices[0]])
                best_val = -float(scores[best_indices[-1]])
                logging.info(f"{scoring_method} iter {iteration+1}: selected {n_select} best, "
                             f"{score_label} range [{best_val:.2f}, {worst_val:.2f}]")

                # Generate children around best samples
                children = self._generate_children(best_samples, n_children, perturbation_scale, bounds)

                # Evaluate children
                child_constraints, child_eks = self.network_simulator.get_data(children, uncertain_params)

                # Evaluate costs for children if needed
                child_costs = None
                if current_costs is not None:
                    child_costs = self._evaluate_costs(children, uncertain_params)

                # Combine samples
                current_samples = jnp.vstack([current_samples, children])
                current_constraints = self._merge_constraints(current_constraints, child_constraints)
                current_eks = self._merge_eks(current_eks, child_eks)
                if current_costs is not None and child_costs is not None:
                    current_costs = self._merge_constraints(current_costs, child_costs)

                logging.info(f"{scoring_method} iter {iteration+1}: total samples now {current_samples.shape[0]}")

        return current_samples, current_constraints, current_eks

    def _evaluate_costs(self, samples, uncertain_params):
        """
        Evaluate node costs for all samples using a cost-specific network simulator.

        Returns
        -------
        costs : dict
            {node: cost_array} where cost_array is (n_samples, n_theta, n_cost)
        """
        from mu_F.constraints.constructor import constraint_evaluator
        from mu_F.unit_evaluators.constructor import network_simulator

        # Create a cost-evaluating simulator
        cost_simulator = network_simulator(
            self.cfg, self.graph.copy(),
            lambda cfg, graph, node, constraint_type='node_cost': constraint_evaluator(cfg, graph, node, constraint_type='node_cost'),
            type_cons='node_cost'
        )
        costs, _ = cost_simulator.get_data(samples, uncertain_params)
        return costs

    def _cost_seek(self, constraints, costs):
        """
        Score samples by total cost across all nodes (lower cost = higher score).

        Parameters
        ----------
        constraints : dict
            {node: constraint_array} - not used, kept for API consistency
        costs : dict
            {node: cost_array} where cost_array is (n_samples, n_theta, n_cost)

        Returns
        -------
        scores : jnp.ndarray
            Shape (n_samples,), higher = better (lower cost)
        """
        n_samples = None
        total_cost = None

        for node, c in costs.items():
            # c has shape (n_samples, n_theta, n_cost)
            # Take mean over uncertain params, sum over cost dims
            if n_samples is None:
                n_samples = c.shape[0]
                total_cost = jnp.zeros(n_samples)

            # Sum all cost components, mean over theta
            node_cost = jnp.mean(jnp.sum(c, axis=-1), axis=1)  # (n_samples,)
            total_cost = total_cost + node_cost

        # Negate: lower cost = higher score
        return -total_cost

    def _boundary_seek(self, constraints):
        """
        Score samples by L1 distance to constraint boundary.

        Points closer to the boundary (g ≈ 0) are preferred for exploring
        the feasibility frontier. We normalize constraints by their range
        and compute L1 distance to zero.

        Lower L1 distance = higher score.
        """
        # First pass: compute min/max for normalization
        all_g = []
        for node, g in constraints.items():
            # g: (n_samples, n_theta, n_g) -> flatten theta dimension
            g_flat = jnp.mean(g, axis=1)  # (n_samples, n_g)
            all_g.append(g_flat)

        all_g = jnp.concatenate(all_g, axis=1)  # (n_samples, total_g)

        g_min = jnp.min(all_g, axis=0, keepdims=True)
        g_max = jnp.max(all_g, axis=0, keepdims=True)
        g_range = g_max - g_min + 1e-8  # avoid division by zero

        # Normalize to [0, 1] range
        g_norm = (all_g - g_min) / g_range

        # L1 distance to 0.5 (middle of normalized range, representing boundary)
        l1_distance = jnp.sum(jnp.abs(g_norm), axis=1)  # (n_samples,)

        # Lower L1 distance = closer to boundary = higher score
        return -l1_distance

    def _generate_children(self, best_samples, n_children, perturbation_scale, bounds):
        """
        Generate children around best samples using Gaussian perturbation.

        Parameters
        ----------
        best_samples : jnp.ndarray
            Shape (n_best, n_design)
        n_children : int
            Number of children per best sample
        perturbation_scale : float
            Fraction of bound range for std dev
        bounds : tuple
            (lower_bounds, upper_bounds) arrays

        Returns
        -------
        children : jnp.ndarray
            Shape (n_best * n_children, n_design)
        """
        lb, ub = bounds
        bound_range = ub - lb
        std = perturbation_scale * bound_range

        n_best, n_design = best_samples.shape

        # Generate all children at once
        key = PRNGKey(42)  # Fixed seed for reproducibility
        noise = normal(key, shape=(n_best, n_children, n_design))

        # Broadcast: best_samples[:, None, :] + noise * std
        children = best_samples[:, None, :] + noise * std[None, None, :]

        # Clip to bounds
        children = jnp.clip(children, lb[None, None, :], ub[None, None, :])

        # Reshape to (n_best * n_children, n_design)
        children = children.reshape(-1, n_design)

        return children

    def _merge_constraints(self, c1, c2):
        """Merge two constraint dicts by concatenating along sample axis."""
        merged = {}
        for node in c1.keys():
            merged[node] = jnp.concatenate([c1[node], c2[node]], axis=0)
        return merged

    def _merge_eks(self, e1, e2):
        """Merge two eks dicts by concatenating along sample axis."""
        merged = {}
        for edge in e1.keys():
            merged[edge] = jnp.concatenate([e1[edge], e2[edge]], axis=0)
        return merged
    
    def update_eks_data(self, eks_data):
        # fit approximator to eks data
        for edge in eks_data.keys():
            self.graph.edges[edge[0], edge[1]]["input_data_bounds"] = self.approximator(eks_data[edge], self.cfg)

        return 

    def sample_design_space(self):
        bounds = self.get_bounds()
        n_d = len(bounds[0])
        return self.sampler.sample_design_space(n_d, bounds, self.cfg.init.sobol_samples)
        

    def get_bounds(self):
        bounds = self.cfg.case_study.KS_bounds
        return self.process_bounds(self.bounds_to_dictionary(bounds))
    
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
        

        return selected_params # sample selected parameters from the list according to probability mass specified by the user

    @staticmethod
    def process_bounds(bounds):
        """ from dictionary to array """
        print(bounds)
        # get lower and upper bounds in array form
        lower_bound = jnp.array([bounds[i][i][0] for i in bounds.keys()])
        upper_bound = jnp.array([bounds[i][i][1] for i in bounds.keys()])
        return lower_bound, upper_bound
    
    @staticmethod
    def bounds_to_dictionary(bounds):
        """ Method to construct a list of bounds for the design space"""

        bounds_ = {}
        index = 0
        for j, unit_bounds in enumerate(bounds.design_args):
            for i, bound in enumerate(unit_bounds):
                if bound[0] != 'None' and bound[1] != 'None':
                    bounds_[f'd{index}'] = {f'd{index}': [bound[0], bound[1]]}
                    index += 1
    
        for j, unit_bounds in enumerate(bounds.aux_args):
            for i, bound in enumerate(unit_bounds):
                if bound[0] != 'None' and bound[1] != 'None':
                    # global auxiliary varibles
                    bounds_[f'd{index}'] = {f'd{index}': [bound[0], bound[1]]}
                    index += 1

            
        return bounds_


