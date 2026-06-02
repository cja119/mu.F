"""Joint reconstruction of feasibility live sets across the graph."""
from abc import ABC
import jax.numpy as jnp
import numpy as np
from jax.random import choice, PRNGKey

from mu_F.reconstruction.samplers import SobolSampler
from mu_F.reconstruction.objects import LiveSet
from mu_F.reconstruction.methods import construct_cartesian_product_of_live_sets
from mu_F.reconstruction.utils import post_process_sampling_setup, post_process_setup

class ReconstructBase(ABC):
    """Abstract interface for reconstruction routines.

    Declares the sampling and evaluation hooks that concrete
    reconstruction classes implement.

    """

    # ---- External Methods ----

    def __init__(self):
        pass

    def run(self):
        pass

    def sample_live_sets(self):
        pass

    def evaluate_joint_model(self):
        pass


class Reconstruction(ReconstructBase):
    """Joint reconstruction over the product of per-node live sets.

    Samples candidates from each node's live set, evaluates the joint
    model constraints, and grows a feasible live set until complete,
    optionally handing off to a post-process step.

    """

    # ---- External Methods ----

    def __init__(self, cfg, graph, model, iterate):
        self.cfg = cfg
        self.graph = graph
        self.model = model
        self.live_sets_nd_proj = construct_cartesian_product_of_live_sets(graph)
        self.ls_holder = LiveSet(cfg, cfg.samplers.notion_of_feasibility)
        self.feasible = False
        self.post_process_bool = cfg.reconstruction.post_process#[iterate]
        self.iterate = iterate
        self.desired_edge_index = list(self.graph.nodes)[-1]
        self.desired_regressor_data = LiveSet(cfg, cfg.samplers.notion_of_feasibility)

    def update_live_set(self, candidates, constraint_vals):
        """
        Append the feasible candidates to the live set and report
        whether it is now complete.
        """
        # keep only the feasible candidates
        feasible_points, feasible_prob = self.ls_holder.check_live_set_membership(candidates, constraint_vals)
        self.ls_holder.append_to_live_set(feasible_points, feasible_prob)
        return self.ls_holder.check_if_live_set_complete()


    def run(self, mode='sobol'):
        """
        Run the joint reconstruction: sample, evaluate and grow the live
        set until feasible, then optionally post-process.
        """
        feasible = False
        ls_holder = self.ls_holder
        uncertain_params = self.get_uncertain_params()

        while not feasible:
            live_sets_nd_proj, candidates = self.sample_live_sets(scheme=mode)
            constraint_vals = self.evaluate_joint_model(candidates, uncertain_params=uncertain_params)
            constraint_vals = jnp.concatenate([g for g in constraint_vals.values()], axis=-1)
            feasible = self.update_live_set(candidates, constraint_vals)

        joint_live_set, joint_live_set_prob = ls_holder.get_live_set()

        if self.post_process_bool:
            # no uncertain parameters are currently passed to the post process
            self.graph = self.post_process_runner(cfg=self.cfg, graph=self.graph, model=self.model, ls_holder=ls_holder, iterate=self.iterate)

        return joint_live_set, joint_live_set_prob, self.graph
    
    def post_process_runner(self, cfg, graph, model, ls_holder, iterate):
        """
        Build and run the post-process step, loading the live-set
        classification and regression data onto the graph first.
        """
        graph = ls_holder.load_classification_data_to_graph(graph, str_='post_process_lower_')
        graph = model.desired_regressor_data.load_regression_data_to_graph(graph, str_='post_process_lower_')
        post_process = post_process_setup(cfg, graph, model)
        if cfg.reconstruction.post_process_sampler:
            post_process = post_process_sampling_setup(cfg, post_process,  lambda : (self.sample_live_sets())[1], live_set)
        graph = post_process.run()

        return graph

    def get_uncertain_params(self):
        """
        Draw the uncertain parameter samples used by the joint model,
        weighting by the user-specified probability mass.
        """
        if self.cfg.formulation == 'probabilistic':
            param_dict = self.cfg.case_study.parameters_samples
            list_of_params = [jnp.array([p['c'] for p in param]) for param in param_dict]
            list_of_weights = [jnp.array([p['w'] for p in param]).reshape(-1) for param in param_dict]

            max_parameter_samples = self.cfg.max_uncertain_samples
            selected_params = [choice(PRNGKey(0), a, shape=(max_parameter_samples,), replace=True, p=weight, axis=0) for a, weight in zip(list_of_params, list_of_weights)]
        elif self.cfg.formulation == 'deterministic':
            selected_params = [jnp.array([param]) for param in self.cfg.case_study.parameters_best_estimate]
            
        return selected_params
    
    def sample_live_sets(self, scheme = "sobol"):
        """
        Sample candidate points from each node's live set, drawing
        indices by the requested scheme and shuffling for the next round.
        """
        sampled_live_sets = {}
        n_aux = self.graph.graph['n_aux_args']
        for node, live_set in self.live_sets_nd_proj.items():
            rng = np.random.default_rng()
            # index bounds spanning the live set
            n_samples = self.cfg.samplers.ns.n_replacements
            n_l = live_set.shape[0]
            bounds = [np.zeros(1), np.ones(1)*n_l]
            if scheme == "sobol":
                unrounded_indices = SobolSampler().sample_design_space(1, bounds, n_samples)
            elif scheme == "uniform":
                unrounded_indices = rng.uniform(bounds[0], bounds[1], (n_samples, 1))
            else:
                raise ValueError("Invalid scheme")

            # round and clip the sampled indices into range
            rnd_ind = np.round(unrounded_indices).astype(int)
            rounded_indices = np.minimum(rnd_ind, self.cfg.samplers.ns.n_live-1)
            try:
                sampled_live_sets[node] = np.copy(live_set[rounded_indices[:].reshape(-1), :]).reshape(-1, live_set.shape[1])
            except:
                sampled_live_sets[node] = live_set[rounded_indices[:].reshape(-1), :].reshape(-1,n_aux)
            # shuffle the live set for the next round
            rng.shuffle(live_set, axis=0)
            self.live_sets_nd_proj[node] = np.copy(live_set)
            
        return self.live_sets_nd_proj, np.hstack([live_set for live_set in sampled_live_sets.values()])

    def evaluate_joint_model(self, candidates, uncertain_params):
        """
        Evaluate the joint model constraints at the sampled candidates.
        """
        constraints = self.model.get_constraints(candidates,  uncertain_params)
        return constraints
    
    




