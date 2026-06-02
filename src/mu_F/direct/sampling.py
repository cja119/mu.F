"""Direct DEUS sampling of the feasible design space and post-processing."""
import jax.numpy as jnp
import numpy as np
import logging
import pandas as pd

from mu_F.unit_evaluators.constructor import NetworkSimulator
from mu_F.samplers.constructor import ConstructDeusProblemNetwork
from mu_F.constraints.constructor import ConstraintEvaluator
from mu_F.samplers.utils import create_problem_description_deus_direct
from mu_F.visualisation.visualiser import Visualiser
from mu_F.reconstruction.constructor import Reconstruction as reconstruct
from mu_F.reconstruction.objects import LiveSet, ReconstructionDataset
from mu_F.reconstruction.utils import post_process_sampling_setup, post_process_setup
from mu_F.direct.base import SolveDirect 

from deus import DEUS


class DirectSampler(SolveDirect):
    """Feasible-set explorer that samples the design space directly via DEUS.

    Delegates the solve to the module-level direct method, which builds the
    DEUS problem from the graph, samples feasible/infeasible designs and
    optionally runs reconstruction post-processing.

    """

    # ---- External Methods ----

    def __init__(self, cfg, G):
        super().__init__(cfg, G)

    def solve(self, problem_data, x0=None):
        """
        Solve by sampling the design space with the direct DEUS method.
        """
        return apply_direct_method(self.cfg, self.G)

    # ---- Private Methods ----

    def _load_solver(self):
        """
        No standalone solver object; sampling is driven by DEUS.
        """
        return None

    def _prepare_model(self, graph):
        """
        No monolithic NLP is built for the sampling route.
        """
        return None

    def _get_solution(self, solver_output):
        """
        Solution extraction is handled inside the direct method.
        """
        return None


# ---------------------------------------------------------------------------
# Direct sampling method
# ---------------------------------------------------------------------------

def apply_direct_method(cfg, graph):

    model = NetworkSimulator(cfg, graph, ConstraintEvaluator)
    problem_description = create_problem_description_deus_direct(cfg, graph)
    solver =  ConstructDeusProblemNetwork(DEUS, problem_description, model)
    solver.solve()
    feasible_set, infeasible_set = solver.get_solution()
    logging.info(f"Feasible set shape: {feasible_set[0].shape}, Infeasible set shape: {infeasible_set[0].shape}")
    for node in graph.nodes:
        graph.nodes[node]['fn_evals'] = model.function_evaluations[node]

    if cfg.reconstruction.plot_reconstruction == 'nominal_map':
        if isinstance(feasible_set, tuple):
            df = pd.DataFrame({key: feasible_set[0][:,i] for i, key in enumerate(cfg.case_study.design_space_dimensions)})
        else:
            df = pd.DataFrame({key: feasible_set[:,i] for i, key in enumerate(cfg.case_study.design_space_dimensions)})
    elif cfg.reconstruction.plot_reconstruction == 'probability_map':
        df = pd.DataFrame({key: feasible_set[:,i] for i, key in enumerate(cfg.case_study.design_space_dimensions)})
        df['probability'] = feasible_set

    graph.graph['feasible_set'] = feasible_set
    Visualiser(cfg, graph, data=df, string='design_space', path=f'design_space_direct').run()

    if cfg.reconstruction.post_process:
        def sampler(
            ):
            
            if isinstance(feasible_set, tuple):
                fs = feasible_set[0]
            else:
                fs = feasible_set
            rng = np.random.default_rng()
            rng.shuffle(fs, axis=0)
            n_l = cfg.samplers.ns.final_sample_live
            n_samples = cfg.samplers.ns.n_replacements
            rng = np.random.default_rng()
            bounds = [np.zeros(1), np.ones(1)*n_l]
            unrounded_indices = rng.uniform(bounds[0], bounds[1], (n_samples, 1))
            rnd_ind = np.round(unrounded_indices).astype(int)
            rounded_indices = np.minimum(rnd_ind, n_l-1)
            feasible_samples = fs[rounded_indices]
            return feasible_samples
        
        graph =  load_classifier_to_graph(feasible_set, infeasible_set, graph, str_='post_process_lower_')
        graph = load_regressor_to_graph(solver, graph, str_='post_process_lower_')
        post_process = post_process_setup(cfg, graph, model)
        if cfg.reconstruction.post_process_sampler:
            post_process = post_process_sampling_setup(cfg, post_process, sampler, live_set)
        graph = post_process.run()

    return feasible_set, infeasible_set


def load_classifier_to_graph(feasible, infeasible, graph, str_):
    assert isinstance(feasible, tuple) 
    assert isinstance(infeasible, tuple)
    # unpack feasible and infeasible sets
    feasible_query, feasible_prob = feasible
    infeasible_query, infeasible_prob = infeasible
    # get samples
    live_set = np.vstack(feasible_query)
    infeasible_set = np.vstack(infeasible_query) 
    # corresponding labels
    live_set_labels = np.ones(live_set.shape[0]).reshape(-1,1) 
    infeasible_set_labels = -np.ones(infeasible_set.shape[0]).reshape(-1,1)
    # create a dataset object
    all_data = np.vstack([live_set, infeasible_set])    
    all_labels = np.vstack([live_set_labels, infeasible_set_labels])
    logging.info(str_ + f"Live set size: {live_set.shape}, Infeasible set size: {infeasible_set.shape}")

    graph.graph[str_+ 'classifier_training'] = ReconstructionDataset(all_data, all_labels)
    return graph

def load_regressor_to_graph(solver, graph, str_):
    return solver.get_regresssion_data(graph, str_=str_)
    