"""Decomposition driver: runs the forward/backward passes and constraint tuner."""

from functools import partial
import pandas as pd
import networkx as nx
import jax
import logging
import torch

from mu_F.integration import ApplyDecomposition
from mu_F.initialisation.methods import Initialisation
from mu_F.reconstruction.constructor import Reconstruction
from mu_F.unit_evaluators.constructor import NetworkSimulator
from mu_F.constraints.constructor import ConstraintEvaluator
from mu_F.samplers.space_filling import SobolSampler
from mu_F.samplers.algorithms.bo.alg import bayesian_optimization
from mu_F.samplers.appproximators import calculate_box_outer_approximation
from mu_F.visualisation.visualiser import Visualiser
from mu_F.utils import *


def _extended_bounds_are_set(cfg) -> bool:
    """
    True iff cfg.case_study.extendedDS_bounds declares usable bounds, so the
    init phase can be skipped. False for None / "None" / all-None entries.
    """
    raw = cfg.case_study.get('extendedDS_bounds', None)
    if raw is None or raw == "None":
        return False
    try:
        return any(b is not None and b != "None" for b in raw)
    except TypeError:
        return True


class Decomposition:
    """Orchestrates the forward/backward decomposition passes over the graph.

    Drives each pass (initialisation, ApplyDecomposition, optional rollout) in
    the configured order, optionally reconstructing the joint live set, and
    maintains the precedence order consumed by the subproblem solvers.

    """

    # ---- External Methods ----

    def __init__(self, cfg, G, precedence_order, mode='forward', max_devices=1):
        self.cfg = cfg
        self.G = G
        self.G.graph['terminate'] = False

        self.original_precedence_order = precedence_order.copy()
        self.precedence_order = precedence_order.copy()
        self.mode = mode
        self.total_iterations = len(mode)
        self.max_devices = max_devices

        self.sampler = self._make_sampler(cfg)
        self.approximator = self._make_approximator(cfg)

    def run(self, iterations=0):
        """
        Run each configured pass, reconstructing and updating the precedence
        order between passes.
        """
        for i, m in enumerate(self.mode):
            operations, visualisations = self._define_operations(m, iterations)

            for key in operations.keys():
                self.G = operations[key](self.cfg, self.G).run()
                if self.G.graph['terminate']:
                    return self.G
                if not self.cfg.case_study.get('bypass_visualisations', False):
                    visualisations[key](self.cfg, self.G).run()
                save_graph(self.G.copy(), m + '_iterate_' + str(iterations))

            if self.cfg.reconstruction.reconstruct[i]:
                jax.clear_caches()
                self._reconstruct(m, i + iterations)

            self._update_precedence_order(m)

        return self.G

    # ---- Private Methods ----

    @staticmethod
    def _make_sampler(cfg):
        """
        Build the space-filling sampler used during initialisation.
        """
        return SobolSampler() if cfg.init.sampler == 'sobol' else None

    @staticmethod
    def _make_approximator(cfg):
        """
        Build the outer-approximation routine for the known/unknown set.
        """
        return calculate_box_outer_approximation if cfg.samplers.ku_approximation == 'box' else None

    @staticmethod
    def _terminal_nodes(G, side):
        """
        Nodes with zero in-degree (source, side='in') or out-degree (sink).
        """
        degree = G.in_degree if side == 'in' else G.out_degree
        return {node for node in G.nodes() if degree(node) == 0}

    def _define_operations(self, m, iteration):
        """
        Build the ordered operations and visualisations for a given pass mode.
        """
        decompose = partial(ApplyDecomposition, mode=m, max_devices=self.max_devices)
        visualise = partial(Visualiser, mode=m, string='decomposition',
                            path=f'decomposition_{m}_iterate_{iteration}')

        if m in ('forward', 'backward-forward'):
            return ({0: partial(decompose, precedence_order=self.precedence_order)}, {0: visualise})

        if m == 'rollout':
            roll = partial(decompose, precedence_order=self.original_precedence_order.copy())
            roll_vis = partial(Visualiser, mode=m, string='rollout', path=f'rollout_iterate_{iteration}')
            return ({0: roll}, {0: roll_vis})

        operations, visualisations = self._initialisation_step(m, iteration)
        k = len(operations)
        operations[k] = partial(decompose, precedence_order=self.precedence_order)
        visualisations[k] = visualise
        return operations, visualisations

    def _initialisation_step(self, m, iteration):
        """
        Optional iterate-0 initialisation op for the backward passes; skipped
        when extendedDS_bounds are pre-set.
        """
        if iteration != 0:
            return {}, {}

        if _extended_bounds_are_set(self.cfg):
            logging.info(
                "Skipping Initialisation: cfg.case_study.extendedDS_bounds "
                "is pre-set; the bounds consumers use it directly and would "
                "ignore init's per-edge input_data_bounds."
            )
            return {}, {}

        op = partial(Initialisation, network_simulator=NetworkSimulator,
                     constraint_evaluator=ConstraintEvaluator,
                     sampler=self.sampler, approximator=self.approximator)
        vis = partial(Visualiser, mode=m, string='initialisation',
                      path=f'initialisation_{m}_iterate_{iteration}')
        return {0: op}, {0: vis}

    def _reconstruct(self, m, i):
        """
        Reconstruct the joint live set for the pass and refresh the graph's
        per-node function evaluations and visualisations.
        """
        network_model = NetworkSimulator(self.cfg, self.G, ConstraintEvaluator)
        reconstructor = Reconstruction(self.cfg, self.G, network_model, iterate=i)
        joint_live_set, joint_live_set_prob, self.G = reconstructor.run()

        for node in self.G.nodes():
            self.G.nodes[node]["fn_evals"] += network_model.function_evaluations[node]

        df = pd.DataFrame({key: joint_live_set[:, j] for j, key in enumerate(self.cfg.case_study.design_space_dimensions)})
        if self.cfg.reconstruction.plot_reconstruction == 'probability_map':
            df['probability'] = joint_live_set_prob

        if not self.cfg.case_study.get('bypass_visualisations', False):
            Visualiser(self.cfg, self.G, df, 'reconstruction', path=f'reconstruction_{m}_iterate_{i}').run()
        df.to_excel(f'inside_samples_{m}_iterate_{i}.xlsx')
        save_graph(self.G.copy(), m + '-reconstructed' + '_iterate_' + str(i))

    def _update_precedence_order(self, m):
        """
        Drop source (backward) or sink (forward) nodes from the precedence
        order after a pass.
        """
        order = self.original_precedence_order.copy()
        if self.cfg.method != 'decomposition_constraint_tuner':
            if m == 'backward' or 'forward-backward':
                drop = self._terminal_nodes(self.G, 'in')
            elif m == 'forward' or 'backward-forward':
                drop = self._terminal_nodes(self.G, 'out')
            else:
                drop = set()
            order = [node for node in order if node not in drop]
        self.precedence_order = order

    def _restore_precedence_order(self):
        """
        Reset the working precedence order back to the original.
        """
        self.precedence_order = self.original_precedence_order.copy()


# ---------------------------------------------------------------------------
# Constraint-tuning driver (Bayesian optimisation over back-offs)
# ---------------------------------------------------------------------------

def _update_constraint_tuning_parameters(G, xi):
    """
    Write the per-node constraint back-offs into the graph.
    """
    for k, node in enumerate(G.nodes()):
        G.nodes[node]['constraint_backoff'] = jnp.array(xi[k]).squeeze()
    return G


def _run_a_single_evaluation(xi, cfg, G):
    """
    Run one backward-forward decomposition for a candidate xi and return the
    negative joint log evidence for the Bayesian optimiser to minimise.
    """
    max_devices = len(jax.devices('cpu'))
    logging.info(f"Running iteration {G.graph['iterate']} with xi: {xi}")

    G = _update_constraint_tuning_parameters(G, xi)
    G = Decomposition(cfg, G, G.graph['precedence_order'], ['backward-forward'], max_devices).run(iterations=G.graph['iterate'])
    G.graph['iterate'] += 1

    log_evidence = sum(G.nodes[node]['log_evidence']['mean'] for node in G.nodes())
    return -torch.tensor(log_evidence, dtype=torch.float32).reshape(1, 1)


def decomposition_constraint_tuner(cfg, G, max_devices):
    """
    Run the backward pass, then tune the per-node constraint back-offs via
    Bayesian optimisation over repeated forward evaluations.
    """
    precedence_order = list(nx.topological_sort(G))
    G_obj = Decomposition(cfg, G, precedence_order, ['backward'], max_devices)
    G = G_obj.run()
    G.graph['precedence_order'] = G_obj.precedence_order
    G.graph['iterate'] = 1

    fn = partial(_run_a_single_evaluation, cfg=cfg, G=G)

    lower_bound = [cfg.samplers.bo.bounds.min] * len(G.nodes())
    upper_bound = [cfg.samplers.bo.bounds.max] * len(G.nodes())
    num_initial_points = cfg.samplers.bo.num_initial_points
    num_iterations = cfg.samplers.bo.num_iterations

    xi_opt, best_index = bayesian_optimization(fn, lower_bound, upper_bound, num_initial_points, num_iterations, acq=cfg.samplers.bo.acq)

    logging.info("------- Finished -------")
    logging.info("Best candidate: {}".format(xi_opt))
    logging.info("Best index: {}".format(best_index + 1))
    logging.info("------------------------")

    return
