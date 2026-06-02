"""Live-set container and dataset wrapper used during reconstruction."""
import logging
import numpy as np
from abc import ABC


class LiveSet:
    """Accumulates feasible and infeasible samples during sampling.

    Tracks the running acceptance ratio, decides live-set membership from
    per-sample reliability, and exposes the gathered data to the graph for
    classifier and regressor training.

    """

    # ---- External Methods ----

    def __init__(self, cfg, notion_of_feasibility):
        self.cfg = cfg
        self.live_set, self.live_set_prob = [], []
        self.notion_of_feasibility = notion_of_feasibility
        self.infeasible_set, self.infeasible_prob = [], []
        self.acceptanceratio = 0
        self.N = 0

    def acceptance_ratio(self, feasible):
        """
        Update the running acceptance ratio with a new batch of
        feasibility flags.
        """
        average_ = np.sum(feasible)
        current = self.acceptanceratio
        update = average_ + current * self.N
        update /= (feasible.shape[0] + self.N)
        self.acceptanceratio = update
        logging.info(f"Acceptance ratio: {self.acceptanceratio}")
        return


    def evaluate_feasibility(self, x):
        """
        Compute the per-sample reliability and flag those meeting the
        target reliability, respecting the notion of feasibility.
        """
        n_s = x.shape[1]

        if self.notion_of_feasibility == 'positive':
            y = np.min(x, axis=-1).reshape(x.shape[0],x.shape[1])
            indicator = np.where(y>=0, 1, 0)
            prob_feasible = np.sum(indicator, axis=1)/n_s
            return prob_feasible >= self.cfg.samplers.target_reliability, prob_feasible
        else:
            y = np.max(x, axis=-1).reshape(x.shape[0],x.shape[1])
            indicator = np.where(y<=0, 1, 0)
            prob_feasible = np.sum(indicator, axis=1)/n_s
            return prob_feasible >= self.cfg.samplers.target_reliability, prob_feasible


    def check_live_set_membership(self, x, g):
        """
        Split points into feasible and infeasible, store the infeasible
        set, and update the acceptance ratio.
        """
        feasible, prob = self.evaluate_feasibility(g)
        feasible_points = x[feasible, :]
        feasible_prob = prob[feasible]
        self.infeasible_set.append(x[~feasible, :])
        self.infeasible_prob.append(prob[~feasible].reshape(-1,1))
        self.acceptance_ratio(feasible=feasible)
        return feasible_points, feasible_prob

    def append_to_live_set(self, x, y):
        """
        Append feasible points and their reliabilities to the live set.
        """
        self.live_set.append(x)
        self.live_set_prob.append(y.reshape(-1,1))
        return

    def get_live_set(self):
        """
        Return the live set and reliabilities, truncated to the
        configured final sample size.
        """
        return np.vstack(self.live_set)[:self.live_set_len(), :], np.vstack(self.live_set_prob)[:self.live_set_len()]

    def live_set_len(self):
        """
        Length of the live set, capped at the configured final sample size.
        """
        return min(np.vstack(self.live_set).shape[0], self.cfg.samplers.ns.final_sample_live)

    def check_if_live_set_complete(self):
        """
        Report whether the live set has reached the configured size.
        """
        if np.vstack(self.live_set).shape[0] >= self.cfg.samplers.ns.final_sample_live:
            print(np.vstack(self.live_set).shape, self.cfg.samplers.ns.final_sample_live)
            return True
        else:
            return False

    def load_classification_data_to_graph(self, graph=None, str_='post_process_lower'):
        """
        Stack feasible and infeasible samples with labels and store the
        classifier training dataset on the graph.
        """
        if graph is None:
            raise ValueError("Graph must be provided to load classification data.")

        live_set = np.vstack(self.live_set)
        infeasible_set = np.vstack(self.infeasible_set)
        live_set_labels = np.ones(live_set.shape[0]).reshape(-1,1)
        infeasible_set_labels = -np.ones(infeasible_set.shape[0]).reshape(-1,1)
        all_data = np.vstack([live_set, infeasible_set])
        all_labels = np.vstack([live_set_labels, infeasible_set_labels])
        logging.info(str_ + f"Live set size: {live_set.shape}, Infeasible set size: {infeasible_set.shape}")

        graph.graph[str_+ 'classifier_training'] = ReconstructionDataset(all_data, all_labels)
        return graph

    def load_regression_data_to_graph(self, graph=None, str_='post_process_lower'):
        """
        Store the live set and reliabilities on the graph as the
        regressor training dataset.
        """
        if graph is None:
            raise ValueError("Graph must be provided to load regression data.")

        live_set, live_set_prob = self.get_live_set()
        graph.graph[str_+ 'regressor_training'] = ReconstructionDataset(live_set, live_set_prob)
        return graph



class ReconstructionDataset(ABC):
    """Lightweight container pairing input samples with their targets.

    Normalises inputs and outputs to at least rank two so downstream
    classifier and regressor training can consume them uniformly.

    """

    # ---- External Methods ----

    def __init__(self, X, y):
        self.input_rank = len(X.shape)
        self.output_rank = len(y.shape)
        self.X = X if self.input_rank >= 2 else np.expand_dims(X,axis=-1)
        self.y = y if self.output_rank >=2 else np.expand_dims(y, axis=-1)
