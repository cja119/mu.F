"""Per-node subproblem model exposing the DEUS callbacks."""
from abc import ABC

import numpy as np
import logging

from mu_F.integration.evaluation import SubproblemEvaluator
from mu_F.integration.utils import _sqp_evaluators_for_node


class SubproblemModel(ABC):
    """Thin per-node wrapper over the per-node evaluator.

    Holds a SubproblemEvaluator and exposes the DEUS callbacks (s,
    get_probability, get_score_constraint) and the Monte-Carlo rollout,
    delegating each to it.

    """

    # ---- External Methods ----

    def __init__(self, unit_index, cfg, G, mode, max_devices):
        self.function_evaluations = 0
        self.unit_index = unit_index
        self.cfg, self.G = cfg, G
        self.mode = mode
        self.max_devices = max_devices
        self.evaluator = SubproblemEvaluator(cfg, G, unit_index, mode)

    def s(self, d, p):
        """
        Native DEUS callback: concatenated per-design constraints, with a
        per-call SQP feasibility / convergence log.
        """
        # snapshot SQP counters so the per-call delta logs as success-counts
        snapshot = [
            (label, ev, ev.n_sqp_calls, ev.n_sqp_viable, ev.n_sqp_converged)
            for label, ev in _sqp_evaluators_for_node(self.G, self.unit_index)
        ]
        g = np.asarray(self.evaluator.s(d, p))                 # (n_d, n_theta, n_g)
        n_theta, n_g = g.shape[-2], g.shape[-1]
        self.function_evaluations += g.shape[0] * g.shape[1]

        feas_diag, conv_diag = [], []
        for label, ev, prev_calls, prev_viable, prev_conv in snapshot:
            d_calls = ev.n_sqp_calls - prev_calls
            if d_calls > 0:
                d_viable = ev.n_sqp_viable - prev_viable
                d_conv   = ev.n_sqp_converged - prev_conv
                feas_diag.append(f"{label} {d_viable}/{d_calls} ({d_viable/d_calls*100:.0f}%)")
                conv_diag.append(f"{label} {d_conv}/{d_calls} ({d_conv/d_calls*100:.0f}%)")
        if feas_diag:
            logging.info("[s] SQP feasible:  " + ", ".join(feas_diag))
            logging.info("[s] SQP converged: " + ", ".join(conv_diag))

        return [g[i, :, :].reshape(n_theta, n_g) for i in range(g.shape[0])]

    def get_constraints(self, d, p):
        """
        Concat-g callback DEUS reads in the deterministic phase.
        """
        return self.s(d, p)

    def get_probability(self, d, p):
        """
        EFP (probabilistic) phase callback, delegated to the evaluator.
        """
        out = self.evaluator.get_probability(d, p)
        self.function_evaluations += len(out)
        return out

    def get_score_constraint(self, d, p):
        """
        Score (deterministic) phase callback, delegated to the evaluator.
        """
        out = self.evaluator.get_score_constraint(d, p)
        self.function_evaluations += len(out)
        return out

    def rollout(self, inputs, aux=None, key=None, n_samples=1):
        """
        Monte-Carlo rollout entry point called by the orchestrator.
        """
        return self.evaluator.rollout(inputs, aux, key, n_samples)
