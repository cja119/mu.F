"""Per-node DEUS evaluation pipeline."""

from __future__ import annotations

import logging

import numpy as np
import jax.numpy as jnp
from jax.random import PRNGKey, choice

from mu_F.constraints.constructor import ConstraintEvaluator
from mu_F.unit_evaluators.constructor import SubproblemUnitWrapper

from mu_F.integration.context import EvalContext, TrainingStore
from mu_F.integration.formulations import make_formulation, _p_target
from mu_F.integration.utils import update_data


# ---------------------------------------------------------------------------
# Current — the REAL node evaluation (shared, every mode)
# ---------------------------------------------------------------------------
class CurrentEvals:
    """The node's process constraints and node cost, from the real model output.

    Both constraints consume the real `outputs` from the forward model; no
    surrogates are involved. Used by SubproblemEvaluator and RolloutEvaluator.

    """

    # ---- External Methods ----

    def __init__(self, cfg, graph, node):
        self.process_constraints = ConstraintEvaluator(cfg, graph, node, constraint_type='process')
        self.node_costs = (ConstraintEvaluator(cfg, graph, node, constraint_type='node_cost')
                           if cfg.case_study.eval_cost else None)

    def evaluate(self, ctx: EvalContext) -> EvalContext:
        """
        Write the real process constraints and node cost onto the context.
        Called by SubproblemEvaluator once the forward model has run.
        """
        ctx.process = jnp.asarray(self.process_constraints.evaluate(
            ctx.design, ctx.inputs, ctx.aux, ctx.outputs))                      # raw g
        if self.node_costs is not None:
            ctx.node_cost = self.node_costs.evaluate(ctx.design, ctx.inputs, ctx.aux, ctx.outputs)
        return ctx


# ---------------------------------------------------------------------------
# Rollout — a second entry point that reuses the real half
# ---------------------------------------------------------------------------
class RolloutEvaluator:
    """Monte-Carlo rollout for one node.

    Decides a control weather-blind, reveals the per-trajectory weather, applies
    it, then evaluates process feasibility and node cost. Reuses the shared
    forward model and CurrentEvals; only the weather draw is rollout-specific.

    """

    # ---- External Methods ----

    def __init__(self, cfg, graph, node, node_evaluator, current):
        self.cfg = cfg
        self.node = node
        self.node_evaluator = node_evaluator                 # shared forward model
        self.current = current                               # shared process + node cost
        self.rollout_costs = ConstraintEvaluator(cfg, graph, node, constraint_type='cost_rollout')

    def rollout(self, inputs, aux=None, key=None, n_samples=1):
        """
        One Markov step for N trajectories: decide weather-blind for each carried
        state, then reveal N paired weathers and apply the control.
        """
        N = int(n_samples)
        if inputs is None:
            inputs = jnp.empty((N, 0))
        if aux is None:
            aux_default = list(self.cfg.case_study.get('aux_default', []) or [])
            aux = (jnp.tile(jnp.asarray(aux_default, dtype=jnp.float32).reshape(1, -1), (inputs.shape[0], 1))
                   if aux_default else jnp.empty((inputs.shape[0], 0)))

        # Decide (weather-blind), then reveal + apply per-trajectory weather.
        decision, opt_ctg, opt_status = self.rollout_costs.evaluate(inputs, aux)    # (N, n_design)
        d = jnp.concatenate([decision, inputs, aux], axis=-1)                       # (N, n_total)
        p = self.get_uncertain_params(key, n_samples=decision.shape[0])            # (N, param)

        outputs = self.node_evaluator.get_constraints_rollout(d, p)                # (N, 1, n_out)
        p_cons = self.current.process_constraints.evaluate(decision, inputs, aux, outputs)  # (N, 1, n_g)
        n_cost = self.current.node_costs.evaluate(decision, inputs, aux, outputs)           # (N, 1, 1)

        feasible = jnp.all(p_cons >= 0.0, axis=tuple(range(1, p_cons.ndim)))        # (N,)
        per_con_frac = np.asarray(jnp.mean((p_cons >= 0.0).astype(jnp.float64), axis=(0, 1))).reshape(-1)
        worst_margin = np.asarray(jnp.min(p_cons, axis=(0, 1))).reshape(-1)
        logging.info(
            f"Rollout node {self.node}: feasible {int(jnp.sum(feasible))}/{feasible.shape[0]}, "
            f"mean node-cost {float(jnp.mean(n_cost)):.4g}, "
            f"optimiser viable {int(jnp.sum(opt_status))}/{int(opt_status.shape[0])}")
        logging.info(
            f"Rollout node {self.node} per-constraint feasible-frac={np.round(per_con_frac, 3).tolist()} "
            f"worst-margin={np.round(worst_margin, 3).tolist()}")
        return outputs, n_cost, decision, feasible

    def get_uncertain_params(self, key=None, n_samples=None):
        """
        Probabilistic draws `n_samples` weathers from the support (one per
        trajectory); deterministic returns the single best estimate.
        """
        if key is None:
            key = PRNGKey(0)
        n = int(n_samples) if n_samples is not None else int(self.cfg.max_uncertain_samples)

        if self.cfg.formulation == 'probabilistic':
            from omegaconf import OmegaConf
            raw = OmegaConf.to_container(self.cfg.case_study.parameters_samples, resolve=True)
            # Accept both per-scenario dicts and the compact parallel-array layout.
            cs, ws = [], []
            for dict_ in raw:
                c, w = dict_['c'], dict_['w']
                if isinstance(w, (list, tuple)):
                    for ci, wi in zip(c, w):
                        cs.append(list(ci) if isinstance(ci, (list, tuple)) else [ci])
                        ws.append(float(wi))
                else:
                    cs.append(list(c) if isinstance(c, (list, tuple)) else [c])
                    ws.append(float(w))
            params = jnp.asarray(cs, dtype=jnp.float64)              # (M, param_dim)
            weights = jnp.asarray(ws, dtype=jnp.float64)
            weights = weights / weights.sum()
            return choice(key, params, shape=(n,), replace=True, p=weights, axis=0)  # (n, param_dim)

        # deterministic: the single best estimate (n is 1 for a deterministic rollout)
        pbe = self.cfg.case_study.parameters_best_estimate
        val = jnp.atleast_2d(jnp.asarray(pbe[self.node]))            # (1, param_dim)
        return jnp.broadcast_to(val, (n, val.shape[1]))


# ---------------------------------------------------------------------------
# The evaluator the sampler talks to
# ---------------------------------------------------------------------------
class SubproblemEvaluator:
    """Per-node DEUS evaluator.

    Runs the fixed evaluation pipeline; the formulation is the only
    config-selected part. The DEUS callbacks are thin batching wrappers that
    SubproblemModel exposes to the sampler.

    """

    # ---- External Methods ----

    def __init__(self, cfg, graph, node, mode):
        self.cfg, self.graph, self.node, self.mode = cfg, graph, node, mode
        self.node_evaluator = SubproblemUnitWrapper(cfg, graph, node)
        self.current = CurrentEvals(cfg, graph, node)
        self.collecting = False

        if mode == 'rollout':
            # rollout reuses the real half (forward + process/cost) via RolloutEvaluator
            self.rollout_evaluator = RolloutEvaluator(cfg, graph, node, self.node_evaluator, self.current)
            self.formulation = self.store = None
        else:
            self.formulation = make_formulation(cfg, graph, node, mode)
            self.store = TrainingStore()
            self.target = _p_target(cfg, node)
            # Tuner forward/composite passes stub the forward model (outputs -> None),
            # so process / cost / ctg are skipped and only the coupling constraints remain.
            self._skip_forward = (cfg.method == 'decomposition_constraint_tuner' and mode != 'backward')
            self.io_data = None         # forward-evaluation Surrogate I/O, when enabled
            self.rollout_evaluator = None

    def rollout(self, inputs, aux=None, key=None, n_samples=1):
        """
        Delegate to RolloutEvaluator; called by SubproblemModel in rollout mode.
        """
        return self.rollout_evaluator.rollout(inputs, aux, key, n_samples)

    def evaluate(self, d, p) -> jnp.ndarray:
        """
        Run the full pipeline once for a batch of designs: forward model, real
        process / cost, the formulation's coupling, then assemble the result.
        """
        ctx = EvalContext(d=d, p=p, collecting=self.collecting)
        ctx.outputs = None if self._skip_forward else self.node_evaluator.get_constraints(d, p)   # forward ONCE
        ctx.design, ctx.inputs, ctx.aux = \
            self.node_evaluator.get_auxilliary_input_decision_split(d)
        if ctx.outputs is not None:
            self.current.evaluate(ctx)         # real (process + node cost)
            if self.collecting and self.cfg.surrogate.forward_evaluation_surrogate:
                self.io_data = update_data(self.io_data, d, p, ctx.outputs)
        self.formulation.upstream(ctx)         # Surrogate (predecessors)
        self.formulation.downstream(ctx)       # Surrogate (successors)
        self.formulation.assemble(ctx, self.store)   # transform -> result + targets
        return ctx.result

    # ---- DEUS Callbacks (batching wrappers over evaluate) ----

    def _p_feas_batched(self, d, p):
        """
        P_feas over all of `d`, chunked at `pmap_batch`; `evaluate` forwards the
        unit model once per chunk.
        """
        bs = int(self.cfg.samplers.pmap_batch)
        out, i = [], 0
        while i < d.shape[0]:
            out.append(np.asarray(self.evaluate(d[i:i + bs], p)).reshape(-1))
            i += bs
        return np.concatenate(out)

    def get_probability(self, d, p):
        """
        EFP (probabilistic) phase: raw P_feas(d), collecting the training data.
        """
        self.collecting = True
        pf = self._p_feas_batched(d, p)
        return [np.asarray(v).reshape(1, 1) for v in pf]

    def get_score_constraint(self, d, p):
        """
        Score (deterministic) phase: P_feas as the feasibility margin
        `(P_feas - p_target)/p_target`, no collection.
        """
        self.collecting = False
        pf = self._p_feas_batched(d, p)
        margin = (pf - self.target) / self.target
        return [np.asarray(v).reshape(1, 1) for v in margin]

    def s(self, d, p):
        """
        Native callback (deterministic formulation): concatenated raw
        constraints `g`, batched; DEUS counts feasibility itself.
        """
        self.collecting = True
        bs = int(self.cfg.samplers.pmap_batch)
        out, i = [], 0
        while i < d.shape[0]:
            out.append(np.asarray(self.evaluate(d[i:i + bs], p)))
            i += bs
        return np.concatenate(out, axis=0)
