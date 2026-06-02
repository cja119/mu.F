"""Evaluation formulations — the variable part of the per-node pipeline."""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod

import numpy as np
import jax.numpy as jnp
from omegaconf import OmegaConf

from mu_F.constraints.constructor import ConstraintEvaluator
from mu_F.utils import ApplyFeasibility

from mu_F.integration.context import EvalContext, TrainingStore


# ---------------------------------------------------------------------------
# Scenario weights / reliability target / shared bellman
# ---------------------------------------------------------------------------
def _expand_weights(cfg):
    """Flat list of scenario weights from the compact `parameters_samples`."""
    raw = OmegaConf.to_container(cfg.case_study.parameters_samples, resolve=True)
    ws = []
    for entry in raw:
        w = entry['w']
        if isinstance(w, (list, tuple)):
            ws.extend(float(wi) for wi in w)
        else:
            ws.append(float(w))
    return ws


def _weights(ws, n_theta):
    """Weights truncated to the support DEUS iterates and normalised."""
    ws = ws[:n_theta]
    if len(ws) != n_theta or sum(ws) == 0:
        return jnp.ones(n_theta, dtype=jnp.float64) / n_theta
    total = sum(ws)
    return jnp.asarray([wi / total for wi in ws], dtype=jnp.float64)


def _p_target(cfg, node):
    """Per-node reliability target, falling back to the shared value."""
    uw = cfg.samplers.unit_wise_target_reliability
    try:
        val = uw[node]
    except (TypeError, KeyError, IndexError):
        val = uw
    return max(float(val), 1e-8)


def _broadcast_theta(c, n_theta, repeat):
    """Rank-3-ify a coupling constraint.  Input-side families (forward/root) are
    θ-independent and repeat across the scenario axis; the per-θ backward does not."""
    c = jnp.asarray(c)
    if c.ndim == 1:
        c = c.reshape(-1, 1)
    if c.ndim == 2:
        c = jnp.expand_dims(c, axis=1)
    if repeat and c.shape[1] == 1 and n_theta > 1:
        c = jnp.repeat(c, n_theta, axis=1)
    return c


def bellman_target(ctx, keep, ctg_value, weights, gamma, node):
    """θ-averaged `node_cost + γ·CTG_down` over the kept (feasible) designs.
    Shared by both formulations; they differ only in how `keep` is chosen."""
    node_cost = ctx.node_cost[keep]
    n_theta = node_cost.shape[1]
    if ctg_value is not None:
        t = time.time()
        ctg_evals, ctg_viable = ctg_value.evaluate(ctx.outputs[keep], ctx.aux[keep])
        logging.info(f"Cost-to-go evaluation time for node {node}: {time.time() - t:.4f} seconds")
        ctg_evals = jnp.where(ctg_viable, ctg_evals, jnp.nan)
        ctg_down = jnp.sum(ctg_evals, axis=-1)                              # (n_keep, n_theta)
        ctg_target = node_cost + gamma * ctg_down[..., None]
    else:
        ctg_target = node_cost
    return jnp.sum(_weights(weights, n_theta)[None, :, None] * ctg_target, axis=1)  # (n_keep, 1)


# ---------------------------------------------------------------------------
# Formulations — upstream / downstream / assemble
# ---------------------------------------------------------------------------
class Formulation(ABC):
    """Interface for the config-selected part of the per-node pipeline.

    SubproblemEvaluator drives every formulation through the same three steps
    (upstream / downstream / assemble); subclasses supply the coupling maths.

    """

    # ---- Base Methods ----

    @abstractmethod
    def upstream(self, ctx: EvalContext) -> EvalContext: ...
    @abstractmethod
    def downstream(self, ctx: EvalContext) -> EvalContext: ...
    @abstractmethod
    def assemble(self, ctx: EvalContext, store: TrainingStore) -> EvalContext: ...


class ProbabilisticFormulation(Formulation):
    r"""Direct-probability feasibility: `P_feas = Σ_θ w_θ · 1[process≥0] · P_up · P_down`.

    `P_down` is the successors' probability_map and `P_up` the predecessor
    coupling (only the backward `P_up = 1` case is derived). `downstream`
    computes P_feas and the CTG; `assemble` is then a pure store.

    """

    # ---- External Methods ----

    def __init__(self, forward, root, pmap_value, ctg_value,
                 weights, target, gamma, node):
        self.forward = forward          # upstream coupling (predecessors), or None
        self.root = root                # upstream coupling (source node), or None
        self.pmap_value = pmap_value    # downstream feasibility Surrogate (P_down)
        self.ctg_value = ctg_value      # downstream cost Surrogate (CTG_down), or None
        self.weights = weights
        self.target = target
        self.gamma = gamma
        self.node = node

    def upstream(self, ctx: EvalContext) -> EvalContext:
        """
        Predecessor coupling P_up; only the backward case (P_up = 1) is derived.
        """
        # Backward mode has no predecessor coupling -> P_up = 1.
        if self.forward is None and self.root is None:
            ctx.upstream = jnp.ones((ctx.outputs.shape[0], ctx.outputs.shape[1]), dtype=jnp.float64)
            return ctx
        raise NotImplementedError(
            "Probabilistic upstream (forward-mode) coupling is not derived yet.")

    def downstream(self, ctx: EvalContext) -> EvalContext:
        """
        Successor coupling P_down, then P_feas over all scenarios and the CTG for
        the reliable designs.
        """
        n_d, n_theta = ctx.outputs.shape[0], ctx.outputs.shape[1]
        w = _weights(self.weights, n_theta)
        proc_ok = jnp.all(ctx.process >= 0.0, axis=-1)                                  # (n_d, n_theta)
        own = proc_ok.astype(jnp.float64) * ctx.upstream

        # Solve P_down only where process-feasible in some scenario; elsewhere P_feas = 0.
        if self.pmap_value is None:
            ctx.downstream = jnp.ones((n_d, n_theta), dtype=jnp.float64)
        else:
            p_down = jnp.zeros((n_d, n_theta), dtype=jnp.float64)
            feas = jnp.where(jnp.any(proc_ok, axis=1))[0]
            if feas.size > 0:
                t = time.time()
                p_evals, _ = self.pmap_value.evaluate(ctx.outputs[feas], ctx.aux[feas])
                logging.info(f"Backward probability evaluation time for node {self.node}: {time.time() - t:.4f} seconds")
                p_down = p_down.at[feas].set(jnp.prod(jnp.asarray(p_evals), axis=-1))
            ctx.downstream = p_down

        # P_feas is now exact; the CTG is solved only for the reliable (feasible-passing) designs.
        ctx.p_feas = jnp.sum(w[None, :] * own * ctx.downstream, axis=-1)                 # (n_d,)
        ctx.keep = jnp.where(ctx.p_feas >= self.target)[0]                               # feasible-passing
        ctx.ctg = (bellman_target(ctx, ctx.keep, self.ctg_value, self.weights, self.gamma, self.node)
                   if (ctx.keep.size > 0 and ctx.node_cost is not None) else None)
        return ctx

    def assemble(self, ctx: EvalContext, store: TrainingStore) -> EvalContext:
        """
        Pure store — P_feas and the CTG were already computed in downstream.
        """
        ctx.result = ctx.p_feas
        if ctx.collecting:
            store.add('pmap', ctx.d, ctx.outputs, np.asarray(ctx.p_feas).reshape(-1, 1))   # ALL designs
            if ctx.ctg is not None:
                store.add('ctg', ctx.d[ctx.keep], ctx.outputs[ctx.keep],
                          np.asarray(ctx.ctg).reshape(-1, 1))
        return ctx


class DeterministicFormulation(Formulation):
    r"""Native per-scenario path returning the concatenated raw constraints `g`.

    `upstream` evaluates the input-side coupling and `downstream` the backward
    coupling; `assemble` concatenates the raw `g` for the native `s` callback.
    The CTG shares the bellman, gated by `ApplyFeasibility`.

    """

    # ---- External Methods ----

    def __init__(self, cfg, forward, decentralised, root, backward,
                 ctg_value, weights, gamma, graph, node):
        self.cfg = cfg
        self.forward = forward                # input-side coupling (predecessors)
        self.decentralised = decentralised
        self.root = root
        self.backward = backward              # output-side coupling (successors)
        self.ctg_value = ctg_value
        self.weights = weights
        self.gamma = gamma
        self.graph = graph
        self.node = node

    def upstream(self, ctx: EvalContext) -> EvalContext:
        """
        Input-side coupling (forward / decentralised / root), θ-broadcast.
        """
        n_theta = ctx.outputs.shape[1]
        cons = []
        if self.forward is not None and self.graph.in_degree(self.node) > 0:
            cons.append(_broadcast_theta(self.forward.evaluate(ctx.inputs, ctx.aux), n_theta, True))
        if self.decentralised is not None and self.graph.in_degree(self.node) > 0:
            cons.append(_broadcast_theta(self.decentralised.evaluate(ctx.design, ctx.aux), n_theta, True))
        if self.root is not None and self.graph.in_degree(self.node) == 0:
            cons.append(_broadcast_theta(self.root.evaluate(ctx.design, ctx.aux), n_theta, True))
        ctx.upstream = jnp.concatenate(cons, axis=-1) if cons else None
        return ctx

    def downstream(self, ctx: EvalContext) -> EvalContext:
        """
        Output-side backward coupling, then the CTG over the feasible designs.
        """
        if self.backward is not None and self.graph.out_degree(self.node) > 0:
            t = time.time()
            evals, _ = self.backward.evaluate(ctx.outputs, ctx.aux)
            logging.info(f"Backward constraint evaluation time for node {self.node}: {time.time() - t:.4f} seconds")
            ctx.downstream = _broadcast_theta(evals, ctx.outputs.shape[1], False)
        else:
            ctx.downstream = None
        # CTG over the feasible designs (shared bellman); gate = ApplyFeasibility.
        if ctx.node_cost is not None:
            keep = self._feasible(ctx, self._concat(ctx))
            ctx.keep = keep
            ctx.ctg = (bellman_target(ctx, keep, self.ctg_value, self.weights, self.gamma, self.node)
                       if keep.size > 0 else None)
        return ctx

    def assemble(self, ctx: EvalContext, store: TrainingStore) -> EvalContext:
        """
        Concatenate the raw constraints for the native `s` callback and store the
        training data.
        """
        cons = self._concat(ctx)
        ctx.result = cons                              # concat-g for the native `s` callback
        if ctx.collecting:
            if self.cfg.surrogate.classifier:
                store.add('classifier', ctx.d, ctx.outputs, cons)
            if ctx.ctg is not None:
                store.add('ctg', ctx.d[ctx.keep], ctx.outputs[ctx.keep],
                          np.asarray(ctx.ctg).reshape(-1, 1))
        return ctx

    # ---- Private Methods ----

    def _concat(self, ctx: EvalContext):
        """
        Stack the process and coupling constraints into the single `g` vector.
        """
        return jnp.concatenate(
            [c for c in (ctx.process, ctx.upstream, ctx.downstream) if c is not None], axis=-1)

    def _feasible(self, ctx: EvalContext, cons):
        """
        Indices of designs passing the full constraint set via ApplyFeasibility.
        """
        _, _, cond = ApplyFeasibility(
            [ctx.outputs], [cons], self.cfg, self.node, self.cfg.formulation
        ).get_feasible(return_indices=True)
        mask = jnp.asarray(cond[0]).ravel()
        return jnp.where(mask != 0)[0]


# ---------------------------------------------------------------------------
# Factory: config key -> formulation.  Wires its upstream/downstream surrogates.
# ---------------------------------------------------------------------------
def _selected_procedure(cfg):
    """The chosen formulation key.  Reads the explicit `evaluation_procedure`
    config once it exists (increment 5); until then, infers it from the legacy
    `direct_probability` flag."""
    key = cfg.samplers.get('evaluation_procedure', None)
    if key:
        return key
    direct = (bool(cfg.samplers.deus.get('direct_probability', False))
              and cfg.formulation == 'probabilistic'
              and bool(cfg.surrogate.probability_map))
    return 'probabilistic_feasibility' if direct else 'deterministic'


def _coupling(cfg, graph, node, ctype):
    """A constraint evaluator of the given coupling type for this node."""
    return ConstraintEvaluator(cfg, graph, node, constraint_type=ctype)


def _ctg_value(cfg, graph, node):
    """Backward cost-to-go Surrogate — only where there's a cost and a successor."""
    if cfg.case_study.eval_cost and graph.out_degree(node) > 0:
        return _coupling(cfg, graph, node, 'backward_cost_to_go')
    return None


def _build_probabilistic(cfg, graph, node, mode):
    """Wire a ProbabilisticFormulation with its coupling surrogates."""
    forward = root = None
    if mode in ('forward', 'forward-backward', 'backward-forward'):
        forward = _coupling(cfg, graph, node, 'forward')
        if mode == 'backward-forward' and cfg.method == 'decomposition_constraint_tuner':
            root = _coupling(cfg, graph, node, 'root_node_decentralized')
    return ProbabilisticFormulation(
        forward=forward, root=root,
        pmap_value=_coupling(cfg, graph, node, 'backward_pmap'),
        ctg_value=_ctg_value(cfg, graph, node),
        weights=_expand_weights(cfg), target=_p_target(cfg, node),
        gamma=float(cfg.case_study.discount_factor), node=node)


def _build_deterministic(cfg, graph, node, mode):
    """Wire a DeterministicFormulation with its coupling evaluators."""
    tuner = (cfg.method == 'decomposition_constraint_tuner')
    forward = decentralised = root = backward = None
    if mode in ('forward', 'forward-backward', 'backward-forward'):
        forward = _coupling(cfg, graph, node, 'forward')
    if mode == 'backward' or (mode in ('forward-backward', 'backward-forward') and not tuner):
        backward = _coupling(cfg, graph, node, 'backward')
    if mode in ('forward-backward', 'backward-forward') and tuner:
        decentralised = _coupling(cfg, graph, node, 'forward_decentralized')
        if mode == 'backward-forward':
            root = _coupling(cfg, graph, node, 'root_node_decentralized')
    return DeterministicFormulation(
        cfg=cfg, forward=forward, decentralised=decentralised, root=root, backward=backward,
        ctg_value=_ctg_value(cfg, graph, node), weights=_expand_weights(cfg),
        gamma=float(cfg.case_study.discount_factor), graph=graph, node=node)


_FORMULATIONS = {
    'probabilistic_feasibility': _build_probabilistic,
    'deterministic': _build_deterministic,
}


def make_formulation(cfg, graph, node, mode) -> Formulation:
    """Build the config-selected formulation, wiring its constraint evaluators."""
    procedure = _selected_procedure(cfg)
    build = _FORMULATIONS.get(procedure)
    if build is None:
        raise ValueError(f"Unknown evaluation_procedure: {procedure!r}")
    return build(cfg, graph, node, mode)
