"""
Vmap-based constraint evaluators.

Pure-JAX evaluators that apply already-compiled constraint callables to a
dynamics profile (or decision batch) via `vmap`.  No NLP solver is
involved, no `pmap` — this is the quiet side of the evaluator family,
surviving unchanged from the pre-septal layout.

Contents
--------
- `ConstraintEvaluatorBase`                   — minimal cfg/graph/node base + `shaping_function`
- `process_constraint_evaluator`              — unit-level process constraints over a dynamics trajectory
- `node_cost_evaluator`                       — same pattern; reads `node_cost` instead of `constraints`
- `post_process_constraint_evaluator`         — same pattern; reads `post_process_constraints`
- `forward_root_constraint_decentralised_evaluator`
      vmap'd classifier evaluation for root-node decentralised feasibility
"""
from __future__ import annotations

from abc import ABC
from functools import partial

import jax.numpy as jnp
from jax import vmap, jit


__all__ = [
    "ConstraintEvaluatorBase",
    "process_constraint_evaluator",
    "node_cost_evaluator",
    "post_process_constraint_evaluator",
    "forward_root_constraint_decentralised_evaluator",
]


class ConstraintEvaluatorBase(ABC):
    """
    Minimal base for the vmap evaluator family.

    Holds `(cfg, graph, node)` and exposes a `shaping_function` that respects
    `cfg.samplers.notion_of_feasibility`.  Subclasses implement `evaluate`
    and optionally `load_unit_constraints`.
    """
    def __init__(self, cfg, graph, node):
        self.cfg = cfg
        self.graph = graph
        self.node = node

        if cfg.samplers.notion_of_feasibility == 'positive':
            # g(x) >= 0 convention — return as-is so minimisers push g positive.
            self.shaping_function = lambda x: x
        elif cfg.samplers.notion_of_feasibility == 'negative':
            # g(x) <= 0 convention — flip sign.
            self.shaping_function = lambda x: -x
        else:
            raise ValueError(
                f"Invalid cfg.samplers.notion_of_feasibility: "
                f"{cfg.samplers.notion_of_feasibility!r}"
            )

    def evaluate(self, dynamics_profile):
        raise NotImplementedError

    def load_unit_constraints(self):
        raise NotImplementedError


class process_constraint_evaluator(ConstraintEvaluatorBase):
    """
    Evaluate unit-level process constraints on a trajectory via vmap.
    """
    def __init__(self, cfg, graph, node):
        super().__init__(cfg, graph, node)
        if cfg.case_study.vmap_evaluations:
            self.vmap_evaluation()

    def __call__(self, design, inputs, dynamics_profile, aux):
        return self.evaluate(design, inputs, aux, dynamics_profile)

    def evaluate(self, design_args, input_args, aux_args, dynamics_profile):
        constraints = self.load_unit_constraints()
        if len(constraints) > 0:
            constraint_holder = []
            for cons_fn in constraints:
                g = cons_fn(dynamics_profile)
                if g.ndim < 2:
                    g = g.reshape(-1, 1)
                if g.ndim < 3:
                    g = jnp.expand_dims(g, axis=-1)
                constraint_holder.append(g)
            return jnp.concatenate(constraint_holder, axis=-1)
        # No unit-level constraints — return an all-feasible tensor shaped
        # like the profile so downstream masking stays well-defined.
        return self.shaping_function(jnp.ones(dynamics_profile.shape))

    def load_unit_constraints(self):
        if self.cfg.case_study.vmap_evaluations:
            return list(self.graph.nodes[self.node]['constraints_vmap'].copy())
        return list(self.graph.nodes[self.node]['constraints'].copy())

    def vmap_evaluation(self):
        """Vectorise per-constraint functions and cache on the graph node."""
        constraints = self.graph.nodes[self.node]['constraints'].copy()
        cons = [
            jit(vmap(jit(vmap(partial(constraint, cfg=self.cfg.model),
                              in_axes=(0), out_axes=0)),
                     in_axes=(1), out_axes=1))
            for constraint in constraints
        ]
        self.graph.nodes[self.node]['constraints_vmap'] = cons


class node_cost_evaluator(process_constraint_evaluator):
    """Per-node cost evaluator; reads `node_cost` / `node_cost_vmap` instead."""

    def __init__(self, cfg, graph, node):
        super().__init__(cfg, graph, node)

    def load_unit_constraints(self):
        if self.cfg.case_study.vmap_evaluations:
            return list(self.graph.nodes[self.node]['node_cost_vmap'].copy())
        return list(self.graph.nodes[self.node]['node_cost'].copy())

    def vmap_evaluation(self):
        constraints = self.graph.nodes[self.node]['node_cost'].copy()
        cons = [
            jit(vmap(jit(vmap(partial(constraint, cfg=self.cfg.model),
                              in_axes=(0), out_axes=0)),
                     in_axes=(1), out_axes=1))
            for constraint in constraints
        ]
        self.graph.nodes[self.node]['node_cost_vmap'] = cons


class post_process_constraint_evaluator(process_constraint_evaluator):
    """Post-processing constraint evaluator; reads `post_process_constraints`."""

    def __init__(self, cfg, graph, node):
        super().__init__(cfg, graph, node)
        if cfg.case_study.vmap_evaluations:
            self.vmap_evaluation()

    def load_unit_constraints(self):
        if self.cfg.case_study.vmap_evaluations:
            return list(self.graph.nodes[self.node]['post_process_constraints_vmap'].copy())
        return list(self.graph.nodes[self.node]['post_process_constraints'].copy())

    def vmap_evaluation(self):
        constraints = self.graph.nodes[self.node]['post_process_constraints'].copy()
        cons = [
            jit(vmap(jit(vmap(partial(constraint, cfg=self.cfg.model),
                              in_axes=(0), out_axes=0)),
                     in_axes=(1), out_axes=1))
            for constraint in constraints
        ]
        self.graph.nodes[self.node]['post_process_constraints_vmap'] = cons


class forward_root_constraint_decentralised_evaluator(ConstraintEvaluatorBase):
    """
    Root-node decentralised feasibility via vmap'd classifier evaluation.

    Used for the first node in a decentralised reconstruction — no inputs
    flow in from predecessors, so we just evaluate the node's classifier
    across the decision + aux batch.  No NLP, no pmap.
    """
    def __init__(self, cfg, graph, node):
        super().__init__(cfg, graph, node)

    def __call__(self, inputs, aux):
        return self.evaluate(inputs, aux)

    def evaluate(self, design, aux):
        return self.evaluate_vmap(design, aux)

    def evaluate_vmap(self, decisions, aux):
        constraints, inputs = self.prepare_forward_problem(jnp.hstack([decisions, aux]))
        g_vals = constraints(inputs)
        return self.shaping_function(g_vals.reshape(-1, 1))

    def prepare_forward_problem(self, v_2):
        if self.cfg.solvers.standardised:
            inputs = jnp.vstack([
                self.standardise_model_decisions([v for v in v_2[i]], self.node)
                for i in range(v_2.shape[0])
            ])
        else:
            inputs = v_2
        node_constraints = vmap(
            partial(
                lambda x, b: -self.graph.nodes[self.node]["classifier"](x) - b,
                b=self.graph.nodes[self.node]['constraint_backoff'],
            ),
            in_axes=0, out_axes=0,
        )
        return node_constraints, inputs

    def standardise_model_decisions(self, decisions, in_node):
        # Three-tier fallback preserved from the casadi-path original:
        #   1. node-level classifier scaler (preferred),
        #   2. generic `x_scalar`,
        #   3. identity (no scaler available).
        try:
            mean = self.graph.nodes[in_node]['classifier_x_scalar'].mean
            std  = self.graph.nodes[in_node]['classifier_x_scalar'].std
            return [
                (decision - m) / s
                for i, (m, s, decision) in enumerate(zip(mean, std, decisions))
                if i < len(decisions)
            ]
        except Exception:
            try:
                mean = self.graph.nodes[in_node]['x_scalar'].mean
                std  = self.graph.nodes[in_node]['x_scalar'].std
                return [(decision - mean) / std for decision in decisions]
            except Exception:
                return decisions
