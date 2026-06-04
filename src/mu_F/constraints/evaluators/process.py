"""Vmap-based constraint evaluators for unit-level and root-node feasibility."""
from __future__ import annotations

from abc import ABC
from functools import partial

import jax.numpy as jnp
from jax import vmap, jit

from mu_F._types import typecheck, OutputScen, ConstraintScen


__all__ = [
    "ConstraintEvaluatorBase",
    "ProcessConstraintEvaluator",
    "NodeCostEvaluator",
    "PostProcessConstraintEvaluator",
    "ForwardRootConstraintDecentralisedEvaluator",
]


class ConstraintEvaluatorBase(ABC):
    """Minimal base for the vmap evaluator family.

    Holds (cfg, graph, node) and exposes a shaping_function that respects
    cfg.samplers.notion_of_feasibility. Subclasses implement evaluate and
    optionally load_unit_constraints.

    """

    # ---- External Methods ----

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
        """
        Abstract constraint evaluation; implemented by each subclass.
        """
        raise NotImplementedError

    def load_unit_constraints(self):
        """
        Abstract constraint loader; implemented by each subclass.
        """
        raise NotImplementedError


class ProcessConstraintEvaluator(ConstraintEvaluatorBase):
    """Evaluate unit-level process constraints on a trajectory via vmap.

    Loads the per-node constraint callables off the graph and applies them
    to a dynamics profile, returning a stacked feasibility tensor.

    """

    # ---- External Methods ----

    def __init__(self, cfg, graph, node):
        super().__init__(cfg, graph, node)
        if cfg.case_study.vmap_evaluations:
            self.vmap_evaluation()

    def __call__(self, design, inputs, dynamics_profile, aux):
        """
        Evaluation entry point invoked on the assembled trajectory.
        """
        return self.evaluate(design, inputs, aux, dynamics_profile)

    @typecheck
    def evaluate(self, design_args, input_args, aux_args,
                 dynamics_profile: OutputScen) -> ConstraintScen:
        """
        Apply each loaded constraint to the profile and stack the results.
        """
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
        """
        Fetch the per-node constraint callables (vmap'd or plain) off the graph.
        """
        if self.cfg.case_study.vmap_evaluations:
            return list(self.graph.nodes[self.node]['constraints_vmap'].copy())
        return list(self.graph.nodes[self.node]['constraints'].copy())

    def vmap_evaluation(self):
        """
        Vectorise per-constraint functions and cache them on the graph node.
        """
        constraints = self.graph.nodes[self.node]['constraints'].copy()
        cons = [
            jit(vmap(jit(vmap(partial(constraint, cfg=self.cfg.model),
                              in_axes=(0), out_axes=0)),
                     in_axes=(1), out_axes=1))
            for constraint in constraints
        ]
        self.graph.nodes[self.node]['constraints_vmap'] = cons


class NodeCostEvaluator(ProcessConstraintEvaluator):
    """Per-node cost evaluator.

    Reuses the ProcessConstraintEvaluator machinery but reads node_cost /
    node_cost_vmap off the graph instead of the constraint callables.

    """

    # ---- External Methods ----

    def __init__(self, cfg, graph, node):
        super().__init__(cfg, graph, node)

    def load_unit_constraints(self):
        """
        Fetch the per-node cost callables (vmap'd or plain) off the graph.
        """
        if self.cfg.case_study.vmap_evaluations:
            return list(self.graph.nodes[self.node]['node_cost_vmap'].copy())
        return list(self.graph.nodes[self.node]['node_cost'].copy())

    def vmap_evaluation(self):
        """
        Vectorise the cost callables and cache them on the graph node.
        """
        constraints = self.graph.nodes[self.node]['node_cost'].copy()
        cons = [
            jit(vmap(jit(vmap(partial(constraint, cfg=self.cfg.model),
                              in_axes=(0), out_axes=0)),
                     in_axes=(1), out_axes=1))
            for constraint in constraints
        ]
        self.graph.nodes[self.node]['node_cost_vmap'] = cons


class PostProcessConstraintEvaluator(ProcessConstraintEvaluator):
    """Post-processing constraint evaluator.

    Reuses the ProcessConstraintEvaluator machinery but reads
    post_process_constraints off the graph node.

    """

    # ---- External Methods ----

    def __init__(self, cfg, graph, node):
        super().__init__(cfg, graph, node)
        if cfg.case_study.vmap_evaluations:
            self.vmap_evaluation()

    def load_unit_constraints(self):
        """
        Fetch the post-process constraint callables (vmap'd or plain) off the graph.
        """
        if self.cfg.case_study.vmap_evaluations:
            return list(self.graph.nodes[self.node]['post_process_constraints_vmap'].copy())
        return list(self.graph.nodes[self.node]['post_process_constraints'].copy())

    def vmap_evaluation(self):
        """
        Vectorise the post-process callables and cache them on the graph node.
        """
        constraints = self.graph.nodes[self.node]['post_process_constraints'].copy()
        cons = [
            jit(vmap(jit(vmap(partial(constraint, cfg=self.cfg.model),
                              in_axes=(0), out_axes=0)),
                     in_axes=(1), out_axes=1))
            for constraint in constraints
        ]
        self.graph.nodes[self.node]['post_process_constraints_vmap'] = cons


class ForwardRootConstraintDecentralisedEvaluator(ConstraintEvaluatorBase):
    """Root-node decentralised feasibility via vmap'd classifier evaluation.

    Used for the first node in a decentralised Reconstruction: no inputs flow
    in from predecessors, so the node's classifier is evaluated across the
    decision + aux batch with no NLP or pmap.

    """

    # ---- External Methods ----

    def __init__(self, cfg, graph, node):
        super().__init__(cfg, graph, node)

    def __call__(self, inputs, aux):
        """
        Evaluation entry point for the root node's decentralised feasibility.
        """
        return self.evaluate(inputs, aux)

    def evaluate(self, design, aux):
        """
        Delegate to the vmap'd classifier evaluation.
        """
        return self.evaluate_vmap(design, aux)

    def evaluate_vmap(self, decisions, aux):
        """
        Build and run the vmap'd classifier over the decision + aux batch.
        """
        constraints, inputs = self.prepare_forward_problem(jnp.hstack([decisions, aux]))
        g_vals = constraints(inputs)
        return self.shaping_function(g_vals.reshape(-1, 1))

    def prepare_forward_problem(self, v_2):
        """
        Assemble the vmap'd, back-off-shifted classifier and pass-through inputs.
        """
        # Inputs passed through in real units; the node's classifier
        # callable self-scales.
        inputs = v_2
        node_constraints = vmap(
            partial(
                lambda x, b: -self.graph.nodes[self.node]["classifier"](x) - b,
                b=self.graph.nodes[self.node]['constraint_backoff'],
            ),
            in_axes=0, out_axes=0,
        )
        return node_constraints, inputs
