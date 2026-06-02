"""Routes each constraint_type to its septal-backed evaluator."""
from abc import ABC
from functools import partial

from mu_F.constraints.evaluators.backward import backward_constraint_evaluator
from mu_F.constraints.evaluators.forward import forward_constraint_evaluator
from mu_F.constraints.evaluators.cost_to_go import cost_to_go_evaluator
from mu_F.constraints.evaluators.current import (
    current_constraint_surrogate,
    current_cost_surrogate,
)
from mu_F.constraints.evaluators.post_process import post_process_upper_level
from mu_F.constraints.evaluators.probability import backward_pmap
from mu_F.constraints.evaluators.process import (
    ProcessConstraintEvaluator,
    NodeCostEvaluator,
    PostProcessConstraintEvaluator,
    ForwardRootConstraintDecentralisedEvaluator,
)
from mu_F.constraints.evaluators.forward_decentralised import (
    forward_constraint_decentralised_evaluator,
)


class ConstraintEvaluator(ABC):
    """Top-level constraint dispatcher.

    Built by the integration layer, it selects a per-phase evaluator from
    constraints.evaluators.* based on constraint_type and binds the matching
    evaluate method that callers invoke.

    """

    # ---- External Methods ----

    def __init__(self, cfg, graph, node, constraint_type='process'):
        self.cfg = cfg
        self.graph = graph
        self.node = node

        if constraint_type == 'process':
            self.constraint_evaluator = ProcessConstraintEvaluator(cfg, graph, node)
            self.evaluate = self.evaluate_process

        elif constraint_type == 'forward':
            self.constraint_evaluator = partial(
                forward_constraint_evaluator, cfg=cfg, graph=graph, node=node,
            )
            self.evaluate = self.evaluate_coupling

        elif constraint_type == 'forward_decentralized':
            self.constraint_evaluator = partial(
                forward_constraint_decentralised_evaluator,
                cfg=cfg, graph=graph, node=node,
            )
            self.evaluate = self.evaluate_coupling

        elif constraint_type == 'backward':
            self.constraint_evaluator = partial(
                backward_constraint_evaluator,
                cfg=cfg, graph=graph, node=node,
            )
            self.evaluate = self.evaluate_coupling

        elif constraint_type == 'root_node_decentralized':
            self.constraint_evaluator = ForwardRootConstraintDecentralisedEvaluator(
                cfg, graph, node,
            )
            self.evaluate = self.evaluate_coupling

        elif constraint_type == 'backward_cost_to_go':
            self.constraint_evaluator = partial(
                cost_to_go_evaluator, cfg=cfg, graph=graph, node=node,
            )
            self.evaluate = self.evaluate_coupling

        elif constraint_type == 'backward_pmap':
            self.constraint_evaluator = partial(
                backward_pmap, cfg=cfg, graph=graph, node=node,
            )
            self.evaluate = self.evaluate_coupling

        elif constraint_type == 'node_cost':
            self.constraint_evaluator = NodeCostEvaluator(cfg, graph, node)
            self.evaluate = self.evaluate_process

        elif constraint_type == 'constraint_rollout':
            self.constraint_evaluator = partial(
                current_constraint_surrogate, cfg=cfg, graph=graph, node=node,
            )
            self.evaluate = self.evaluate_coupling

        elif constraint_type == 'cost_rollout':
            self.constraint_evaluator = partial(
                current_cost_surrogate, cfg=cfg, graph=graph, node=node,
            )
            self.evaluate = self.evaluate_coupling

        elif constraint_type == 'post_process_lower_level':
            self.constraint_evaluator = partial(
                backward_constraint_evaluator,
                cfg=cfg, graph=graph, node=node,
            )
            self.evaluate = self.evaluate_coupling

        elif constraint_type == 'post_process_upper_level':
            self.constraint_evaluator = partial(
                post_process_upper_level, cfg=cfg, graph=graph, node=node,
            )
            self.evaluate = self.evaluate_global

        elif constraint_type == 'post_process_evals':
            self.constraint_evaluator = PostProcessConstraintEvaluator(
                cfg, graph, node,
            )
            self.evaluate = self.evaluate_process

        else:
            raise ValueError(f"Invalid constraint_type: {constraint_type!r}")

    def evaluate_process(self, design, inputs, aux, outputs):
        """
        Bound to self.evaluate for design-level constraint types.
        """
        return self.constraint_evaluator(design, inputs, outputs, aux)

    def evaluate_coupling(self, inputs, aux, **kwargs):
        """
        Bound to self.evaluate for coupling constraint types.
        """
        return self.constraint_evaluator(inputs, aux, **kwargs)

    def evaluate_global(self):
        """
        Bound to self.evaluate for the upper-level post-process pass.
        """
        return self.constraint_evaluator()
