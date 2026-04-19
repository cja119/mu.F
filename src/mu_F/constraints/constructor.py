"""
Constraint evaluator dispatch.

Routes each `constraint_type` to the appropriate evaluator under
`constraints.evaluators.*` (the septal-backed JAX path).  There are no
legacy backends left to fall back to — every branch resolves to a septal
function.
"""
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
from mu_F.constraints.evaluators.process import (
    process_constraint_evaluator,
    node_cost_evaluator,
    post_process_constraint_evaluator,
    forward_root_constraint_decentralised_evaluator,
)
from mu_F.constraints.evaluators.forward_decentralised import (
    forward_constraint_decentralised_evaluator,
)


class constraint_evaluator(ABC):
    """
    Top-level dispatcher invoked by the integration layer.

    Parameters
    ----------
    cfg : OmegaConf DictConfig
    graph : networkx Graph
    node : int | None
    constraint_type : str
        One of the `constraint_type` handlers listed in `__init__`.
    """
    def __init__(self, cfg, graph, node, constraint_type='process'):
        self.cfg = cfg
        self.graph = graph
        self.node = node

        if constraint_type == 'process':
            self.constraint_evaluator = process_constraint_evaluator(cfg, graph, node)
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
            self.constraint_evaluator = forward_root_constraint_decentralised_evaluator(
                cfg, graph, node,
            )
            self.evaluate = self.evaluate_coupling

        elif constraint_type == 'backward_cost_to_go':
            self.constraint_evaluator = partial(
                cost_to_go_evaluator, cfg=cfg, graph=graph, node=node,
            )
            self.evaluate = self.evaluate_coupling

        elif constraint_type == 'node_cost':
            self.constraint_evaluator = node_cost_evaluator(cfg, graph, node)
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
            self.constraint_evaluator = post_process_constraint_evaluator(
                cfg, graph, node,
            )
            self.evaluate = self.evaluate_process

        else:
            raise ValueError(f"Invalid constraint_type: {constraint_type!r}")

    def evaluate_process(self, design, inputs, aux, outputs):
        return self.constraint_evaluator(design, inputs, outputs, aux)

    def evaluate_coupling(self, inputs, aux, **kwargs):
        return self.constraint_evaluator(inputs, aux, **kwargs)

    def evaluate_global(self):
        return self.constraint_evaluator()
