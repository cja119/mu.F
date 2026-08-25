"""Design-space Decomposition orchestration — the per-pass node run loop."""
import pandas as pd
from jax.random import PRNGKey, fold_in
import jax.numpy as jnp
import numpy as np
import logging
from jax import clear_caches
import gc
import contextlib

from mu_F.surrogate.augmentation import augment_training_data
from mu_F.samplers.constructor import ConstructDeusProblem
from mu_F.samplers.utils import (create_problem_description_deus, resolve_aux_override,
                                 aux_var_types)
from deus import DEUS
from mu_F.utils import save_graph

from mu_F.integration.utils import _StdoutToLog, _log_sqp_convergence, _get_rollout_action_columns
from mu_F.integration.data import del_data, process_data_forward
from mu_F.integration.surrogates import (
    surrogate_training_forward, probability_map_construction,
    classifier_construction, cluster_classifier_construction, ctg_surrogate_construction,
    margin_surrogate_construction)
from mu_F.integration.model import SubproblemModel


class ApplyDecomposition:
    """Run loop for one decomposition pass over the graph.

    Walks the nodes in precedence order: a characterisation pass solves each
    node with DEUS, collects data and trains surrogates; a rollout pass walks
    the nodes as one Monte-Carlo chain. Per-node machinery lives in `model` /
    `evaluation`.

    """

    # ---- External Methods ----

    def __init__(self, cfg, graph, precedence_order, mode: str = "forward",
                 iterate=0, max_devices=1, total_iterates=1):
        self.cfg = cfg
        self.graph = graph
        self.precedence_order = precedence_order
        self.mode = mode
        self.iterate = iterate
        self.max_devices = max_devices
        self.total_iterates = total_iterates

    def run(self):
        """
        Entry point: dispatch to a rollout or characterisation pass by mode.
        """
        nodes = self._node_order()
        return self._rollout(nodes) if self.mode == 'rollout' else self._characterise(nodes)

    # ---- Private Methods ----

    def _node_order(self):
        """The nodes in the order this pass visits them."""
        if self.mode in ('backward', 'forward-backward'):
            return list(reversed(self.precedence_order))
        if self.mode in ('forward', 'backward-forward', 'rollout'):
            return list(self.precedence_order)
        raise ValueError(f"Mode {self.mode} not recognized. "
                         "Use 'forward', 'backward', 'forward-backward' or 'backward-forward'.")

    # -----------------------------------------------------------------------
    # Characterisation pass
    # -----------------------------------------------------------------------
    def _characterise(self, nodes):
        """Build, solve, collect, train and persist each node in turn."""
        cfg = self.cfg
        for node in nodes:
            logging.info(f'------- Characterising node {node} according to precedence order -------')
            model = SubproblemModel(node, cfg, self.graph, mode=self.mode, max_devices=self.max_devices)

            feasible_set = self._solve_node(node, model)
            if feasible_set is None:                       # no feasible region -> stop
                self.graph.graph['terminate'] = True
                return self.graph

            self._collect_and_train(node, model, feasible_set)

            save_graph(self.graph.copy(), f"{self.mode}_iterate_{self.iterate}_node_{node}")
            self.graph = del_data(self.graph, node)
            del model
            gc.collect()
            clear_caches()
        return self.graph

    def _solve_node(self, node, model):
        """Run DEUS for the node; return its feasible set, or None if empty."""
        cfg, graph = self.cfg, self.graph
        problem_sheet = create_problem_description_deus(cfg, model, graph, node, self.mode)
        solver = ConstructDeusProblem(DEUS, problem_sheet, model)
        with contextlib.redirect_stdout(_StdoutToLog(logging.getLogger())):
            solver.solve()
        if cfg.method == 'decomposition_constraint_tuner':
            graph.nodes[node]['log_evidence'] = solver.get_log_evidence()

        feasible, _ = solver.get_solution()
        feasible_set = feasible[0]
        if feasible_set.size == 0:
            logging.warning(f"No feasible set found for node {node}. Terminating simulation.")
            return None
        graph.nodes[node]["fn_evals"] += model.function_evaluations
        return feasible_set

    def _collect_and_train(self, node, model, feasible_set):
        """Augment around the feasibles, write training data to the graph, train surrogates."""
        cfg, graph, mode = self.cfg, self.graph, self.mode
        if mode != 'backward-forward':
            augment_training_data(cfg, graph, node, model)
        _log_sqp_convergence(graph, node)
        process_data_forward(cfg, graph, node, model, feasible_set, mode)
        self._train_surrogates(node, model)

    def _train_surrogates(self, node, model):
        """Train whichever surrogates this pass and config call for."""
        cfg, graph, mode, iterate = self.cfg, self.graph, self.mode, self.iterate

        direct_prob = (bool(cfg.samplers.deus.get('direct_probability', False))
                       and cfg.formulation == 'probabilistic')
        tuner_pass = (mode == 'backward-forward')

        # forward-evaluation Surrogate — only the end nodes of a pass propagate it
        end_node = ((mode == 'forward' and graph.out_degree(node) != 0)
                    or (mode == 'backward' and graph.in_degree(node) == 0))
        if end_node and cfg.surrogate.forward_evaluation_surrogate:
            surrogate_training_forward(cfg, graph, node)

        # cost-to-go Surrogate (ctg targets are written by process_data_forward from the store)
        if cfg.case_study.eval_cost and not tuner_pass:
            ctg_surrogate_construction(cfg, graph, node, iterate)

        # feasibility Surrogate: classifier (deterministic) or probability_map (direct).
        if cfg.surrogate.classifier and not tuner_pass and not direct_prob:
            if cfg.samplers.ns.get('rejector', '') == 'sumb-xmeans':
                cluster_classifier_construction(cfg, graph, node, iterate)
            else:
                classifier_construction(cfg, graph, node, iterate)
            # signed-margin regressor: near-boundary feasibility with a per-constraint backoff
            if cfg.surrogate.get('margin_regressor', False):
                margin_surrogate_construction(cfg, graph, node, iterate)
        if cfg.surrogate.probability_map and not tuner_pass and direct_prob:
            probability_map_construction(cfg, graph, node, iterate)

    # -----------------------------------------------------------------------
    # Rollout pass
    # -----------------------------------------------------------------------
    def _rollout(self, nodes):
        """Walk the nodes as one Monte-Carlo chain of N trajectories."""
        cfg = self.cfg
        n = int(cfg.case_study.get('num_rollout_samples', 1)) if cfg.formulation == 'probabilistic' else 1
        key = PRNGKey(int(cfg.case_study.get('rollout_seed', 0)))
        cost = jnp.zeros(n)
        feasible = jnp.ones(n, dtype=bool)
        node_input = self._rollout_seed(nodes[0], n)

        # Only a solved global_var is carried from the root; a parametrisation is
        # re-resolved at every node, so aux_evaluation can vary along the horizon.
        carries_aux = (resolve_aux_override(cfg) is None
                       and any(t == 'global_var' for t in aux_var_types(cfg)))
        carried_aux = None
        aux_trace = []

        steps = []
        for node in nodes:
            model = SubproblemModel(node, cfg, self.graph, mode='rollout', max_devices=self.max_devices)
            if node_input is not None:
                if node_input.ndim < 3:
                    node_input = jnp.expand_dims(node_input, axis=0)
                node_input = jnp.squeeze(node_input, axis=1)     # drop the size-1 scenario axis -> (N, state)

            state_in = ('root' if node_input is None
                        else np.round(np.asarray(node_input[0]).reshape(-1), 2).tolist())
            aux_in = carried_aux if (carries_aux and self.graph.in_degree(node) > 0) else None
            outputs, n_cost, decision, node_feasible, p_cons, aux_used = model.rollout(
                node_input, aux=aux_in, key=fold_in(key, int(node)), n_samples=n)
            aux_trace.append(np.asarray(aux_used[0]).reshape(-1) if aux_used.size else np.zeros(0))
            if carries_aux and self.graph.in_degree(node) == 0:
                carried_aux = aux_used
                logging.info(f"Rollout root node {node} optimised aux="
                             f"{np.round(np.asarray(aux_used[0]).reshape(-1), 4).tolist()} (carried forward)")
            logging.info(f"Rollout node {node} state-in(traj0)={state_in} "
                         f"decision(traj0)={np.round(np.asarray(decision[0]).reshape(-1), 2).tolist()}")
            self._record_rollout_action(node, decision)
            steps.append(self._trajectory_step(node_input, decision, p_cons, n_cost, n))

            feasible = feasible & node_feasible
            if self.graph.out_degree(node) > 0:
                succ = list(self.graph.successors(node))[0]
                node_input = self.graph.edges[node, succ]["edge_fn"](outputs)
            cost = cost + jnp.asarray(n_cost).reshape(-1)
            logging.info(f"Rollout through node {node}: mean cost-so-far {float(jnp.mean(cost)):.4g}, "
                         f"feasible paths {int(jnp.sum(feasible))}/{n}")

        self._save_rollout_trajectories(steps, aux_trace)
        self._rollout_summary(cost, feasible, n)
        return self.graph

    def _rollout_seed(self, first_node, n):
        """Seed the chain with the true initial state so the root node decides from
        root_node_inputs (not zeros); downstream nodes inherit it via edge_fn."""
        root_in = self.cfg.model.root_node_inputs[first_node]
        if root_in in (None, 'None'):
            return None
        rs = jnp.asarray(root_in, dtype=jnp.float64).reshape(1, 1, -1)
        return jnp.broadcast_to(rs, (n, 1, rs.shape[-1]))

    def _record_rollout_action(self, node, decision):
        """Store the representative (trajectory-0) action for the policy plot."""
        vec = np.ravel(np.asarray(decision[0], dtype=float))
        cols = _get_rollout_action_columns(self.cfg, node, vec.shape[0])
        self.graph.nodes[node]["rollout_action"] = vec
        self.graph.nodes[node]["rollout_action_columns"] = cols
        self.graph.nodes[node]["rollout_action_named"] = {c: float(v) for c, v in zip(cols, vec)}

    def _trajectory_step(self, node_input, decision, p_cons, n_cost, n):
        """Per-node, per-trajectory (state, decision, constraints, cost) for the rollout plot."""
        state = (np.zeros((n, int(self.cfg.case_study.sizes.F_SIZE))) if node_input is None
                 else np.asarray(node_input))
        return {'state': state, 'decision': np.asarray(decision),
                'constraint': np.asarray(p_cons[:, 0, :]), 'cost': np.asarray(n_cost).reshape(-1)}

    def _save_rollout_trajectories(self, steps, aux_trace=None):
        """Stack the per-node steps into (N, n_nodes, .) tensors and save for the
        trajectory plot and the monolithic's decomposition warm start.  aux is the
        per-node trace, so a local_param's schedule is recoverable alongside it."""
        for key in ('state', 'decision', 'constraint'):
            if len({step[key].shape[1] for step in steps}) > 1:
                logging.info("Rollout trajectories not saved: per-node variable dimensions differ.")
                return
        aux = np.zeros((0, 0)) if not aux_trace else np.stack(aux_trace, axis=0)
        np.savez(
            f'rollout_trajectories_iterate_{self.iterate}.npz',
            states=np.stack([step['state'] for step in steps], axis=1),            # (N, n_nodes, F_SIZE)
            decisions=np.stack([step['decision'] for step in steps], axis=1),      # (N, n_nodes, n_design)
            constraints=np.stack([step['constraint'] for step in steps], axis=1),  # (N, n_nodes, n_g)
            costs=np.stack([step['cost'] for step in steps], axis=1),              # (N, n_nodes)
            aux=aux,                                                               # (n_nodes, n_aux)
        )
        logging.info("Saved %d rollout trajectories over %d nodes.", steps[0]['cost'].shape[0], len(steps))

    def _rollout_summary(self, cost, feasible, n):
        """Save the recovered policy and log the Monte-Carlo cost / feasibility."""
        cfg, graph = self.cfg, self.graph
        from mu_F.visualisation.methods import _rollout_policy_from_graph
        cols = list(cfg.case_study.design_space_dimensions)
        policy_vec = _rollout_policy_from_graph(cfg, graph)
        pd.DataFrame([dict(zip(cols, policy_vec))]).to_excel(f'rollout_policy_iterate_{self.iterate}.xlsx')
        logging.info(f"Saved rollout policy ({len(cols)}-d): {dict(zip(cols, policy_vec))}")

        costs = np.asarray(cost).reshape(-1)
        feas = np.asarray(feasible).reshape(-1)
        logging.info("Average cost-to-go (rollout, root to terminal): %.4g", float(costs.mean()))
        logging.info(
            "Rollout MC summary (N=%d): realised cost mean=%.4g std=%.4g "
            "[p10=%.4g median=%.4g p90=%.4g]; feasible paths=%d/%d (%.1f%%)",
            n, float(costs.mean()), float(costs.std()),
            float(np.quantile(costs, 0.1)), float(np.median(costs)), float(np.quantile(costs, 0.9)),
            int(feas.sum()), n, 100.0 * float(feas.mean()))
