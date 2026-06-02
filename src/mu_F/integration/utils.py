"""Shared helpers for the Decomposition run: data holder, action naming, and diagnostics."""
import io
import logging

from mu_F.utils import DatasetObject as dataset_holder


def update_data(data, *args):
    """Append to the growing data holder, creating it on the first call."""
    if data is None:
        return dataset_holder(*args)
    data.add(*args)
    return data


def _get_rollout_action_columns(cfg, node, n_actions):
    """
    Best-guess column names for a node's rollout action vector, preferring
    process-space names, then this node's design columns, then all design
    columns, falling back to positional labels.
    """
    process_names = cfg.case_study.process_space_names
    node_dims = process_names[node] if isinstance(process_names, (list, tuple)) else process_names
    if not isinstance(node_dims, (list, tuple)):
        node_dims = [node_dims]
    if len(node_dims) == n_actions:
        return [str(c) for c in node_dims]

    ds_cols = list(cfg.case_study.design_space_dimensions)
    node_ds_cols = [c for c in ds_cols if f"N{node+1}" in str(c)]
    if len(node_ds_cols) == n_actions:
        return [str(c) for c in node_ds_cols]
    if len(ds_cols) == n_actions:
        return [str(c) for c in ds_cols]

    return [f"node_{node}_action_{i}" for i in range(n_actions)]


class _StdoutToLog(io.TextIOBase):
    """Pipe writes into a logger so DEUS prints land in main.log.

    DEUS calls bare print() inside its solver loop; redirecting stdout into
    this TextIOBase routes the iteration trace into the hydra-managed log.

    """

    # ---- External Methods ----

    def __init__(self, logger):
        self.logger, self._buf = logger, ""
    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.logger.info(line.rstrip())
        return len(s)
    def flush(self):
        if self._buf.strip():
            self.logger.info(self._buf.rstrip())
        self._buf = ""


def _sqp_evaluators_for_node(graph, node):
    """
    Return (label, instance) for every SQP-running evaluator cached against
    (graph, node); each carries the base-class SQP counters.
    """
    from mu_F.constraints.evaluators.backward              import _BACKWARD_EVALUATOR_CACHE
    from mu_F.constraints.evaluators.cost_to_go            import _CTG_EVALUATOR_CACHE
    from mu_F.constraints.evaluators.probability           import _BACKWARDPMAP_EVALUATOR_CACHE
    from mu_F.constraints.evaluators.forward               import _FORWARD_EVALUATOR_CACHE
    from mu_F.constraints.evaluators.forward_decentralised import _FWDDEC_EVALUATOR_CACHE

    g = id(graph)
    return [(label, ev) for label, ev in (
        ('bwd-cls', _BACKWARD_EVALUATOR_CACHE.get((g, node, False))),
        ('bwd-CTG', _CTG_EVALUATOR_CACHE.get((g, node))),
        ('prob',    _BACKWARDPMAP_EVALUATOR_CACHE.get((g, node))),
        ('fwd',     _FORWARD_EVALUATOR_CACHE.get((g, node))),
        ('fwd-dec', _FWDDEC_EVALUATOR_CACHE.get((g, node))),
    ) if ev is not None]


def _log_sqp_convergence(graph, node):
    """
    Per-node summary of cumulative SQP feasibility and KKT-convergence across
    every evaluator in this node's DEUS sweep, one log line per evaluator.
    """
    for label, ev in _sqp_evaluators_for_node(graph, node):
        n_calls = int(ev.n_sqp_calls)
        if n_calls == 0:
            continue
        n_viable = int(ev.n_sqp_viable)
        n_conv   = int(ev.n_sqp_converged)
        v_rate = n_viable / n_calls
        c_rate = n_conv   / n_calls
        msg = (f"Node {node}: {label} — "
               f"feasible {n_viable}/{n_calls} ({v_rate*100:.1f}%), "
               f"converged {n_conv}/{n_calls} ({c_rate*100:.1f}%)")
        if v_rate < 0.95 or c_rate < 0.5:
            logging.warning(msg + " — consider bumping cfg.solvers.max_iter "
                                  "or cfg.solvers.n_starts.")
        else:
            logging.info(msg)
