"""Entry point: build the case study graph and dispatch to the chosen solver method."""
import os
import multiprocessing

# Force spawn before any other imports run.
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

import copy
import logging
import hydra
from omegaconf import DictConfig
import networkx as nx

"""
TODO :
- visualisation of probability maps
- extensive unit tests
- documentation
"""


def _log_fn_evals(G):
    """Log the per-node function-evaluation tally after a solve."""
    for node in G.nodes():
        logging.info(f"Function evaluations for node {node}: {G.nodes[node]['fn_evals']}")


def _dispatch(cfg, G, max_devices):
    """
    Run the single solver named by cfg.method on the prepared graph and
    return the solved graph.
    """
    from mu_F.direct.constructor import apply_direct_method
    from mu_F.decomposition import Decomposition, decomposition_constraint_tuner
    from mu_F.utils import save_graph

    if cfg.method == 'decomposition':
        precedence_order = list(nx.topological_sort(G))
        return Decomposition(cfg, G, precedence_order, cfg.case_study.mode, max_devices).run()

    if cfg.method in ('direct', 'single_shooting', 'multiple_shooting'):
        outs = apply_direct_method(cfg, G, method=cfg.method)
        logging.info(f"Direct method {cfg.method} completed with outputs: {outs}")
        save_graph(G.copy(), f'{cfg.method}_complete')
        return G

    if cfg.method == 'decomposition_constraint_tuner':
        decomposition_constraint_tuner(cfg, G, max_devices)
        return G

    raise ValueError(f"Method not recognised: {cfg.method!r}")


def _run_method(pristine, method, max_devices):
    """
    Build a fresh graph from the pristine config and solve it with one
    method; returns the (config, solved graph) pair.
    """
    from mu_F.cs_assembly import case_study_constructor, make_markov
    from mu_F.utils import save_graph
    from mu_F.runtime import _prepare_cfg

    cfg = _prepare_cfg(pristine, method)
    G = case_study_constructor(cfg)
    cfg = make_markov(cfg)
    save_graph(G.copy(), f'initial_{method}')

    G = _dispatch(cfg, G, max_devices)
    _log_fn_evals(G)
    return cfg, G


def _plot_trajectories(solved):
    """
    Re-simulate the saved policies and plot the rollout (and monolithic)
    node-wise trajectories; a no-op when no policy was written.
    """
    from mu_F.visualisation.rollout_trajectory import plot_rollout_trajectory
    cfg, G = solved[0]
    plot_rollout_trajectory(cfg, G)


@hydra.main(config_path="config", config_name="integrator")
def main(cfg: DictConfig) -> None:
    import multiprocessing
    from mu_F.runtime import set_log, set_jax, _as_methods   # lazy: env must be set before jax is imported

    set_log(cfg.log_level)

    if str(cfg.device).lower() == "cpu":  # cap virtual CPU devices to core count
        cfg.max_devices = min(cfg.max_devices, multiprocessing.cpu_count())
    max_devices = set_jax(cfg.device, cfg.max_devices)
    cfg.max_devices = max_devices

    methods = _as_methods(cfg.method)
    pristine = copy.deepcopy(cfg)

    solved = []
    for method in methods:
        try:
            solved.append(_run_method(pristine, method, max_devices))
        except Exception:
            logging.exception(f"Method {method!r} failed; continuing with the remaining methods.")

    if solved:
        _plot_trajectories(solved)


if __name__ == "__main__":
    
    
    import sys

    #jax.config.update("jax_enable_x64", True)
    sys.path.append(os.path.join(os.getcwd(),'src'))
    # run the program
    main()
    print("Done")