"""Entry point: build the case study graph and dispatch to the chosen solver method."""
import os
import multiprocessing

# Force spawn before any other imports run.
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# Set JAX/Ray flags before JAX inits
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

import copy
import logging
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
import networkx as nx

"""
TODO :
- visualisation of probability maps
- extensive unit tests
- documentation
"""

def _set_log(level):
    """Set the logging level for the application."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(level=numeric_level, format="%(asctime)s [%(levelname)s] %(message)s")

def _set_jax(max_devices):
    import jax
    import os
    
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={max_devices}"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    jax.config.update('jax_platform_name', 'cpu')
    n_jax_devices = len(jax.devices())
    logging.info(f"Jax maximum devices set to: {n_jax_devices}")
    return n_jax_devices

_DECOMP_METHODS = {'decomposition', 'decomposition_constraint_tuner'}
_MONO_METHODS   = {'direct', 'single_shooting', 'multiple_shooting'}


def _select_solver_block(cfg: DictConfig) -> DictConfig:
    """
    Promote the method-specific solver block to cfg.solvers so downstream
    consumers read cfg.solvers.* flat.
    """
    if cfg.method in _DECOMP_METHODS:
        cfg.solvers = cfg.solvers.decomposition
    elif cfg.method in _MONO_METHODS:
        cfg.solvers = cfg.solvers.monolithic
    else:
        raise ValueError(
            f"Unknown method {cfg.method!r}; expected one of "
            f"{sorted(_DECOMP_METHODS | _MONO_METHODS)}"
        )
    return cfg


def _as_methods(method):
    """
    Normalise the configured method into a list so a single solve and a
    chained multi-method solve share one driver loop.
    """
    if isinstance(method, (list, ListConfig)):
        return [str(m) for m in method]
    return [str(method)]


def _select_integration_block(cfg: DictConfig) -> DictConfig:
    """
    Promote a method-specific integration override onto cfg.model.integration
    when present (e.g. fixed-step for the monolithic, adaptive for sampling).
    """
    key = 'integration_monolithic' if cfg.method in _MONO_METHODS else 'integration_decomposition'
    override = OmegaConf.select(cfg.model, key)
    if override is not None:
        cfg.model.integration = override
    return cfg


def _prepare_cfg(pristine, method):
    """
    Copy the pristine config, fix the scalar method, and promote that
    method's solver and integration blocks for the downstream consumers.
    """
    cfg = copy.deepcopy(pristine)
    cfg.method = method
    return _select_integration_block(_select_solver_block(cfg))


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

    total_devices = multiprocessing.cpu_count()
    cfg.max_devices = min(cfg.max_devices, total_devices)

    _set_log(cfg.log_level)
    methods = _as_methods(cfg.method)
    pristine = copy.deepcopy(cfg)
    max_devices = _set_jax(cfg.max_devices)

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